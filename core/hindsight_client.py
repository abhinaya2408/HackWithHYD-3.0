"""Hindsight long-term memory for PulseMind.

Hindsight is the *only* memory layer in this project. There is no dictionary, JSON
file, SQLite table, Chroma/FAISS store or custom vector search standing in for it:
what this module writes with `retain` is what `recall`/`reflect` read back, and the
Memory Explorer renders those returned objects verbatim.

Three operations, exactly as the Hindsight Python client exposes them:

* ``retain``  - store feedback, product changes, outcomes, emerging issues and PM
  decisions. Hindsight runs an LLM over the content to extract facts, entities and
  temporal information.
* ``recall``  - multi-strategy retrieval (semantic + keyword + graph + temporal),
  optionally filtered by the tags PulseMind writes.
* ``reflect`` - deeper reasoning over the accumulated memories when a question
  needs synthesis rather than lookup.

Deployment: by default PulseMind starts a Hindsight server *in-process* with an
embedded PostgreSQL (``db_url="pg0"``) and Groq as that server's LLM provider. That
keeps the whole MVP on free tiers with no Docker and no cloud account. Setting
``HINDSIGHT_MODE=remote`` points the same code at an existing Hindsight server.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Callable, Iterable, TypeVar

from core.config import (
    MEMORY_KINDS,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalised view of what Hindsight returns
# ---------------------------------------------------------------------------


@dataclass
class MemoryHit:
    """One recalled memory (a Hindsight `RecallResult`)."""

    id: str
    text: str
    kind: str = "unknown"
    fact_type: str | None = None
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """One stored memory unit as listed by `list_memories` (Memory Explorer)."""

    id: str
    text: str
    kind: str = "unknown"
    fact_type: str | None = None
    context: str | None = None
    occurred_start: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    proof_count: int | None = None
    state: str | None = None
    entities: list[str] = field(default_factory=list)


@dataclass
class ReflectAnswer:
    """Result of a `reflect` call: the answer plus the facts it was based on."""

    text: str
    based_on: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetainReceipt:
    """What a retain call actually did, so the UI can show real memory writes."""

    ok: bool
    items_count: int = 0
    document_id: str | None = None
    tags: list[str] = field(default_factory=list)
    error: str | None = None
    # Hindsight reports the provider tokens its extraction pass used. Kept here so the
    # free-tier cost of writing memory is measurable rather than guessed.
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", 0) or 0)


def _as_iso(value: Any) -> str | None:
    """Render Hindsight's date-ish fields as a string without losing info."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def kind_from_tags(tags: Iterable[str] | None) -> str:
    """Recover PulseMind's memory kind from the Hindsight tags on a memory."""
    for tag in tags or []:
        if tag in MEMORY_KINDS:
            return tag
    return "unknown"


T = TypeVar("T")


def _to_aware_datetime(value: str | date | datetime | None) -> datetime | None:
    """Hindsight wants a real datetime; the dataset stores plain dates."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class HindsightMemory:
    """PulseMind's connection to its Hindsight memory bank."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._server: Any = None
        self._client: Any = None
        self._bank_ready = False
        self.last_error: str | None = None
        self.endpoint: str | None = None
        self._owner: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # One thread owns the client
    # ------------------------------------------------------------------

    def _owner_thread(self) -> ThreadPoolExecutor:
        """The single thread that owns the Hindsight client (created on demand)."""
        if self._owner is None:
            self._owner = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hindsight-memory"
            )
        return self._owner

    def _call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run one Hindsight client call on the thread that owns the client.

        The generated client keeps a single ``aiohttp.ClientSession`` bound to the
        asyncio loop of whichever thread first used it. Reusing that session from
        another thread raises ``Timeout context manager should be used inside a
        task`` (the request's timeout looks for a task in a loop that is not the
        one running it), which is what broke webhook retains: ``ThreadingHTTPServer``
        handles each delivery on a fresh thread while the Streamlit app created the
        session on the main one.

        Funnelling every call through one worker keeps all memory I/O on the loop
        that owns the session, so retain/recall/reflect are safe from any thread.
        """
        return self._owner_thread().submit(fn, *args, **kwargs).result()

    def _stop_owner(self) -> None:
        owner, self._owner = self._owner, None
        if owner is not None:
            owner.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def bank_id(self) -> str:
        return self.settings.bank_id

    @property
    def is_started(self) -> bool:
        return self._client is not None

    # The Groq model Hindsight itself calls for extraction and consolidation.
    @property
    def llm_model(self) -> str:
        return self.settings.hindsight_llm_model

    def start(self, timeout: float = 180.0) -> "HindsightMemory":
        """Bring up the memory backend and make sure the bank exists."""
        if self._client is not None:
            return self

        if not self.settings.has_groq_key:
            raise RuntimeError(
                "Hindsight needs an LLM provider for memory extraction. Set "
                "GROQ_API_KEY in .env (free key: https://console.groq.com/keys)."
            )

        if self.settings.is_embedded:
            # Imported lazily so the app can still run tests/UI that don't touch memory.
            from hindsight import HindsightServer

            self._server = HindsightServer(
                db_url="pg0",  # embedded PostgreSQL, no Docker / no cloud
                llm_provider=self.settings.hindsight_llm_provider,
                llm_api_key=self.settings.groq_api_key,
                # Hindsight's own model, separate from the reasoning model: Groq rate
                # limits are per model, so memory extraction gets its own TPM budget.
                llm_model=self.settings.hindsight_llm_model,
                log_level="warning",
            )
            self._server.start(timeout=timeout)
            self.endpoint = self._server.url
        else:
            self.endpoint = self.settings.hindsight_base_url

        from hindsight_client import Hindsight

        self._client = Hindsight(
            base_url=self.endpoint or "http://localhost:8888",
            api_key=self.settings.hindsight_api_key or None,
        )
        self._wait_until_ready(timeout=timeout)
        self.ensure_bank()
        return self

    def _wait_until_ready(self, timeout: float = 180.0) -> None:
        """Poll the API until it answers.

        The embedded server starts uvicorn before the database migrations finish,
        and the very first run also has to provision the embedded PostgreSQL, so a
        successful socket bind is not proof that memory is usable yet.
        """
        import time as time_module

        deadline = time_module.time() + timeout
        last_exc: Exception | None = None
        while time_module.time() < deadline:
            try:
                self._call(self._client.get_version)
                return
            except Exception as exc:  # noqa: BLE001 - report the real cause later
                last_exc = exc
                time_module.sleep(1.0)
        raise RuntimeError(f"Hindsight did not become ready: {last_exc}")

    def ensure_bank(self) -> None:
        """Create the bank once, with a mission that shapes memory extraction."""
        if self._bank_ready or self._client is None:
            return
        try:
            self._call(
                self._client.create_bank,
                bank_id=self.bank_id,
                name="PulseMind product intelligence",
                mission=(
                    "Long-term memory for a Product Manager. Retain customer "
                    "feedback themes, product changes, the outcomes of those "
                    "changes, emerging issues and the decisions taken. Keep dates, "
                    "product areas, themes and numbers attached to every fact so "
                    "they can be compared across time."
                ),
                reflect_mission=(
                    "Reason like a product intelligence analyst: connect current "
                    "feedback to earlier feedback, product changes, outcomes and "
                    "past decisions. Be explicit about what the memory does and "
                    "does not support."
                ),
                enable_observations=True,
            )
        except Exception as exc:  # noqa: BLE001 - "already exists" is the normal path
            logger.debug("create_bank returned %s (usually already exists)", exc)
        self._bank_ready = True

    def delete_bank(self) -> bool:
        """Delete this bank on the client's owner thread and report whether it happened.

        The generated client must be used from the thread that owns it (its aiohttp
        session is loop-bound), so the deletion goes through ``_call`` like every
        other operation. Callers previously hit the client directly from the UI thread
        and got ``RuntimeError: Timeout context manager should be used inside a task``,
        which was swallowed — the UI reported a reset that never happened.
        """
        if self._client is None:
            return False
        try:
            self._call(self._client.delete_bank, self.bank_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller, never hidden
            logger.warning("delete_bank failed: %s", exc)
            self.last_error = str(exc)
            return False
        self._bank_ready = False
        return True

    def close(self) -> None:
        try:
            if self._client is not None:
                self._call(self._client.close)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._client = None
            self._bank_ready = False
            self._stop_owner()
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._server = None

    def check_connection(self) -> bool:
        """One real, read-only round-trip that answers "can we reach memory right now?".

        `is_started` only reports that a client object exists, and `list_memories`
        deliberately swallows read errors (it returns an empty list so the UI keeps
        working), so neither can distinguish "nothing stored" from "backend
        unreachable". Without an explicit probe the UI would be claiming a check it
        never performed, so this performs one: `get_version` is the cheapest
        side-effect-free call the client exposes.
        """
        if self._client is None:
            return False
        try:
            self._call(self._client.get_version)
        except Exception as exc:  # noqa: BLE001 - the caller renders the outcome
            self.last_error = str(exc)
            logger.warning("memory connection check failed: %s", exc)
            return False
        return True

    def status(self) -> dict[str, Any]:
        return {
            "mode": "embedded" if self.settings.is_embedded else "remote",
            "endpoint": self.endpoint,
            "bank_id": self.bank_id,
            "llm_provider": self.settings.hindsight_llm_provider,
            "llm_model": self.settings.hindsight_llm_model,
            "reasoning_model": self.settings.groq_model,
            "started": self.is_started,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    # Write: RETAIN
    # ------------------------------------------------------------------

    def retain(
        self,
        content: str,
        *,
        kind: str,
        when: str | date | datetime | None = None,
        context: str | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> RetainReceipt:
        """Store one PulseMind memory in Hindsight.

        `kind` becomes both a Hindsight tag (so recall can filter by memory type)
        and part of the metadata, which is what makes Memory Explorer able to show
        feedback next to product changes next to decisions.
        """
        if self._client is None:
            return RetainReceipt(ok=False, error="Hindsight is not started")

        all_tags = [kind, *[str(t) for t in (tags or [])]]
        merged_metadata = {"kind": kind}
        merged_metadata.update({str(k): str(v) for k, v in (metadata or {}).items()})
        if when is not None:
            merged_metadata.setdefault("memory_date", _as_iso(when) or "")

        try:
            response = self._call(
                self._client.retain,
                bank_id=self.bank_id,
                content=content,
                timestamp=_to_aware_datetime(when),
                context=context,
                document_id=document_id,
                metadata=merged_metadata,
                tags=all_tags,
            )
            self.last_error = None
            usage = getattr(response, "usage", None)
            usage_dict: dict[str, Any] = {}
            if usage is not None:
                usage_dict = (
                    usage.model_dump()
                    if hasattr(usage, "model_dump")
                    else dict(usage)
                    if isinstance(usage, dict)
                    else {}
                )
            return RetainReceipt(
                ok=bool(getattr(response, "success", True)),
                items_count=int(getattr(response, "items_count", 0) or 0),
                document_id=document_id,
                tags=all_tags,
                usage=usage_dict,
            )
        except Exception as exc:  # noqa: BLE001 - surface the real error in the UI
            self.last_error = str(exc)
            logger.warning("retain failed: %s", exc)
            return RetainReceipt(ok=False, error=str(exc), tags=all_tags)

    # ------------------------------------------------------------------
    # Read: RECALL
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        budget: str = "mid",
        max_tokens: int = 4096,
        include_chunks: bool = False,
    ) -> list[MemoryHit]:
        """Retrieve memories relevant to `query`.

        `kinds` are PulseMind's memory kinds and are applied as Hindsight tag
        filters, so "only what we did before" is a real server-side filter.
        """
        if self._client is None:
            return []

        tag_filter: list[str] = [str(t) for t in (kinds or [])] + [
            str(t) for t in (tags or [])
        ]
        try:
            response = self._call(
                self._client.recall,
                bank_id=self.bank_id,
                query=query,
                budget=budget,
                max_tokens=max_tokens,
                include_chunks=include_chunks,
                tags=tag_filter or None,
                tags_match="any",
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            logger.warning("recall failed: %s", exc)
            return []

        hits: list[MemoryHit] = []
        for result in getattr(response, "results", None) or []:
            tags_found = list(getattr(result, "tags", None) or [])
            hits.append(
                MemoryHit(
                    id=str(getattr(result, "id", "")),
                    text=str(getattr(result, "text", "")),
                    kind=kind_from_tags(tags_found),
                    fact_type=getattr(result, "type", None),
                    context=getattr(result, "context", None),
                    occurred_start=_as_iso(getattr(result, "occurred_start", None)),
                    occurred_end=_as_iso(getattr(result, "occurred_end", None)),
                    tags=tags_found,
                    metadata=dict(getattr(result, "metadata", None) or {}),
                    scores=dict(getattr(result, "scores", None) or {}),
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Reason: REFLECT
    # ------------------------------------------------------------------

    def reflect(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        budget: str = "low",
        include_facts: bool = True,
        max_tokens: int | None = None,
    ) -> ReflectAnswer:
        """Ask Hindsight to reason over everything it has retained."""
        if self._client is None:
            return ReflectAnswer(text="", based_on=[])

        tag_filter = [str(t) for t in (kinds or [])]
        try:
            response = self._call(
                self._client.reflect,
                bank_id=self.bank_id,
                query=query,
                budget=budget,
                include_facts=include_facts,
                max_tokens=max_tokens,
                tags=tag_filter or None,
                tags_match="any",
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            logger.warning("reflect failed: %s", exc)
            return ReflectAnswer(text="", based_on=[])

        based_on: list[str] = []
        basis = getattr(response, "based_on", None)
        raw_facts = getattr(basis, "memories", None) if basis is not None else None
        for fact in raw_facts or []:
            text = getattr(fact, "text", None) or str(fact)
            if text:
                based_on.append(str(text))

        return ReflectAnswer(
            text=str(getattr(response, "text", "") or ""),
            based_on=based_on,
            usage=dict(getattr(response, "usage", None) or {}),
        )

    # ------------------------------------------------------------------
    # Inspect: LIST
    # ------------------------------------------------------------------

    def list_memories(
        self,
        *,
        kind: str | None = None,
        fact_type: str | None = None,
        search_query: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[MemoryRecord], int]:
        """List the raw memory units Hindsight holds (total, filtered locally by kind).

        Hindsight can filter by fact type server-side; PulseMind's own memory kinds
        live in tags, so those are filtered here over the objects Hindsight returns.
        """
        if self._client is None:
            return [], 0

        try:
            response = self._call(
                self._client.list_memories,
                bank_id=self.bank_id,
                type=fact_type,
                search_query=search_query,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            logger.warning("list_memories failed: %s", exc)
            return [], 0

        records: list[MemoryRecord] = []
        for item in getattr(response, "items", None) or []:
            tags_found = list(getattr(item, "tags", None) or [])
            item_kind = kind_from_tags(tags_found)
            if kind and item_kind != kind:
                continue
            entities = [
                str(getattr(entity, "name", None) or entity)
                for entity in (getattr(item, "entities", None) or [])
            ]
            records.append(
                MemoryRecord(
                    id=str(getattr(item, "id", "")),
                    text=str(getattr(item, "text", "")),
                    kind=item_kind,
                    fact_type=getattr(item, "fact_type", None),
                    context=getattr(item, "context", None),
                    occurred_start=_as_iso(
                        getattr(item, "occurred_start", None)
                        or getattr(item, "var_date", None)
                    ),
                    tags=tags_found,
                    metadata=dict(getattr(item, "metadata", None) or {}),
                    proof_count=getattr(item, "proof_count", None),
                    state=getattr(item, "state", None),
                    entities=entities,
                )
            )
        total = int(getattr(response, "total", len(records)) or 0)
        return records, total

    def kind_counts(self, limit: int = 500) -> dict[str, int]:
        """How many memory units of each kind Hindsight currently holds."""
        records, _ = self.list_memories(limit=limit)
        counts = {kind: 0 for kind in MEMORY_KINDS}
        counts["unknown"] = 0
        for record in records:
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return counts
