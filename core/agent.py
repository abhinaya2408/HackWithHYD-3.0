"""The PulseMind agent.

One focused agent, no microservices and no multi-agent theatre. It owns the whole
product-intelligence loop:

    customer feedback -> analyse -> RETAIN in Hindsight
    new feedback      -> RECALL history -> connect to product changes -> compare outcomes
                      -> detect emerging problems -> recommend what to investigate
    PM decision       -> RETAIN in Hindsight -> future learning

The memory route is always the same: every insight, outcome, emerging issue and
decision that leaves this agent is written back with ``HindsightMemory.retain``, and
every explanation is built from what ``recall``/``reflect`` return. Nothing is cached
in a private store, so Memory Explorer shows the real contents of the memory bank.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from core.config import (
    MEMORY_KIND_LABELS,
    TAG_DECISION,
    TAG_EMERGING_ISSUE,
    TAG_FEEDBACK,
    TAG_OUTCOME,
    TAG_PRODUCT_CHANGE,
    THEME_LABELS,
    Settings,
    get_settings,
)
from core.groq_client import GroqLLM
from core.hindsight_client import HindsightMemory, MemoryHit, MemoryRecord, RetainReceipt
from services import feedback_analyzer as analyzer
from services import insight_engine as insights
from services import recommendation_engine as recommender
from services import trend_analyzer as trends

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """An answer to a Product Manager question, with the memory it used."""

    question: str
    text: str
    memories_used: list[MemoryHit] = field(default_factory=list)
    reflect_text: str = ""
    basis: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class IngestReport:
    """What a memory-ingestion run actually wrote into Hindsight."""

    scope: str
    receipts: list[RetainReceipt] = field(default_factory=list)

    @property
    def written(self) -> int:
        return sum(1 for receipt in self.receipts if receipt.ok)

    @property
    def failed(self) -> int:
        return sum(1 for receipt in self.receipts if not receipt.ok)

    def errors(self) -> list[str]:
        return [receipt.error or "unknown error" for receipt in self.receipts if not receipt.ok][:5]


class PulseMindAgent:
    """Memory-driven product intelligence agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        memory: HindsightMemory | None = None,
        llm: GroqLLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.memory = memory or HindsightMemory(self.settings)
        self.llm = llm or GroqLLM(self.settings)
        self._feedback_raw: pd.DataFrame | None = None
        self._frame: pd.DataFrame | None = None
        self._changes: pd.DataFrame | None = None
        # Populated by the UI so pages can explain setup problems instead of crashing.
        self.data_error: str | None = None
        self.memory_error: str | None = None
        # Monotonic timestamp of the last Hindsight write, used to pace free-tier calls.
        self._last_llm_write_at: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle and data
    # ------------------------------------------------------------------

    def start(self) -> "PulseMindAgent":
        self.memory.start()
        return self

    def close(self) -> None:
        self.memory.close()

    def reload_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._feedback_raw, self._frame, self._changes = analyzer.load_all(self.settings)
        return self._frame, self._changes

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self.reload_data()
        assert self._frame is not None
        return self._frame

    @property
    def changes(self) -> pd.DataFrame:
        if self._changes is None:
            self.reload_data()
        assert self._changes is not None
        return self._changes

    @property
    def raw_feedback(self) -> pd.DataFrame:
        if self._feedback_raw is None:
            self.reload_data()
        assert self._feedback_raw is not None
        return self._feedback_raw

    def periods(self) -> list[str]:
        return trends.periods(self.frame)

    # ------------------------------------------------------------------
    # RETAIN
    # ------------------------------------------------------------------

    # -- free-tier pacing / retry --------------------------------------
    #
    # Every `retain` makes Hindsight call the LLM provider, and its fact-extraction
    # prompt is ~1.7k tokens of fixed overhead per call. On Groq's free tier (8k
    # tokens/minute) back-to-back RETAINs get HTTP 429 and the retain is lost, so all
    # writes go through this one place: keep a minimum gap between them, and retry the
    # ones the provider says were rate limited using the delay it asks for.

    def _respect_pace(self) -> None:
        pace = self.settings.retain_pace_seconds
        if pace <= 0 or not self._last_llm_write_at:
            return
        remaining = pace - (time.monotonic() - self._last_llm_write_at)
        if remaining > 0:
            logger.info("pacing %.1fs before next Hindsight write", remaining)
            time.sleep(remaining)

    @staticmethod
    def _rate_limit_delay(error: str | None, default: float) -> float:
        """Wait as long as the provider itself asked for, when it says.

        The provider's retry hint is authoritative — waiting for our own pacing gap
        instead would either hammer the API (when the hint is longer) or waste time
        (when it is shorter). A one-second margin avoids retrying a moment too early,
        and the wait is capped so a bad hint cannot stall ingestion indefinitely.
        """
        if not error:
            return default
        match = re.search(r"try again in ([\d.]+)s", error)
        if match:
            return min(60.0, max(1.0, float(match.group(1)) + 1.0))
        return default

    @staticmethod
    def _is_rate_limited(error: str | None) -> bool:
        text = (error or "").lower()
        return "429" in text or "rate limit" in text or "quota" in text

    def _retain_document(self, document: analyzer.MemoryDocument, *, max_attempts: int = 3) -> RetainReceipt:
        """Retain one memory document with pacing and bounded rate-limit retry."""
        receipt = RetainReceipt(ok=False, error="not attempted")
        for attempt in range(1, max_attempts + 1):
            self._respect_pace()
            receipt = self.memory.retain(
                document.content,
                kind=document.kind,
                when=document.when,
                context=document.context,
                tags=document.tags,
                metadata=document.metadata,
                document_id=document.document_id,
            )
            self._last_llm_write_at = time.monotonic()
            if receipt.ok:
                return receipt
            if attempt < max_attempts and self._is_rate_limited(receipt.error):
                wait = self._rate_limit_delay(receipt.error, self.settings.retain_pace_seconds)
                logger.warning(
                    "retain rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            return receipt
        return receipt

    def _retain_documents(self, documents: Iterable[analyzer.MemoryDocument], scope: str) -> IngestReport:
        report = IngestReport(scope=scope)
        for document in documents:
            report.receipts.append(self._retain_document(document))
        return report

    def feedback_memory_documents(self, period: str | None = None) -> list[analyzer.MemoryDocument]:
        """One retainable memory per period (one Hindsight extraction call each)."""
        frame = self.frame
        selected = self.periods()
        if period:
            selected = [period]
        return [
            analyzer.build_period_feedback_memory(
                trends.filter_period(frame, current), period=current
            )
            for current in selected
        ]

    def ingest_feedback(self, period: str | None = None) -> IngestReport:
        """Analyse a period's feedback and retain it in Hindsight."""
        documents = self.feedback_memory_documents(period)
        scope = f"feedback {period}" if period else f"all feedback ({len(self.periods())} periods)"
        return self._retain_documents(documents, scope)

    def retain_product_change_memory(self, change: pd.Series | dict[str, Any]) -> RetainReceipt:
        """Retain an *already recorded* product change without touching the CSV.

        Used when (re)building demo memory, so re-running ingestion never duplicates
        rows in `data/product_changes.csv`.
        """
        return self._retain_document(analyzer.build_product_change_memory(change))

    def record_product_change(self, change: dict[str, Any]) -> tuple[str, RetainReceipt]:
        """Persist a PM-recorded product change and retain it in Hindsight."""
        path = analyzer.append_product_change(change, self.settings)
        self.reload_data()
        row = self.changes[self.changes["change_name"] == change.get("change_name")]
        record = row.iloc[-1] if not row.empty else self.changes.iloc[-1]
        return str(path), self._retain_document(analyzer.build_product_change_memory(record))

    def measure_outcome(
        self,
        change_name: str,
        *,
        theme: str | None = None,
        window_days: int = 28,
    ) -> tuple[trends.OutcomeComparison | None, RetainReceipt | None]:
        """Compare the situation before vs after a recorded change and retain the outcome."""
        matches = self.changes[self.changes["change_name"].str.lower() == change_name.lower()]
        if matches.empty:
            return None, None
        change = matches.iloc[-1]
        area = str(change["product_area"])

        comparison = trends.before_after(
            self.frame,
            change_date=change["date"],
            change_name=str(change["change_name"]),
            product_area=area,
            theme=theme or self.theme_for_area(area),
            window_days=window_days,
        )
        return comparison, self._retain_document(insights.build_outcome_memory(comparison))

    def recall_historical_context(
        self, theme: str, *, product_area: str | None = None, period: str | None = None, limit: int = 4
    ) -> list[str]:
        """Historical context for a topic, taken straight from Hindsight recall.

        Deliberately *not* an LLM summarisation step: the emerging-issue memory should
        carry the memory system's own words as its historical context, and skipping the
        extra reasoning call keeps the free-tier budget for work that needs it.
        """
        label = THEME_LABELS.get(theme, theme)
        area = f" in {product_area}" if product_area else ""
        before = f" before {period}" if period else ""
        hits = self.memory.recall(
            f"What did we know{before} about {label}{area}: earlier feedback, product "
            "changes, measured outcomes and past decisions?",
            budget="mid",
            max_tokens=2048,
        )
        return [f"{hit.text.strip()[:240]}" for hit in hits[:limit]]

    def retain_emerging_issue(self, theme: str, period: str | None = None) -> RetainReceipt:
        """Retain a detected emerging issue, with recall-derived historical context.

        No LLM reasoning call happens here on purpose: the signal itself is computed
        deterministically and the historical context comes from recall, so writing this
        memory costs exactly one Hindsight extraction call.
        """
        period = period or (self.periods()[-1] if self.periods() else "")
        signal = next(
            (sig for sig in trends.emerging_signals(self.frame, period) if sig.theme == theme), None
        )
        if signal is None:
            return RetainReceipt(ok=False, error="theme is not an emerging signal")
        context = self.recall_historical_context(
            theme, product_area=signal.product_area, period=period
        )
        document = insights.build_emerging_issue_memory(
            signal, self.frame, historical_context=context, period=period
        )
        return self._retain_document(document)

    def record_decision(
        self,
        recommendation: recommender.Recommendation,
        *,
        decision: str,
        reason: str,
        learning: str,
    ) -> RetainReceipt:
        """Retain the Product Manager's decision and the learning that follows it."""
        document = insights.build_decision_memory(
            recommendation=recommendation.title or recommendation.action,
            decision=decision,
            reason=reason,
            learning=learning,
            theme=recommendation.theme,
            product_area=recommendation.product_area,
        )
        return self._retain_document(document)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_period(self, period: str, *, use_memory: bool) -> insights.CurrentAnalysis:
        """Current-period analysis with or without long-term memory."""
        return insights.analyze_period(
            self.frame, period, self.llm, self.memory, use_memory=use_memory
        )

    def compare_memory_modes(self, period: str) -> tuple[insights.CurrentAnalysis, insights.CurrentAnalysis]:
        """Run the same period twice: without memory, then with Hindsight."""
        without = self.analyze_period(period, use_memory=False)
        with_memory = self.analyze_period(period, use_memory=True)
        return without, with_memory

    def emerging_signals(self, period: str | None = None) -> list[trends.EmergingSignal]:
        return trends.emerging_signals(self.frame, period)

    def build_insight(self, theme: str, period: str | None = None) -> insights.Insight:
        period = period or (self.periods()[-1] if self.periods() else "")
        return insights.build_insight(
            self.frame, self.changes, theme, self.llm, self.memory, period=period
        )

    def opportunities(self, period: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        return recommender.rank_opportunities(self.frame, self.changes, period=period, limit=limit)

    def recommend_next(self, period: str | None = None) -> recommender.Recommendation:
        """Rank candidates deterministically, then reason about the top one."""
        period = period or (self.periods()[-1] if self.periods() else "")
        ranked = self.opportunities(period, limit=1)
        if not ranked:
            return recommender.Recommendation(
                title="No candidates",
                action="No complaints are available to analyse.",
                why="",
                theme="",
                product_area="",
                confidence="low",
                error="No feedback data",
            )
        candidate = ranked[0]
        insight = None
        if candidate["direction"] in {"emerging", "improving"}:
            insight = self.build_insight(candidate["theme"], period)
        return recommender.recommend(
            candidate, self.llm, insight=insight, memory=self.memory, frame=self.frame, changes=self.changes
        )

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------

    def ask(self, question: str, *, use_reflect: bool = True) -> Answer:
        """Answer a PM question using recalled memory (+ Hindsight reflect)."""
        frame = self.frame
        period = self.periods()[-1] if self.periods() else ""
        context = [
            trends.render_period_brief(frame, current)
            for current in self.periods()[-3:]
        ]
        context.append("")
        context.append("Emerging/improving signals (deterministic):")
        for signal in self.emerging_signals(period)[:5]:
            context.append(
                f"  - {signal.direction}: {signal.theme} ({signal.product_area}) "
                f"{signal.latest_share:.1f}% vs {signal.baseline_share:.1f}% earlier "
                f"({signal.share_delta_pts:+.1f} points)"
            )
        if not self.changes.empty:
            context.append("")
            context.append("Product changes recorded by the Product Manager:")
            for _, row in self.changes.iterrows():
                context.append(
                    f"  - {row['change_name']} on {pd.Timestamp(row['date']).date().isoformat()} "
                    f"({row['product_area']}): targeted '{row['problem_targeted']}', "
                    f"expected '{row['expected_outcome']}'"
                )

        memories = self.memory.recall(question, budget="mid", max_tokens=3072)
        reflect_answer = self.memory.reflect(question, budget="low") if use_reflect else None

        context.append("")
        context.append("RECALLED FROM HINDSIGHT (long-term memory):")
        context.append(insights.render_memory_context(memories))
        if reflect_answer and reflect_answer.text:
            context.append("")
            context.append("HINDSIGHT REFLECT (the memory system's own synthesis):")
            context.append(reflect_answer.text)
            if reflect_answer.based_on:
                context.append("Reflect was based on:")
                for fact in reflect_answer.based_on[:8]:
                    context.append(f"  - {fact}")

        prompt = "\n".join(context)
        instruction = (
            f"The Product Manager asks: {question}\n\n"
            "Answer directly in 3-6 sentences. Use the deterministic numbers exactly as "
            "given. Ground every historical claim in the recalled memories or the reflect "
            "output, and name the product change and its measured outcome when they are "
            "relevant. If the memory does not contain the answer, say so explicitly."
        )

        try:
            text = self.llm.complete(
                prompt + "\n\n" + instruction,
                system=insights.ANALYST_SYSTEM,
                max_tokens=800,
                label=f"ask[{question[:40]}]",
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            text = (
                "Groq was unreachable, so here is what Hindsight recalled for your "
                "question:\n\n"
                + (reflect_answer.text if reflect_answer and reflect_answer.text else "(no reflect output)")
            )
            error = str(exc)

        return Answer(
            question=question,
            text=text,
            memories_used=memories,
            reflect_text=reflect_answer.text if reflect_answer else "",
            basis=reflect_answer.based_on if reflect_answer else [],
            metrics={
                "periods": self.periods(),
                "memories_recalled": len(memories),
                "reflect_based_on": len(reflect_answer.based_on) if reflect_answer else 0,
            },
            error=error,
        )

    def learnings(self) -> Answer:
        """'What have we learned?' — reflect over decisions, outcomes and history."""
        question = (
            "What have we learned so far? Summarise the product changes that were made, "
            "whether they worked, the decisions taken and what they taught us, and what "
            "that implies for the next investigation."
        )
        return self.ask(question)

    # ------------------------------------------------------------------
    # Memory inspection
    # ------------------------------------------------------------------

    def memory_overview(self) -> dict[str, Any]:
        """Real contents of the Hindsight bank, for Memory Explorer."""
        records, total = self.memory.list_memories(limit=500)
        counts = {kind: 0 for kind in MEMORY_KIND_LABELS}
        counts["unknown"] = 0
        for record in records:
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return {
            "total": total,
            "listed": len(records),
            "counts": counts,
            "labels": MEMORY_KIND_LABELS,
            "records": records,
            "status": self.memory.status(),
        }

    def list_memory_records(
        self,
        *,
        kind: str | None = None,
        fact_type: str | None = None,
        search_query: str | None = None,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        records, _ = self.memory.list_memories(
            kind=kind, fact_type=fact_type, search_query=search_query, limit=limit
        )
        return records

    def recall(
        self,
        query: str,
        *,
        kinds: list[str] | None = None,
        budget: str = "mid",
        max_tokens: int = 4096,
    ) -> list[MemoryHit]:
        return self.memory.recall(query, kinds=kinds, budget=budget, max_tokens=max_tokens)

    def change_retention(self) -> dict[str, dict[str, Any]]:
        """Map each recorded change onto the real Hindsight bank.

        Returns ``{key: {"retained": bool, "units": int}}`` keyed by change id (or the
        GitHub URL / change name when no id exists). Hindsight splits one retained
        document into one or more memory units, so this counts *units* whose metadata
        identifies the change — it is derived from `list_memories`, never assumed, and
        degrades to "not retained" when the bank is unavailable.
        """
        result: dict[str, dict[str, Any]] = {}
        if not self.memory.is_started:
            return result
        try:
            records, _ = self.memory.list_memories(limit=500)
        except Exception as exc:  # noqa: BLE001 - a status column must never crash the page
            logger.warning("change_retention could not list memories: %s", exc)
            return result

        by_change_id: dict[str, int] = {}
        by_github_url: dict[str, int] = {}
        for record in records:
            meta = getattr(record, "metadata", None) or {}
            change_id = str(meta.get("change_id", "") or "")
            if change_id:
                by_change_id[change_id] = by_change_id.get(change_id, 0) + 1
            url = str(meta.get("github_url", "") or meta.get("url", "") or "")
            if url:
                by_github_url[url] = by_github_url.get(url, 0) + 1

        for _, change in self.changes.iterrows():
            change_id = str(change.get("change_id", "") or "")
            url = str(change.get("github_url", "") or "")
            units = by_change_id.get(change_id, 0)
            if not units and url:
                units = by_github_url.get(url, 0)
            key = change_id or url or str(change.get("change_name", ""))
            result[key] = {"retained": units > 0, "units": units}
        return result

    def reset_memory(self) -> bool:
        """Delete and recreate the bank — an explicit, destructive demo control.

        Returns whether the deletion actually succeeded; the UI must not claim a reset
        happened when it did not.
        """
        deleted = self.memory.delete_bank()
        if deleted:
            self.memory.ensure_bank()
        return deleted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def theme_for_area(self, product_area: str) -> str | None:
        """Most common complaint topic for a product area (deterministic)."""
        neg = trends.negatives(self.frame)
        subset = neg[neg["product_area"] == product_area]
        if subset.empty:
            return None
        return str(subset["theme"].mode().iloc[0])

    def theme_label(self, theme: str) -> str:
        return THEME_LABELS.get(theme, theme)

    def status(self) -> dict[str, Any]:
        groq_ok, groq_message = (self.llm.health() if self.llm.configured else (False, "GROQ_API_KEY not set"))
        return {
            "memory": self.memory.status(),
            "groq_ok": groq_ok,
            "groq_message": groq_message,
            "groq_model": self.llm.model,
            "llm_calls": self.llm.usage.calls,
            "llm_tokens": self.llm.usage.total_tokens,
            "periods": self.periods(),
        }


__all__ = [
    "Answer",
    "IngestReport",
    "PulseMindAgent",
    "TAG_DECISION",
    "TAG_EMERGING_ISSUE",
    "TAG_FEEDBACK",
    "TAG_OUTCOME",
    "TAG_PRODUCT_CHANGE",
]
