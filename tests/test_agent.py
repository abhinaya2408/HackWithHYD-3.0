"""Agent orchestration.

These tests use doubles for Hindsight and Groq so they run instantly and offline,
while still exercising the real memory and document-building code paths. They assert
the property this project is about: with memory, the agent's context contains recalled
history; without it, the same metrics are handed over with no history at all.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from core.agent import PulseMindAgent
from core.config import Settings, get_settings
from core.groq_client import LLMUsage
from core.hindsight_client import (
    HindsightMemory,
    MemoryHit,
    MemoryRecord,
    ReflectAnswer,
    RetainReceipt,
    _to_aware_datetime,
    kind_from_tags,
)
from services import feedback_analyzer as analyzer
from services import insight_engine as insights
from services import trend_analyzer as trends


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeMemory:
    """In-memory stand-in with the same surface the agent uses."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.endpoint = "http://fake-hindsight:8888"
        self.last_error: str | None = None
        self.is_started = True
        self.retained: list[dict] = []
        self.recall_queries: list[str] = []
        self.reflect_queries: list[str] = []

    # lifecycle
    def start(self, timeout: float = 180.0) -> "FakeMemory":
        self.is_started = True
        return self

    def ensure_bank(self) -> None: ...

    def close(self) -> None:
        self.is_started = False

    def check_connection(self) -> bool:
        """Mirror the real client's reachability probe."""
        return self.is_started

    @property
    def bank_id(self) -> str:
        return self.settings.bank_id

    def status(self) -> dict:
        return {
            "mode": "embedded",
            "endpoint": self.endpoint,
            "bank_id": self.bank_id,
            "llm_provider": "groq",
            "llm_model": self.settings.groq_model,
            "started": self.is_started,
            "last_error": self.last_error,
        }

    # write
    def retain(self, content, *, kind, when=None, context=None, tags=None, metadata=None, document_id=None):
        self.retained.append(
            {
                "content": content,
                "kind": kind,
                "when": when,
                "context": context,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
                "document_id": document_id,
            }
        )
        return RetainReceipt(ok=True, items_count=1, document_id=document_id, tags=list(tags or []))

    # read
    def recall(self, query, *, kinds=None, tags=None, budget="mid", max_tokens=4096, include_chunks=False):
        self.recall_queries.append(query)
        wanted = set(kinds or [])
        hits = []
        for index, item in enumerate(self.retained):
            if wanted and item["kind"] not in wanted:
                continue
            hits.append(
                MemoryHit(
                    id=f"mem-{index}",
                    text=item["content"][:200],
                    kind=item["kind"],
                    fact_type="world",
                    tags=item["tags"],
                    metadata={key: str(value) for key, value in item["metadata"].items()},
                    scores={"semantic": 0.91},
                )
            )
        return hits

    def reflect(self, query, *, kinds=None, budget="low", include_facts=True, max_tokens=None):
        self.reflect_queries.append(query)
        return ReflectAnswer(
            text="Reflect synthesis: checkout improved after Checkout V2, UPI is the open risk.",
            based_on=[item["content"][:80] for item in self.retained[:3]],
            usage={"total_tokens": 42},
        )

    def list_memories(self, *, kind=None, fact_type=None, search_query=None, limit=200, offset=0):
        records = []
        for index, item in enumerate(self.retained):
            record_kind = kind_from_tags(item["tags"])
            if kind and record_kind != kind:
                continue
            records.append(
                MemoryRecord(
                    id=f"mem-{index}",
                    text=item["content"][:200],
                    kind=record_kind,
                    fact_type="world",
                    tags=item["tags"],
                    metadata={key: str(value) for key, value in item["metadata"].items()},
                    proof_count=1,
                    state="active",
                )
            )
        return records[offset : offset + limit], len(records)


class FakeLLM:
    """Returns deterministic payloads shaped like the real prompts expect."""

    def __init__(self, model: str = "fake/groq-model") -> None:
        self.model = model
        self.configured = True
        self.usage = LLMUsage()
        self.prompts: dict[str, str] = {}
        self.last_error: str | None = None

    def complete(self, prompt, *, system=None, temperature=0.2, max_tokens=1200, json_mode=False, label="completion", max_retries=3):
        self.prompts[label] = prompt
        self.usage.calls += 1
        if label.startswith("ask["):
            return "Checkout complaints fell after Checkout V2; UPI reliability is now the risk."
        return f"[{label}] response"

    def complete_json(self, prompt, *, system=None, temperature=0.1, max_tokens=1600, label="json"):
        self.prompts[label] = prompt
        self.usage.calls += 1
        if label.startswith("analyze["):
            return {"narrative": f"Analysis for {label}."}
        if label.startswith("insight["):
            return {
                "narrative": "Checkout improved; the history supports the improvement.",
                "trend_text": "checkout_friction shares fell from 39.1% to 14.6%.",
                "sentiment_change": "Negative share is steady; average rating 1.56/5.",
                "historical_context": ["January: 18 checkout complaints.", "Checkout V2 shipped 2026-02-15."],
            }
        if label.startswith("recommend["):
            return {
                "title": "Investigate UPI payment reliability",
                "action": "Trace UPI failure codes for the last 30 days.",
                "why": "Payment complaints triple while checkout complaints fall after Checkout V2.",
                "confidence": "high",
                "what_to_check": ["UPI failure rate", "Bank-specific errors"],
                "supporting_evidence": ["Early UPI failures were noted in November 2025."],
            }
        return {}

    def complete_json_safe(self, prompt, *, system=None, fallback=None, temperature=0.1, max_tokens=1600, label="json"):
        try:
            return self.complete_json(prompt, system=system, temperature=temperature, max_tokens=max_tokens, label=label)
        except Exception:  # noqa: BLE001
            return fallback

    def health(self):
        return True, "fake Groq reachable"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Real settings pointed at a throwaway copy of the dataset.

    Recording a product change writes to product_changes.csv, so tests must never use
    the committed dataset in place.
    """
    source = get_settings()
    for name in ["feedback.csv", "product_changes.csv"]:
        shutil.copy(source.data_dir / name, tmp_path / name)
    # Pacing off: it exists to protect the free tier at runtime, not to slow tests down.
    return dataclasses.replace(
        source, data_dir=tmp_path, groq_api_key="test-key", retain_pace_seconds=0.0
    )


@pytest.fixture()
def agent(settings: Settings) -> PulseMindAgent:
    fake_memory = FakeMemory(settings)
    return PulseMindAgent(settings, memory=fake_memory, llm=FakeLLM())


# ---------------------------------------------------------------------------
# Retain path
# ---------------------------------------------------------------------------


def test_ingest_feedback_retains_exactly_one_memory_per_period(agent: PulseMindAgent) -> None:
    """One RETAIN per period = one Hindsight extraction call per period."""
    documents = agent.feedback_memory_documents("2026-01")
    assert len(documents) == 1

    report = agent.ingest_feedback("2026-01")
    assert report.failed == 0 and report.written == 1
    assert len(agent.memory.retained) == 1
    retained = agent.memory.retained[0]
    assert retained["kind"] == "feedback"
    assert retained["document_id"] == "feedback-2026-01"
    assert all(isinstance(value, str) for value in retained["metadata"].values())


def test_ingest_feedback_covers_every_period_with_one_call_each(agent: PulseMindAgent) -> None:
    report = agent.ingest_feedback()
    assert report.written == len(agent.periods())
    assert report.failed == 0


def test_record_product_change_persists_and_retains_in_memory(agent: PulseMindAgent) -> None:
    path, receipt = agent.record_product_change(
        {
            "change_id": "",
            "date": "2026-04-05",
            "change_name": "UPI Retry Flow",
            "product_area": "Payments",
            "problem_targeted": "Payment reliability (UPI failures and pending transactions)",
            "expected_outcome": "Reduce failed UPI payments",
        }
    )
    assert receipt.ok
    saved = agent.changes
    assert "UPI Retry Flow" in set(saved["change_name"])
    assert Path(path).exists()
    retained = agent.memory.retained[-1]
    assert retained["kind"] == "product_change"
    assert "UPI Retry Flow" in retained["content"]
    assert retained["metadata"]["release_date"] == "2026-04-05"


def test_measure_outcome_retains_an_outcome_memory(agent: PulseMindAgent) -> None:
    comparison, receipt = agent.measure_outcome("Checkout V2")
    assert comparison is not None and receipt is not None and receipt.ok
    assert comparison.verdict == "improved"
    retained = agent.memory.retained[-1]
    assert retained["kind"] == "outcome"
    assert "Situation before the change" in retained["content"]
    assert retained["metadata"]["verdict"] == "improved"


def test_record_decision_retains_the_learning(agent: PulseMindAgent) -> None:
    from services.recommendation_engine import Recommendation

    recommendation = Recommendation(
        title="Investigate UPI payment reliability",
        action="Trace UPI failure codes.",
        why="Payment complaints are rising.",
        theme="payment_reliability",
        product_area="Payments",
        confidence="high",
    )
    receipt = agent.record_decision(
        recommendation,
        decision="accepted",
        reason="Payment failures have a direct revenue impact.",
        learning="Check payment reliability after every checkout change.",
    )
    assert receipt.ok
    retained = agent.memory.retained[-1]
    assert retained["kind"] == "product_manager_decision"
    assert "accepted" in retained["content"]
    assert "direct revenue impact" in retained["content"]
    assert retained["metadata"]["learning"].startswith("Check payment reliability")


def test_retain_emerging_issue_only_stores_real_signals(agent: PulseMindAgent) -> None:
    # History must exist first — the memory's context is taken from Hindsight recall.
    agent.ingest_feedback("2026-01")
    receipt = agent.retain_emerging_issue("payment_reliability", "2026-03")
    assert receipt.ok
    retained = agent.memory.retained[-1]
    assert retained["kind"] == "emerging_issue"
    assert "Historical context recalled from memory:" in retained["content"]
    # The context section must hold text that actually came back from recall, i.e. the
    # retained January period document, not a placeholder.
    assert "period 2026-01" in retained["content"]

    skipped = agent.retain_emerging_issue("search_relevance", "2026-03")
    assert not skipped.ok, "a theme that is not an emerging signal must not be retained"


def test_retain_emerging_issue_does_not_spend_a_reasoning_call(agent: PulseMindAgent) -> None:
    """The retain path must not add a Groq reasoning call on top of Hindsight's own."""
    agent.ingest_feedback("2026-01")
    before = agent.llm.usage.calls
    agent.retain_emerging_issue("payment_reliability", "2026-03")
    assert agent.llm.usage.calls == before, "retaining an emerging issue must not call Groq"


def test_historical_context_comes_from_recall(agent: PulseMindAgent) -> None:
    agent.ingest_feedback("2026-01")
    context = agent.recall_historical_context(
        "payment_reliability", product_area="Payments", period="2026-03"
    )
    assert context, "expected recalled context after ingesting a period"
    assert all(isinstance(item, str) and item.strip() for item in context)
    assert agent.memory.recall_queries, "context must come from Hindsight recall"


# ---------------------------------------------------------------------------
# Recall path — the memory difference
# ---------------------------------------------------------------------------


def test_with_memory_context_contains_recalled_history_and_without_does_not(
    agent: PulseMindAgent,
) -> None:
    agent.ingest_feedback("2026-01")
    agent.record_product_change(
        {
            "change_id": "CHG-001",
            "date": "2026-02-15",
            "change_name": "Checkout V2",
            "product_area": "Checkout",
            "problem_targeted": "Checkout friction",
            "expected_outcome": "Reduce checkout complaints",
        }
    )

    without = agent.analyze_period("2026-03", use_memory=False)
    with_memory = agent.analyze_period("2026-03", use_memory=True)

    assert without.memories_used == []
    assert with_memory.memories_used, "with-memory analysis must recall memories"
    assert "RECALLED FROM HINDSIGHT" not in without.prompt_context
    assert "RECALLED FROM HINDSIGHT" in with_memory.prompt_context
    assert "Checkout V2" in with_memory.prompt_context
    assert without.mode == "without_memory" and with_memory.mode == "with_memory"
    # Both runs still use the same deterministic metrics for numbers.
    assert "checkout_friction" in without.prompt_context
    assert "checkout_friction" in with_memory.prompt_context


def test_build_insight_links_change_outcome_and_history(agent: PulseMindAgent) -> None:
    insight = agent.build_insight("checkout_friction", "2026-03")
    assert insight.theme == "checkout_friction"
    assert insight.related_change is not None
    assert insight.related_change["change_name"] == "Checkout V2"
    assert insight.outcome is not None and insight.outcome["verdict"] == "improved"
    assert insight.historical_context
    assert insight.metrics["counts_by_period"]["2026-01"] > insight.metrics["counts_by_period"]["2026-03"]


def test_ask_uses_recall_and_reflect(agent: PulseMindAgent) -> None:
    agent.ingest_feedback("2026-01")
    answer = agent.ask("Have we seen UPI issues before?")
    assert answer.text
    assert answer.memories_used, "recall must contribute memories"
    assert answer.reflect_text, "reflect output must be surfaced"
    assert answer.basis
    assert agent.memory.reflect_queries, "reflect must actually be called"
    assert "RECALLED FROM HINDSIGHT" in agent.llm.prompts[[k for k in agent.llm.prompts if k.startswith("ask[")][-1]]


def test_learnings_asks_a_reflective_question(agent: PulseMindAgent) -> None:
    answer = agent.learnings()
    assert "learned" in answer.question.lower()
    assert answer.reflect_text


def test_memory_overview_counts_by_kind(agent: PulseMindAgent) -> None:
    agent.ingest_feedback("2026-01")
    agent.measure_outcome("Checkout V2")
    overview = agent.memory_overview()
    assert overview["counts"]["feedback"] > 0
    assert overview["counts"]["outcome"] == 1
    assert overview["total"] == len(overview["records"])
    assert overview["labels"]["feedback"] == "Customer feedback"


# ---------------------------------------------------------------------------
# Recommendation path
# ---------------------------------------------------------------------------


def test_opportunities_rank_payment_reliability_top(agent: PulseMindAgent) -> None:
    ranked = agent.opportunities("2026-03")
    assert ranked, "expected at least one opportunity"
    assert ranked[0]["theme"] == "payment_reliability"
    assert ranked[0]["direction"] == "emerging"
    assert ranked[0]["share_delta_pts"] > 0
    scores = [item["score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_recommend_next_returns_grounded_recommendation(agent: PulseMindAgent) -> None:
    recommendation = agent.recommend_next("2026-03")
    assert recommendation.theme == "payment_reliability"
    assert "UPI" in recommendation.title
    assert recommendation.confidence == "high"
    assert recommendation.supporting_evidence
    assert recommendation.metrics["complaints"] > 0


def test_opportunities_shortlist_is_bounded(agent: PulseMindAgent) -> None:
    assert len(agent.opportunities("2026-03", limit=3)) == 3


def test_opportunities_carry_the_period_for_downstream_evidence(agent: PulseMindAgent) -> None:
    """The recommendation prompt needs the period to pull the latest verbatims."""
    ranked = agent.opportunities("2026-03")
    assert ranked and all(item["period"] == "2026-03" for item in ranked)


def test_recommendation_prompt_includes_latest_verbatims(agent: PulseMindAgent) -> None:
    agent.recommend_next("2026-03")
    prompts = [value for key, value in agent.llm.prompts.items() if key.startswith("recommend[")]
    assert prompts, "the recommendation prompt should have been built"
    assert "Latest verbatim" in prompts[-1]
    assert "RECALLED FROM HINDSIGHT" in prompts[-1]


def test_agent_status_reports_memory_and_llm(agent: PulseMindAgent) -> None:
    status = agent.status()
    assert status["memory"]["bank_id"] == agent.settings.bank_id
    assert status["groq_ok"] is True
    assert status["periods"] == agent.periods()


def test_every_memory_document_type_has_a_parseable_date(agent: PulseMindAgent) -> None:
    """Hindsight parses `when` into a datetime, so a period key like '2026-03' fails.

    Regression test: the emerging-issue memory used to pass its `YYYY-MM` period key
    as the timestamp and Hindsight rejected the retain with
    `Invalid isoformat string: '2026-03'`.
    """
    documents = list(agent.feedback_memory_documents())
    documents += [
        analyzer.build_product_change_memory(row) for _, row in agent.changes.iterrows()
    ]
    for _, change in agent.changes.iterrows():
        comparison = trends.before_after(
            agent.frame,
            change_date=change["date"],
            change_name=str(change["change_name"]),
            product_area=str(change["product_area"]),
            theme=agent.theme_for_area(str(change["product_area"])),
        )
        documents.append(insights.build_outcome_memory(comparison))

    period = agent.periods()[-1]
    for signal in agent.emerging_signals(period):
        documents.append(
            insights.build_emerging_issue_memory(signal, agent.frame, period=period)
        )
    documents.append(
        insights.build_decision_memory(
            recommendation="Investigate payments",
            decision="accepted",
            reason="revenue impact",
            learning="watch payment reliability",
            theme="payment_reliability",
            product_area="Payments",
        )
    )

    assert documents
    for document in documents:
        assert document.when, f"{document.kind} memory has no date"
        _to_aware_datetime(document.when)  # raises ValueError if unparseable


def test_hindsight_uses_a_separate_llm_model_from_pulsemind_reasoning(
    settings: Settings,
) -> None:
    """Groq enforces rate limits per model, so the two workloads must not share one.

    Memory extraction and PulseMind's reasoning each have their own tokens-per-minute
    budget only if they call different models.
    """
    assert settings.hindsight_llm_model != settings.groq_model
    memory = HindsightMemory(settings)
    assert memory.llm_model == settings.hindsight_llm_model
    assert memory.status()["llm_model"] == settings.hindsight_llm_model
    assert memory.status()["reasoning_model"] == settings.groq_model


def test_theme_for_area_is_deterministic(agent: PulseMindAgent) -> None:
    assert agent.theme_for_area("Checkout") == "checkout_friction"
    assert agent.theme_for_area("Payments") == "payment_reliability"
    assert agent.theme_for_area("Nonexistent Area") is None


class RateLimitedOnceMemory(FakeMemory):
    """Reports a provider rate limit on the first retain, then succeeds."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.attempts = 0

    def retain(self, content, *, kind, when=None, context=None, tags=None, metadata=None, document_id=None):
        self.attempts += 1
        if self.attempts == 1:
            return RetainReceipt(
                ok=False,
                error="ServiceException: (500) HTTP 429 rate limit reached for this org",
                tags=list(tags or []),
            )
        return super().retain(
            content,
            kind=kind,
            when=when,
            context=context,
            tags=tags,
            metadata=metadata,
            document_id=document_id,
        )


class AlwaysFailingMemory(FakeMemory):
    """Fails with a non-retryable error."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.attempts = 0

    def retain(self, content, *, kind, when=None, context=None, tags=None, metadata=None, document_id=None):
        self.attempts += 1
        return RetainReceipt(ok=False, error="400 invalid_request_error: bad metadata", tags=list(tags or []))


def test_rate_limited_retain_is_retried(settings: Settings) -> None:
    """A 429 must not silently drop a memory on the free tier."""
    memory = RateLimitedOnceMemory(settings)
    agent = PulseMindAgent(settings, memory=memory, llm=FakeLLM())
    report = agent.ingest_feedback("2026-01")
    assert report.written == 1 and report.failed == 0
    assert memory.attempts > 1, "the rate-limited attempt should have been retried"


def test_non_rate_limit_failure_is_not_retried(settings: Settings) -> None:
    memory = AlwaysFailingMemory(settings)
    agent = PulseMindAgent(settings, memory=memory, llm=FakeLLM())
    report = agent.ingest_feedback("2026-01")
    assert report.failed == 1
    assert memory.attempts == 1, "only provider rate limits are worth retrying"


def test_rate_limit_classification_and_backoff(settings: Settings) -> None:
    assert PulseMindAgent._is_rate_limited("HTTP 429 rate limit reached")
    assert PulseMindAgent._is_rate_limited("Provider quota exhausted")
    assert not PulseMindAgent._is_rate_limited("400 invalid_request_error")
    assert not PulseMindAgent._is_rate_limited(None)
    # A provider-suggested delay is honoured (plus a small safety margin).
    assert PulseMindAgent._rate_limit_delay("please try again in 12.0s", 25.0) == 13.0
    assert PulseMindAgent._rate_limit_delay("no hint here", 25.0) == 25.0
    assert PulseMindAgent._rate_limit_delay(None, 5.0) == 5.0


def test_pacing_gap_is_configurable(settings: Settings) -> None:
    assert dataclasses.replace(settings, retain_pace_seconds=0.0).retain_pace_seconds == 0.0
    assert dataclasses.replace(settings, retain_pace_seconds=40.0).retain_pace_seconds == 40.0


def test_incremental_ingestion_accumulates_memories(agent: PulseMindAgent) -> None:
    """The demo ingests January, then February, then March: memory must accumulate."""
    before = len(agent.memory.retained)
    agent.ingest_feedback("2026-01")
    after_january = len(agent.memory.retained)
    agent.ingest_feedback("2026-03")
    after_march = len(agent.memory.retained)
    assert before < after_january < after_march
    january_themes = trends.complaint_share_matrix(agent.frame)["2026-01"]
    assert january_themes["checkout_friction"] > january_themes["payment_reliability"]


def test_record_product_change_returns_path_like_value(agent: PulseMindAgent) -> None:
    """The UI renders `{path.name}` from this return value, so it must be path-like.

    Regression: the agent used to return a plain `str`, and
    `ui/product_changes.py` crashed with ``AttributeError: 'str' object has no
    attribute 'name'`` the moment a change was recorded. The agent keeps its
    documented `str` contract (asserted by
    `test_record_product_change_persists_and_retains_in_memory` via
    `Path(path).exists()`), so the UI normalizes with `Path(...)` at its boundary —
    this test pins that the returned value survives exactly that call.
    """
    from pathlib import Path as _Path

    path, receipt = agent.record_product_change(
        {
            "change_id": "",
            "date": "2026-04-05",
            "change_name": "Path Contract Check",
            "product_area": "Checkout",
            "problem_targeted": "checkout friction",
            "expected_outcome": "fewer complaints",
        }
    )
    assert receipt.ok
    assert _Path(path).name == "product_changes.csv"  # the exact expression the UI uses
