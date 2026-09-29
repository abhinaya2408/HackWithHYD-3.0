"""Insight generation: deterministic metrics + Hindsight recall + Groq reasoning.

This module is where memory changes the answer. The same current-period numbers are
handed to Groq twice:

* ``use_memory=False`` - only the current period's statistics, i.e. a generic
  "what are people complaining about right now" analysis.
* ``use_memory=True``  - the same statistics *plus* what Hindsight recalls from
  earlier feedback, the product changes that were recorded, the outcomes measured
  for those changes and the decisions the Product Manager already took.

Every recalled memory used is kept on the result object, so the Memory Explorer can
show the exact evidence the insight was built from rather than claiming memory was
used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.config import (
    TAG_DECISION,
    TAG_EMERGING_ISSUE,
    TAG_FEEDBACK,
    TAG_OUTCOME,
    TAG_PRODUCT_CHANGE,
    THEME_LABELS,
)
from core.groq_client import GroqLLM
from core.hindsight_client import HindsightMemory, MemoryHit
from services import trend_analyzer as trends
from services.feedback_analyzer import MemoryDocument, stringify_metadata, theme_mix

logger = logging.getLogger(__name__)

ANALYST_SYSTEM = (
    "You are PulseMind, a product intelligence analyst for a Product Manager. "
    "You are given deterministic metrics computed in Python and, when available, "
    "memories recalled from a long-term memory system (Hindsight). "
    "Rules you must follow: never invent numbers, never invent a product release or "
    "release date that is not in the provided memories, and separate what the data "
    "shows from what you infer. Ratings use a 1-5 star scale where 1 is the worst "
    "and 5 is the best, so a falling average rating means customers are LESS "
    "satisfied, never more. Complaint share falling means the topic is improving. "
    "When memories are provided, explicitly connect the "
    "current situation to the earlier feedback, product changes, outcomes and past "
    "decisions. Keep answers concrete and short. Claim causation only when the data "
    "actually supports it: otherwise say 'After the release, the measured feedback "
    "changed…' rather than 'because of the release'. When a product change came from "
    "GitHub, mention the repository or version it shipped under."
)


def render_memory_context(hits: list[MemoryHit], limit: int = 8) -> str:
    """Render recalled memories as the context block handed to Groq.

    Capped and truncated deliberately: the JSON instruction, the deterministic facts
    and this block must all fit inside the completion budget together, or the model
    runs out of tokens before it emits a valid document.
    """
    if not hits:
        return "(no memories were recalled)"
    blocks: list[str] = []
    for index, hit in enumerate(hits[:limit], start=1):
        header = f"[M{index}] kind={hit.kind} fact_type={hit.fact_type}"
        when = hit.occurred_start or hit.metadata.get("memory_date") or ""
        if when:
            header += f" date={str(when)[:10]}"
        blocks.append(f"{header}\n{hit.text.strip()[:500]}")
    return "\n\n".join(blocks)


def render_period_facts(frame: pd.DataFrame, period: str) -> str:
    """Deterministic facts for a period: what the 'no memory' agent sees."""
    return trends.render_period_brief(frame, period)


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class CurrentAnalysis:
    """An answer to 'what are customers complaining about now?'"""

    period: str
    mode: str  # "without_memory" | "with_memory"
    narrative: str
    themes: list[dict[str, Any]] = field(default_factory=list)
    emerging: list[dict[str, Any]] = field(default_factory=list)
    memories_used: list[MemoryHit] = field(default_factory=list)
    prompt_context: str = ""
    error: str | None = None

    @property
    def memory_count(self) -> int:
        return len(self.memories_used)


@dataclass
class Insight:
    """A context-aware insight about one theme, with the evidence behind it."""

    theme: str
    label: str
    product_area: str
    period: str
    direction: str
    narrative: str
    trend_text: str = ""
    sentiment_change: str = ""
    historical_context: list[str] = field(default_factory=list)
    related_change: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    evidence_memories: list[MemoryHit] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_count(self) -> int:
        return len(self.evidence_memories)


# ---------------------------------------------------------------------------
# Current-period analysis (with vs without memory)
# ---------------------------------------------------------------------------


def analyze_period(
    frame: pd.DataFrame,
    period: str,
    llm: GroqLLM,
    memory: HindsightMemory | None = None,
    *,
    use_memory: bool = True,
) -> CurrentAnalysis:
    """Answer 'what are customers complaining about now?' with or without memory."""
    available = trends.periods(frame)
    if not available:
        return CurrentAnalysis(period=period, mode="without_memory", narrative="", error="No feedback loaded.")

    period = period if period in available else available[-1]
    earlier = [p for p in available if p < period]
    signals = trends.emerging_signals(frame, period)
    themes = theme_mix(trends.filter_period(frame, period))

    context_lines = [render_period_facts(frame, period)]
    for previous in earlier[-3:]:
        context_lines.append("")
        context_lines.append(render_period_facts(frame, previous))

    context_lines.append("")
    context_lines.append("Deterministic emerging/improving signals for this period:")
    if signals:
        for signal in signals:
            context_lines.append(
                f"  - {signal.direction}: {signal.theme} ({signal.product_area}); "
                f"{signal.latest_count} complaints = {signal.latest_share:.1f}% of complaints, "
                f"vs {signal.baseline_share:.1f}% in earlier periods "
                f"({signal.share_delta_pts:+.1f} points)."
            )
    else:
        context_lines.append("  - none crossed the detection thresholds")

    recalled: list[MemoryHit] = []
    if use_memory and memory is not None and memory.is_started:
        recalled = _recall_history(
            memory,
            queries=[
                f"What did customers complain about in {period} and in earlier periods?",
                "What product changes were recorded and what did they target?",
                "What outcomes were measured after those product changes?",
            ],
        )
        context_lines.append("")
        context_lines.append("RECALLED FROM HINDSIGHT (long-term memory):")
        context_lines.append(render_memory_context(recalled))

    prompt = "\n".join(context_lines)
    if use_memory:
        instruction = (
            f"Write the executive analysis for period {period}.\n"
            "Use the deterministic metrics for all numbers. Use the RECALLED FROM "
            "HINDSIGHT block to explain what is new versus what has been seen before: "
            "reference earlier feedback on the same topics, the product changes that "
            "were recorded, whether the outcomes suggest they worked, and any past "
            "decisions that are relevant. If the memory contains nothing relevant to a "
            "point you are making, say so instead of guessing.\n"
            "Return JSON: {\"narrative\": \"...\"} - 4 to 6 sentences."
        )
    else:
        instruction = (
            f"Write the executive analysis for period {period}.\n"
            "You have only the current and immediately preceding periods' metrics. "
            "Do not claim to know about previous product changes, previous decisions or "
            "historical outcomes - you have no memory of them.\n"
            "Return JSON: {\"narrative\": \"...\"} - 4 to 6 sentences."
        )

    narrative = ""
    error = None
    try:
        payload = llm.complete_json(f"{prompt}\n\n{instruction}", system=ANALYST_SYSTEM, max_tokens=1600,
                                    label=f"analyze[{period}|memory={use_memory}]")
        narrative = str(payload.get("narrative", "")) if isinstance(payload, dict) else str(payload)
    except Exception as exc:  # noqa: BLE001 - show the failure in the UI, don't crash
        error = str(exc)
        narrative = _fallback_narrative(period, themes, signals, use_memory)

    return CurrentAnalysis(
        period=period,
        mode="with_memory" if use_memory else "without_memory",
        narrative=narrative,
        themes=themes,
        emerging=[signal.as_dict() for signal in signals],
        memories_used=recalled,
        prompt_context=prompt,
        error=error,
    )


def _fallback_narrative(
    period: str,
    themes: list[dict[str, Any]],
    signals: list[trends.EmergingSignal],
    use_memory: bool,
) -> str:
    """Deterministic stand-in text so the app still works if Groq is rate limited."""
    if not themes:
        return f"No negative feedback was recorded in {period}."
    top = themes[0]
    parts = [
        f"In {period}, the largest share of complaints was {top['label']} "
        f"({top['count']} items, {top['share']}% of all complaints)."
    ]
    for signal in signals[:2]:
        parts.append(
            f"{THEME_LABELS.get(signal.theme, signal.theme)} moved "
            f"{signal.share_delta_pts:+.1f} points versus earlier periods "
            f"({signal.direction})."
        )
    if not use_memory:
        parts.append("(Generated without long-term memory: Groq was unavailable.)")
    else:
        parts.append("(Generated without long-term memory reasoning: Groq was unavailable.)")
    return " ".join(parts)


def _recall_history(
    memory: HindsightMemory,
    queries: list[str],
    *,
    kinds: list[str] | None = None,
    per_query: int = 5,
) -> list[MemoryHit]:
    """Run several recall queries and de-duplicate by memory id."""
    seen: set[str] = set()
    collected: list[MemoryHit] = []
    for query in queries:
        for hit in memory.recall(query, kinds=kinds, budget="mid", max_tokens=2048):
            if hit.id in seen:
                continue
            seen.add(hit.id)
            collected.append(hit)
            if len(collected) >= per_query * len(queries):
                return collected
    return collected


# ---------------------------------------------------------------------------
# Insight for a specific theme
# ---------------------------------------------------------------------------


def find_related_change(
    changes: pd.DataFrame,
    *,
    product_area: str,
    theme: str | None = None,
    before_period: str | None = None,
) -> dict[str, Any] | None:
    """Deterministically link an insight to a PM-recorded product change."""
    if changes is None or changes.empty:
        return None
    candidates = changes[changes["product_area"] == product_area]
    if candidates.empty:
        return None
    if theme:
        label_words = THEME_LABELS.get(theme, theme).lower().split()
        keywords = [word for word in label_words if len(word) > 3]
        keyword_hits = candidates[
            candidates["problem_targeted"].str.lower().apply(
                lambda text: any(word in text for word in keywords)
            )
        ]
        if not keyword_hits.empty:
            candidates = keyword_hits
    if before_period:
        earlier = candidates[candidates["date"].dt.strftime("%Y-%m") <= before_period]
        if not earlier.empty:
            candidates = earlier
    row = candidates.sort_values("date").iloc[-1]
    return {
        "change_id": str(row.get("change_id", "")),
        "change_name": str(row["change_name"]),
        "date": pd.Timestamp(row["date"]).date().isoformat(),
        "product_area": str(row["product_area"]),
        "problem_targeted": str(row.get("problem_targeted", "")),
        "expected_outcome": str(row.get("expected_outcome", "")),
        # Optional GitHub provenance (empty strings for PM-recorded changes).
        "source": str(row.get("source", "") or ""),
        "github_repo": str(row.get("github_repo", "") or ""),
        "github_kind": str(row.get("github_kind", "") or ""),
        "github_number": str(row.get("github_number", "") or ""),
        "github_version": str(row.get("github_version", "") or ""),
        "github_url": str(row.get("github_url", "") or ""),
    }


def build_insight(
    frame: pd.DataFrame,
    changes: pd.DataFrame,
    theme: str,
    llm: GroqLLM,
    memory: HindsightMemory | None = None,
    *,
    period: str | None = None,
    window_days: int = 28,
) -> Insight:
    """Build one context-aware insight: trend + history + related change + outcome."""
    available = trends.periods(frame)
    period = period or (available[-1] if available else "")
    signals = trends.emerging_signals(frame, period)
    signal = next((sig for sig in signals if sig.theme == theme), None)

    product_area = (
        signal.product_area
        if signal
        else _area_for_theme(frame, theme)
    )

    counts = trends.theme_period_series(frame, theme, "count")
    shares = trends.theme_period_series(frame, theme, "share")
    metrics = {
        "theme": theme,
        "product_area": product_area,
        "period": period,
        "counts_by_period": {str(k): int(v) for k, v in counts.items()},
        "share_by_period": {str(k): round(float(v), 1) for k, v in shares.items()},
        "trend": trends.trend_direction(shares.tail(3)) if not shares.empty else "stable",
    }
    if signal:
        metrics["signal"] = signal.as_dict()

    related_change = find_related_change(
        changes, product_area=product_area, theme=theme, before_period=period
    )

    outcome: dict[str, Any] | None = None
    if related_change:
        comparison = trends.before_after(
            frame,
            change_date=related_change["date"],
            change_name=related_change["change_name"],
            product_area=related_change["product_area"],
            theme=theme if signal else None,
            window_days=window_days,
        )
        outcome = comparison.as_dict()

    sentiment = trends.sentiment_trend(frame)
    sentiment_change = ""
    if period in sentiment.index:
        row = sentiment.loc[period]
        earlier = sentiment[sentiment.index < period]
        trend_word = "steady"
        if not earlier.empty:
            prior_share = float(earlier["negative_share"].iloc[-1])
            delta = float(row["negative_share"]) - prior_share
            trend_word = "worsening" if delta > 1 else "improving" if delta < -1 else "steady"
        sentiment_change = (
            f"{int(row['negative'])} of {int(row['total'])} items negative "
            f"({float(row['negative_share']):.1f}%), average rating {float(row['avg_rating']):.2f}/5 "
            f"and the overall tone is {trend_word} versus the previous period."
        )

    evidence: list[MemoryHit] = []
    if memory is not None and memory.is_started:
        evidence = _recall_history(
            memory,
            queries=[
                f"What feedback did customers give about {theme} in {product_area}?",
                f"Have we seen {THEME_LABELS.get(theme, theme)} problems before {period}?",
                f"What happened after product changes in {product_area}?",
                "What did the Product Manager decide previously and what was learned?",
            ],
            per_query=4,
        )

    facts = [
        f"Theme: {theme} ({THEME_LABELS.get(theme, theme)}), product area {product_area}.",
        trends.render_theme_series(frame, theme),
    ]
    if signal:
        facts.append(
            f"Emerging signal: share of complaints moved from {signal.baseline_share:.1f}% "
            f"(earlier periods) to {signal.latest_share:.1f}% ({period}), "
            f"i.e. {signal.share_delta_pts:+.1f} percentage points."
        )
    if single_period_rows := trends.recent_complaints(frame, period, limit=4, theme=theme):
        facts.append("Latest verbatims:")
        for row in single_period_rows:
            facts.append(f'  - {row["date"]} rating {row["rating"]}/5: "{row["text"]}"')
    if related_change:
        facts.append(
            f"Product change linked by product area: {related_change['change_name']} "
            f"({related_change['date']}) targeting '{related_change['problem_targeted']}' "
            f"with expected outcome '{related_change['expected_outcome']}'."
        )
    if outcome:
        facts.append(
            f"Deterministic before/after for that change ({outcome['window_days']}-day windows): "
            f"complaints {outcome['before']['complaints']} -> {outcome['after']['complaints']}, "
            f"share of all complaints {outcome['before']['share_of_complaints']}% -> "
            f"{outcome['after']['share_of_complaints']}% "
            f"({outcome['share_delta_pts']:+.1f} points), verdict: {outcome['verdict']}."
        )
    if sentiment_change:
        facts.append(f"Sentiment: {sentiment_change}")

    facts.append("")
    facts.append("RECALLED FROM HINDSIGHT (long-term memory):")
    facts.append(render_memory_context(evidence))

    instruction = (
        "Produce one insight object about this theme.\n"
        "Return JSON with exactly these keys:\n"
        '  "narrative": 3-4 sentences connecting the current trend to the recalled '
        "history (earlier feedback, the product change, its measured outcome, past decisions).\n"
        '  "trend_text": one sentence stating the deterministic trend.\n'
        '  "sentiment_change": one sentence on the sentiment move.\n'
        '  "historical_context": an array of 2-4 short strings, each a specific thing '
        "recalled from Hindsight that matters here (quote the memory content).\n"
        "If the recalled memory is empty or irrelevant to this theme, return an empty "
        "historical_context array and say in the narrative that there is no relevant "
        "history. Never invent a release, date or number."
    )

    payload = llm.complete_json_safe(
        "\n".join(facts) + "\n\n" + instruction,
        system=ANALYST_SYSTEM,
        fallback={},
        max_tokens=1100,
        label=f"insight[{theme}]",
    )

    historical = payload.get("historical_context", []) if isinstance(payload, dict) else []
    if isinstance(historical, str):
        historical = [historical]

    return Insight(
        theme=theme,
        label=THEME_LABELS.get(theme, theme),
        product_area=product_area,
        period=period,
        direction=signal.direction if signal else metrics["trend"],
        narrative=str(payload.get("narrative", "")) if isinstance(payload, dict) else "",
        trend_text=str(payload.get("trend_text", "")) if isinstance(payload, dict) else "",
        sentiment_change=str(payload.get("sentiment_change", "")) if isinstance(payload, dict) else sentiment_change,
        historical_context=[str(item) for item in historical][:4],
        related_change=related_change,
        outcome=outcome,
        evidence_memories=evidence,
        metrics=metrics,
    )


def _area_for_theme(frame: pd.DataFrame, theme: str) -> str:
    subset = frame[frame["theme"] == theme]
    if subset.empty:
        return "unknown"
    return str(subset["product_area"].mode().iloc[0])


# ---------------------------------------------------------------------------
# Memory documents produced by this engine
# ---------------------------------------------------------------------------


def build_outcome_memory(
    comparison: trends.OutcomeComparison,
    insight: Insight | None = None,
    *,
    observed_impact: str = "",
) -> MemoryDocument:
    """Serialise a measured before/after change into a retainable outcome memory."""
    data = comparison.as_dict()
    impact = observed_impact or insight.narrative if insight else ""
    content = "\n".join(
        [
            f"Measured outcome of the product change '{data['change_name']}' "
            f"released on {data['change_date']} in {data['product_area']}.",
            "",
            f"Situation before the change ({data['before']['window_start']} to "
            f"{data['before']['window_end']}): {data['before']['complaints']} "
            f"complaints about {data['theme'] or data['product_area']}, which was "
            f"{data['before']['share_of_complaints']}% of all complaints in that window; "
            f"average rating {data['before']['avg_rating']}/5.",
            f"Situation after the change ({data['after']['window_start']} to "
            f"{data['after']['window_end']}): {data['after']['complaints']} complaints, "
            f"{data['after']['share_of_complaints']}% of all complaints; average rating "
            f"{data['after']['avg_rating']}/5.",
            f"Feedback trend: complaint volume moved {data['count_delta']:+d} items and "
            f"complaint share moved {data['share_delta_pts']:+.1f} percentage points.",
            f"Ratings: {data.get('rating_text') or 'rating movement not recorded'} "
            f"(1 = worst, 5 = best).",
            f"Verdict: the change {data['verdict']}.",
            "",
            f"Observed impact: {impact}" if impact else "",
        ]
    )
    return MemoryDocument(
        kind=TAG_OUTCOME,
        content=content,
        when=data["after"]["window_end"],
        context=f"Outcome measured for {data['change_name']}",
        tags=[
            TAG_OUTCOME,
            f"area:{data['product_area'].lower().replace(' ', '-')}",
            f"change:{data['change_name'].lower().replace(' ', '-')}",
        ],
        metadata=stringify_metadata(
            {
                "change_name": data["change_name"],
                "release_date": data["change_date"],
                "product_area": data["product_area"],
                "window_days": data["window_days"],
                "before_complaints": data["before"]["complaints"],
                "after_complaints": data["after"]["complaints"],
                "share_delta_pts": data["share_delta_pts"],
                "verdict": data["verdict"],
            }
        ),
        document_id=f"outcome-{data['change_name'].lower().replace(' ', '-')}",
    )


def build_emerging_issue_memory(
    signal: trends.EmergingSignal,
    frame: pd.DataFrame,
    *,
    historical_context: list[str] | None = None,
    period: str | None = None,
) -> MemoryDocument:
    """Serialise a detected emerging issue (with its historical context) as memory."""
    period = period or signal.latest_period
    rows = trends.recent_complaints(frame, period, limit=4, theme=signal.theme)
    # `when` must be a real date: Hindsight parses it into a datetime, so a bare
    # "2026-03" period key would raise `Invalid isoformat string`. Use the last day
    # feedback actually arrived in this period.
    period_rows = trends.filter_period(frame, period)
    when = str(period_rows["date"].max())[:10] if not period_rows.empty else f"{period}-01"
    evidence = [
        f'  - {row["date"]} rating {row["rating"]}/5 ({row["product_area"]}): "{row["text"]}"'
        for row in rows
    ]
    history = [f"  - {item}" for item in (historical_context or [])]

    content = "\n".join(
        [
            f"Emerging issue detected on {period}: {signal.theme} "
            f"({THEME_LABELS.get(signal.theme, signal.theme)}) in {signal.product_area}.",
            "",
            f"Trend: this topic accounted for {signal.latest_share:.1f}% of all complaints in "
            f"{period}, compared with {signal.baseline_share:.1f}% across earlier periods "
            f"({signal.share_delta_pts:+.1f} percentage points). "
            f"Complaint count in {period}: {signal.latest_count}.",
            f"Average rating for these complaints: {signal.latest_avg_rating:.2f}/5 "
            f"(earlier periods: {signal.baseline_avg_rating:.2f}/5).",
            "",
            "Evidence:",
            *evidence,
            "",
            "Historical context recalled from memory:",
            *(history or ["  - none: this topic had no comparable history in memory"]),
        ]
    )
    return MemoryDocument(
        kind=TAG_EMERGING_ISSUE,
        content=content,
        when=when,
        context=f"Emerging issue: {signal.theme}",
        tags=[TAG_EMERGING_ISSUE, f"theme:{signal.theme}", f"area:{signal.product_area.lower().replace(' ', '-')}"],
        metadata=stringify_metadata(
            {
                "theme": signal.theme,
                "product_area": signal.product_area,
                "period": period,
                "latest_share": f"{signal.latest_share:.1f}",
                "baseline_share": f"{signal.baseline_share:.1f}",
                "share_delta_pts": f"{signal.share_delta_pts:.1f}",
            }
        ),
        document_id=f"emerging-{signal.theme}-{period}",
    )


def build_decision_memory(
    *,
    recommendation: str,
    decision: str,
    reason: str,
    learning: str,
    theme: str = "",
    product_area: str = "",
    when: str | None = None,
) -> MemoryDocument:
    """Serialise the Product Manager's decision so future analysis can learn from it."""
    stamp = when or datetime.now(timezone.utc).date().isoformat()
    content = "\n".join(
        [
            "Product Manager decision recorded in PulseMind.",
            f"Date: {stamp}",
            f"Recommendation: {recommendation}",
            f"Decision: {decision}",
            f"Reason given by the Product Manager: {reason or 'not provided'}",
            f"Resulting learning to carry forward: {learning or 'not provided'}",
            f"Related theme: {THEME_LABELS.get(theme, theme) if theme else 'not specified'}",
            f"Related product area: {product_area or 'not specified'}",
        ]
    )
    return MemoryDocument(
        kind=TAG_DECISION,
        content=content,
        when=stamp,
        context="Product Manager decision and learning",
        tags=[TAG_DECISION, *( [f"theme:{theme}"] if theme else [] )],
        metadata=stringify_metadata(
            {
                "decision": decision,
                "recommendation": recommendation,
                "reason": reason,
                "learning": learning,
                "theme": theme,
                "product_area": product_area,
                "decision_date": stamp,
            }
        ),
        document_id=f"decision-{stamp}-{theme or 'general'}",
    )


__all__ = [
    "ANALYST_SYSTEM",
    "CurrentAnalysis",
    "Insight",
    "analyze_period",
    "build_decision_memory",
    "build_emerging_issue_memory",
    "build_insight",
    "build_outcome_memory",
    "find_related_change",
    "render_memory_context",
    "render_period_facts",
    "TAG_FEEDBACK",
    "TAG_PRODUCT_CHANGE",
]
