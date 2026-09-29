"""What should the Product Manager investigate next?

Two steps, kept deliberately separate:

1. ``rank_opportunities`` is deterministic Python. It scores every theme using the
   numbers PulseMind already computed (how fast its complaint share is moving, how
   many complaints it carries, how bad the ratings are) so the shortlist is
   reproducible and auditable.
2. ``recommend`` asks Groq to turn the top candidate, the measured outcome of any
   related change and the memories recalled from Hindsight into one actionable
   recommendation, with the evidence it used attached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.config import THEME_LABELS
from core.groq_client import GroqLLM
from core.hindsight_client import HindsightMemory, MemoryHit
from services import trend_analyzer as trends
from services.insight_engine import ANALYST_SYSTEM, Insight, render_memory_context

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    title: str
    action: str
    why: str
    theme: str
    product_area: str
    confidence: str
    what_to_check: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    memories_used: list[MemoryHit] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def rank_opportunities(
    frame: pd.DataFrame,
    changes: pd.DataFrame | None = None,
    *,
    period: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Deterministically shortlist what to investigate next.

    Scoring (all inputs are pure Python metrics):
      * movement  - how far the theme's complaint share moved vs earlier periods
      * volume    - how many complaints it carries in the latest period
      * severity  - how low the average rating is for those complaints
      * fresh     - themes with a recently recorded product change *after* the
                    measurement window are less urgent; themes with no recorded
                    change behind them are more urgent
    """
    available = trends.periods(frame)
    if not available:
        return []
    period = period or available[-1]
    period = period if period in available else available[-1]

    signals = {signal.theme: signal for signal in trends.emerging_signals(frame, period)}
    shares = trends.complaint_share_matrix(frame)
    counts = trends.theme_count_matrix(frame)
    if counts.empty or period not in counts.columns:
        return []

    total_complaints = int(counts[period].sum()) or 1
    latest_neg = trends.negatives(trends.filter_period(frame, period))
    changed_areas: set[str] = set()
    if changes is not None and not changes.empty:
        recent_changes = changes[changes["date"].dt.strftime("%Y-%m") >= period]
        changed_areas = set(recent_changes["product_area"].astype(str))

    ranked: list[dict[str, Any]] = []
    for theme in counts.index:
        count = int(counts.loc[theme, period])
        if count == 0:
            continue
        share = float(shares.loc[theme, period]) if period in shares.columns else 0.0
        signal = signals.get(theme)
        rows = latest_neg[latest_neg["theme"] == theme]
        avg_rating = float(rows["rating"].mean()) if rows["rating"].notna().any() else 3.0
        area = str(rows["product_area"].mode().iloc[0]) if not rows.empty else "unknown"

        movement = abs(signal.share_delta_pts) if signal else 0.0
        has_change = area in changed_areas
        score = (
            movement * 1.4
            + share * 0.6
            + count * 0.5
            + (3.0 - min(avg_rating, 3.0)) * 4.0
            + (0.0 if has_change else 6.0)
        )

        ranked.append(
            {
                "theme": theme,
                "label": THEME_LABELS.get(theme, theme),
                "product_area": area,
                "period": period,
                "complaints": count,
                "share": round(share, 1),
                "share_delta_pts": round(movement, 1) if signal else 0.0,
                "direction": signal.direction if signal else "stable",
                "avg_rating": round(avg_rating, 2),
                "total_complaints": total_complaints,
                "score": round(score, 1),
                "already_addressed_by_change": has_change,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def recommend(
    candidate: dict[str, Any],
    llm: GroqLLM,
    *,
    insight: Insight | None = None,
    memory: HindsightMemory | None = None,
    frame: pd.DataFrame | None = None,
    changes: pd.DataFrame | None = None,
) -> Recommendation:
    """Turn the top-ranked candidate into one grounded recommendation."""
    theme = candidate["theme"]
    product_area = candidate["product_area"]

    recalled: list[MemoryHit] = []
    if memory is not None and memory.is_started:
        recalled = memory.recall(
            f"What do we know about {theme} in {product_area}: earlier complaints, "
            "product changes, measured outcomes and previous decisions?",
            kinds=None,
            budget="mid",
            max_tokens=2560,
        )
    if insight and insight.evidence_memories:
        known = {hit.id for hit in recalled}
        recalled.extend(hit for hit in insight.evidence_memories if hit.id not in known)

    facts = [
        f"Candidate to investigate: {candidate['label']} in {product_area}.",
        f"Complaints in the latest period: {candidate['complaints']} "
        f"({candidate['share']}% of all complaints).",
        f"Complaint share movement versus earlier periods: {candidate['share_delta_pts']:+.1f} points "
        f"({candidate['direction']}).",
        f"Average rating of these complaints: {candidate['avg_rating']}/5.",
    ]
    if insight and insight.outcome:
        outcome = insight.outcome
        facts.append(
            f"A related recorded change, {outcome['change_name']} ({outcome['change_date']}), "
            f"measured {outcome['share_delta_pts']:+.1f} points: verdict {outcome['verdict']}."
        )
    if candidate.get("already_addressed_by_change"):
        facts.append("A product change in this area was already recorded for this or a later period.")
    if frame is not None:
        for row in trends.recent_complaints(frame, candidate.get("period", "") or "", limit=3, theme=theme):
            facts.append(f'Latest verbatim ({row["date"]}, {row["rating"]}/5): "{row["text"]}"')

    facts.append("")
    facts.append("RECALLED FROM HINDSIGHT (long-term memory):")
    facts.append(render_memory_context(recalled))

    instruction = (
        "Write ONE recommendation for the Product Manager.\n"
        "Return JSON with exactly these keys:\n"
        '  "title": short imperative title (max 8 words)\n'
        '  "action": 1-2 sentences on what to investigate and how\n'
        '  "why": 2-3 sentences that use the recalled history (earlier feedback, the '
        "product change, its measured outcome, past decisions) to justify the priority\n"
        '  "confidence": "high" | "medium" | "low"\n'
        '  "what_to_check": array of 2-4 concrete things to check next\n'
        '  "supporting_evidence": array of 2-4 short strings, each taken from the '
        "recalled memories (paraphrase tightly, keep the dates)\n"
        "If memory contains nothing relevant, return an empty supporting_evidence array "
        "and lower the confidence. Never invent releases, dates or numbers."
    )

    payload = llm.complete_json_safe(
        "\n".join(facts) + "\n\n" + instruction,
        system=ANALYST_SYSTEM,
        fallback={},
        max_tokens=1100,
        label=f"recommend[{theme}]",
    )

    def _list(key: str) -> list[str]:
        value = payload.get(key, []) if isinstance(payload, dict) else []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value][:4]

    if isinstance(payload, dict) and payload.get("title"):
        return Recommendation(
            title=str(payload.get("title", "")),
            action=str(payload.get("action", "")),
            why=str(payload.get("why", "")),
            theme=theme,
            product_area=product_area,
            confidence=str(payload.get("confidence", "medium")).lower(),
            what_to_check=_list("what_to_check"),
            supporting_evidence=_list("supporting_evidence"),
            memories_used=recalled,
            metrics=candidate,
        )

    return Recommendation(
        title=f"Investigate {candidate['label']}",
        action=(
            f"PulseMind ranked {candidate['label']} (in {product_area}) as the most "
            f"urgent topic: {candidate['complaints']} complaints, {candidate['share']}% of all "
            f"complaints, {candidate['share_delta_pts']:+.1f} points versus earlier periods."
        ),
        why="Groq was unavailable, so this is the deterministic ranking without narrative reasoning.",
        theme=theme,
        product_area=product_area,
        confidence="low",
        what_to_check=[
            "Review the latest verbatims for this topic",
            "Check whether a recorded product change already targets this area",
        ],
        supporting_evidence=[hit.text[:220] for hit in recalled[:3]],
        memories_used=recalled,
        metrics=candidate,
        error="Groq call failed; showing deterministic fallback recommendation.",
    )
