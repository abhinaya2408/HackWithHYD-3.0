"""Deterministic product-feedback analytics.

Everything numeric that PulseMind shows or reasons about is computed here, in
plain Python/pandas: counts, percentages, complaint shares, trends, time
comparisons and before/after outcome deltas. The LLM is never asked to do this
arithmetic — it is handed these finished numbers as context.

Two conventions matter for the whole app:

* A *theme* is a topic (checkout friction, payment reliability, ...). A topic can
  appear in negative or positive feedback.
* *Complaint share* is always measured over negative feedback only, so a spike in
  positive reviews cannot disguise a rising complaint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from core.config import PRODUCT_AREAS, THEMES

NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class PeriodSummary:
    period: str
    total: int
    negative: int
    neutral: int
    positive: int
    avg_rating: float
    top_theme: str | None
    top_theme_share: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "total": self.total,
            "negative": self.negative,
            "neutral": self.neutral,
            "positive": self.positive,
            "avg_rating": round(self.avg_rating, 2),
            "top_theme": self.top_theme,
            "top_theme_share": round(self.top_theme_share, 1),
        }


@dataclass
class EmergingSignal:
    """A theme whose share of complaints moved materially between periods."""

    theme: str
    product_area: str
    direction: str  # "emerging" | "improving"
    latest_period: str
    latest_count: int
    latest_share: float
    baseline_share: float
    share_delta_pts: float
    latest_avg_rating: float
    baseline_avg_rating: float
    per_period_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "product_area": self.product_area,
            "direction": self.direction,
            "latest_period": self.latest_period,
            "latest_count": self.latest_count,
            "latest_share": round(self.latest_share, 1),
            "baseline_share": round(self.baseline_share, 1),
            "share_delta_pts": round(self.share_delta_pts, 1),
            "latest_avg_rating": round(self.latest_avg_rating, 2),
            "baseline_avg_rating": round(self.baseline_avg_rating, 2),
            "per_period_counts": dict(sorted(self.per_period_counts.items())),
        }


@dataclass
class OutcomeComparison:
    """Before/after measurement of one product change."""

    change_name: str
    change_date: str
    product_area: str
    window_days: int
    theme: str | None
    before: dict[str, Any]
    after: dict[str, Any]
    share_delta_pts: float
    count_delta: int
    rating_delta: float
    verdict: str
    rating_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "change_name": self.change_name,
            "change_date": self.change_date,
            "product_area": self.product_area,
            "window_days": self.window_days,
            "theme": self.theme,
            "before": self.before,
            "after": self.after,
            "share_delta_pts": round(self.share_delta_pts, 1),
            "count_delta": self.count_delta,
            "rating_delta": round(self.rating_delta, 2),
            "verdict": self.verdict,
            "rating_text": self.rating_text,
        }


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame has a real datetime column and a period key."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "period", "sentiment", "theme", "product_area", "rating"])
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    out["period"] = out["date"].dt.strftime("%Y-%m")
    out["rating"] = pd.to_numeric(out.get("rating"), errors="coerce")
    out["sentiment"] = out["sentiment"].fillna("").astype(str).str.lower()
    out["theme"] = out["theme"].fillna("unknown").astype(str)
    out["product_area"] = out["product_area"].fillna("unknown").astype(str)
    return out


def periods(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return sorted(frame["period"].unique().tolist())


def filter_period(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    return frame[frame["period"] == period]


def negatives(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["sentiment"] == NEGATIVE]


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def period_summary(frame: pd.DataFrame, period: str) -> PeriodSummary:
    subset = filter_period(frame, period)
    neg = negatives(subset)
    counts = neg["theme"].value_counts()
    top_theme = str(counts.index[0]) if len(counts) else None
    top_share = float(counts.iloc[0]) / len(neg) * 100 if len(neg) and len(counts) else 0.0
    return PeriodSummary(
        period=period,
        total=int(len(subset)),
        negative=int(len(neg)),
        neutral=int((subset["sentiment"] == NEUTRAL).sum()),
        positive=int((subset["sentiment"] == POSITIVE).sum()),
        avg_rating=float(subset["rating"].mean()) if subset["rating"].notna().any() else 0.0,
        top_theme=top_theme,
        top_theme_share=top_share,
    )


def all_period_summaries(frame: pd.DataFrame) -> list[PeriodSummary]:
    return [period_summary(frame, period) for period in periods(frame)]


def theme_count_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Negative-feedback counts per theme per period."""
    neg = negatives(frame)
    if neg.empty:
        return pd.DataFrame()
    matrix = neg.pivot_table(
        index="theme", columns="period", values="feedback_id", aggfunc="count"
    ).fillna(0).astype(int)
    return matrix.reindex(sorted(matrix.index))


def complaint_share_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Each theme's share (%) of that period's negative feedback."""
    matrix = theme_count_matrix(frame)
    if matrix.empty:
        return matrix
    totals = matrix.sum(axis=0)
    return (matrix / totals * 100).round(1)


def area_count_matrix(frame: pd.DataFrame, complaint_only: bool = True) -> pd.DataFrame:
    source = negatives(frame) if complaint_only else frame
    if source.empty:
        return pd.DataFrame()
    matrix = source.pivot_table(
        index="product_area", columns="period", values="feedback_id", aggfunc="count"
    ).fillna(0).astype(int)
    return matrix.reindex(sorted(matrix.index))


def sentiment_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """Sentiment counts + average rating per period (a tidy frame for charts)."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("period")
    out = pd.DataFrame(
        {
            "total": grouped.size(),
            "negative": grouped["sentiment"].apply(lambda s: (s == NEGATIVE).sum()),
            "neutral": grouped["sentiment"].apply(lambda s: (s == NEUTRAL).sum()),
            "positive": grouped["sentiment"].apply(lambda s: (s == POSITIVE).sum()),
            "avg_rating": grouped["rating"].mean().round(2),
        }
    )
    out["negative_share"] = (out["negative"] / out["total"] * 100).round(1)
    return out


def theme_period_series(frame: pd.DataFrame, theme: str, metric: str = "count") -> pd.Series:
    """One theme's negative count or complaint share over time."""
    matrix = theme_count_matrix(frame)
    if matrix.empty or theme not in matrix.index:
        return pd.Series(dtype=float)
    if metric == "count":
        return matrix.loc[theme]
    shares = complaint_share_matrix(frame)
    return shares.loc[theme] if theme in shares.index else pd.Series(dtype=float)


def trend_direction(values: Iterable[float], tolerance: float = 0.5) -> str:
    """Classify a short series as rising / falling / stable."""
    series = [float(v) for v in values]
    if len(series) < 2:
        return "stable"
    delta = series[-1] - series[0]
    if delta > tolerance:
        return "rising"
    if delta < -tolerance:
        return "falling"
    return "stable"


def emerging_signals(
    frame: pd.DataFrame,
    latest_period: str | None = None,
    *,
    min_delta_pts: float = 8.0,
    min_latest_count: int = 4,
    min_latest_share: float = 12.0,
) -> list[EmergingSignal]:
    """Detect themes whose complaint share moved sharply into the latest period.

    The baseline is the mean complaint share across all *earlier* periods, which is
    what makes this a genuine time comparison rather than a single-period ranking.
    """
    available = periods(frame)
    if not available:
        return []
    latest_period = latest_period or available[-1]
    if latest_period not in available:
        latest_period = available[-1]
    baseline_periods = [p for p in available if p < latest_period]
    if not baseline_periods:
        return []

    shares = complaint_share_matrix(frame)
    counts = theme_count_matrix(frame)
    if shares.empty or latest_period not in shares.columns:
        return []

    signals: list[EmergingSignal] = []
    latest_neg = negatives(filter_period(frame, latest_period))

    for theme in shares.index:
        latest_share = float(shares.loc[theme, latest_period])
        baseline_share = float(shares.loc[theme, baseline_periods].mean())
        delta = latest_share - baseline_share
        latest_count = int(counts.loc[theme, latest_period]) if latest_period in counts.columns else 0

        if latest_count < min_latest_count or latest_share < min_latest_share:
            continue
        if abs(delta) < min_delta_pts:
            continue

        latest_rows = latest_neg[latest_neg["theme"] == theme]
        baseline_rows = negatives(frame[frame["period"].isin(baseline_periods)])
        baseline_rows = baseline_rows[baseline_rows["theme"] == theme]
        area = (
            str(latest_rows["product_area"].iloc[0])
            if not latest_rows.empty
            else PRODUCT_AREAS[0]
        )

        signals.append(
            EmergingSignal(
                theme=theme,
                product_area=area,
                direction="emerging" if delta > 0 else "improving",
                latest_period=latest_period,
                latest_count=latest_count,
                latest_share=latest_share,
                baseline_share=baseline_share,
                share_delta_pts=delta,
                latest_avg_rating=float(latest_rows["rating"].mean())
                if latest_rows["rating"].notna().any()
                else 0.0,
                baseline_avg_rating=float(baseline_rows["rating"].mean())
                if baseline_rows["rating"].notna().any()
                else 0.0,
                per_period_counts={
                    period: int(counts.loc[theme, period])
                    for period in counts.columns
                    if theme in counts.index
                },
            )
        )

    signals.sort(key=lambda s: abs(s.share_delta_pts), reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Before / after a product change
# ---------------------------------------------------------------------------


def before_after(
    frame: pd.DataFrame,
    *,
    change_date: str | date | datetime,
    change_name: str,
    product_area: str,
    theme: str | None = None,
    window_days: int = 28,
) -> OutcomeComparison:
    """Compare complaint volume and rating before vs after a recorded change.

    The comparison is scoped to the changed product area (and its theme when known),
    and it is also expressed as a *share* of all complaints in each window so a
    change in overall feedback volume cannot masquerade as a change in this area.
    """
    change_dt = pd.Timestamp(change_date)
    start_before = change_dt - timedelta(days=window_days)
    end_after = change_dt + timedelta(days=window_days)

    prepared = frame.copy()

    def window_stats(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        in_window = prepared[(prepared["date"] >= start) & (prepared["date"] < end)]
        all_neg = in_window[in_window["sentiment"] == NEGATIVE]
        # Masks are built from the window itself so the boolean index always aligns.
        in_area = in_window["product_area"] == product_area
        if theme:
            in_area = in_area & (in_window["theme"] == theme)
        subject = in_window[in_area & (in_window["sentiment"] == NEGATIVE)]
        count = int(len(subject))
        total_neg = int(len(all_neg))
        share = (count / total_neg * 100) if total_neg else 0.0
        return {
            "window_start": start.date().isoformat(),
            "window_end": end.date().isoformat(),
            "complaints": count,
            "all_complaints": total_neg,
            "share_of_complaints": round(share, 1),
            "avg_rating": round(float(subject["rating"].mean()), 2)
            if count and subject["rating"].notna().any()
            else 0.0,
            "total_feedback": int(len(in_window)),
        }

    before = window_stats(start_before, change_dt)
    after = window_stats(change_dt, end_after)

    share_delta = float(after["share_of_complaints"]) - float(before["share_of_complaints"])
    count_delta = int(after["complaints"]) - int(before["complaints"])
    rating_delta = float(after["avg_rating"]) - float(before["avg_rating"])

    if share_delta <= -5:
        verdict = "improved"
    elif share_delta >= 5:
        verdict = "worsened"
    else:
        verdict = "no clear change"

    # Ratings run 1 (worst) to 5 (best). A *falling* average is the sign that the
    # experience got worse; expose that interpretation so no reader (or model) ever
    # has to guess which direction is good. `rating_text` is always phrased from
    # the customer's point of view.
    if before["complaints"] and after["complaints"] and rating_delta < -0.1:
        rating_read = "worsened"
    elif before["complaints"] and after["complaints"] and rating_delta > 0.1:
        rating_read = "improved"
    else:
        rating_read = "effectively unchanged"
    rating_text = (
        f"average rating moved {before['avg_rating']:.2f} -> {after['avg_rating']:.2f} "
        f"on the 1-5 scale ({rating_read}; 1 is worst, 5 is best)"
    )

    return OutcomeComparison(
        change_name=change_name,
        change_date=change_dt.date().isoformat(),
        product_area=product_area,
        window_days=window_days,
        theme=theme,
        before=before,
        after=after,
        share_delta_pts=share_delta,
        count_delta=count_delta,
        rating_delta=rating_delta,
        verdict=verdict,
        rating_text=rating_text,
    )


def recent_complaints(frame: pd.DataFrame, period: str, limit: int = 5, theme: str | None = None) -> list[dict[str, Any]]:
    """Most recent negative feedback, optionally for one theme, as dicts."""
    subset = negatives(filter_period(frame, period))
    if theme:
        subset = subset[subset["theme"] == theme]
    subset = subset.sort_values("date", ascending=False).head(limit)
    return [
        {
            "date": row["date"].date().isoformat(),
            "product_area": row["product_area"],
            "theme": row["theme"],
            "rating": int(row["rating"]) if pd.notna(row["rating"]) else None,
            "text": str(row["feedback_text"]),
            "channel": row.get("channel", ""),
        }
        for _, row in subset.iterrows()
    ]


# ---------------------------------------------------------------------------
# Text rendering used as LLM context and as retained memory content
# ---------------------------------------------------------------------------


def render_period_brief(frame: pd.DataFrame, period: str) -> str:
    """A compact, purely factual brief for one period (LLM context, not memory)."""
    summary = period_summary(frame, period)
    shares = complaint_share_matrix(frame)
    lines = [
        f"Period {period}: {summary.total} feedback items "
        f"({summary.negative} negative, {summary.neutral} neutral, {summary.positive} positive), "
        f"average rating {summary.avg_rating:.2f}/5."
    ]
    if not shares.empty and period in shares.columns:
        ranked = shares[period].sort_values(ascending=False)
        lines.append("Share of negative feedback by topic:")
        for theme, share in ranked.items():
            if share <= 0:
                continue
            count = int(theme_count_matrix(frame).loc[theme, period])
            lines.append(f"  - {theme}: {share:.1f}% ({count} complaints)")
    return "\n".join(lines)


def render_theme_series(frame: pd.DataFrame, theme: str) -> str:
    """Complaint count + share for one theme across every period."""
    counts = theme_period_series(frame, theme, "count")
    shares = theme_period_series(frame, theme, "share")
    if counts.empty:
        return f"No complaints recorded for {theme}."
    parts = [
        f"{period}: {int(counts[period])} complaints ({float(shares[period]):.1f}% of all complaints)"
        for period in counts.index
    ]
    return f"{theme} over time -> " + "; ".join(parts)


__all__ = [
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
    "EmergingSignal",
    "OutcomeComparison",
    "PeriodSummary",
    "all_period_summaries",
    "area_count_matrix",
    "before_after",
    "complaint_share_matrix",
    "emerging_signals",
    "filter_period",
    "negatives",
    "period_summary",
    "periods",
    "prepare",
    "recent_complaints",
    "render_period_brief",
    "render_theme_series",
    "sentiment_trend",
    "theme_count_matrix",
    "theme_period_series",
    "trend_direction",
    "THEMES",
]
