"""Deterministic analytics: aggregation, shares, emerging detection, outcomes."""

from __future__ import annotations

import pandas as pd
import pytest

from services import feedback_analyzer as analyzer
from services import trend_analyzer as trends


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return trends.prepare(analyzer.load_feedback())


def test_prepare_adds_period_and_types(frame: pd.DataFrame) -> None:
    assert frame["period"].iloc[0] == frame["date"].iloc[0].strftime("%Y-%m")
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert pd.api.types.is_numeric_dtype(frame["rating"])


def test_periods_are_sorted(frame: pd.DataFrame) -> None:
    assert trends.periods(frame) == sorted(trends.periods(frame))


def test_period_summary_counts_add_up(frame: pd.DataFrame) -> None:
    for period in trends.periods(frame):
        summary = trends.period_summary(frame, period)
        assert summary.total == summary.negative + summary.neutral + summary.positive
        assert 1.0 <= summary.avg_rating <= 5.0
        assert summary.total > 0


def test_complaint_shares_are_percentages_of_negative_feedback(frame: pd.DataFrame) -> None:
    shares = trends.complaint_share_matrix(frame)
    counts = trends.theme_count_matrix(frame)
    assert not shares.empty
    for period in shares.columns:
        assert shares[period].sum() == pytest.approx(100.0, abs=0.5)
        assert shares[period].notna().all()
    # Shares must be built from negative feedback only.
    negatives = trends.negatives(frame)
    assert counts["2026-01"].sum() == int((negatives["period"] == "2026-01").sum())


def test_checkout_complaints_fall_and_payments_rise_across_the_demo_story(
    frame: pd.DataFrame,
) -> None:
    shares = trends.complaint_share_matrix(frame)
    assert shares.loc["checkout_friction", "2026-01"] > shares.loc["checkout_friction", "2026-03"]
    assert shares.loc["payment_reliability", "2026-03"] > shares.loc["payment_reliability", "2026-01"]


def test_payment_reliability_has_historical_precedent(frame: pd.DataFrame) -> None:
    """The demo must be able to answer 'have we seen this before?' with a real yes."""
    counts = trends.theme_count_matrix(frame)
    assert counts.loc["payment_reliability", "2025-11"] > 0
    assert counts.loc["payment_reliability", "2025-12"] > 0


def test_emerging_signals_flags_payments_and_checkout(frame: pd.DataFrame) -> None:
    signals = {signal.theme: signal for signal in trends.emerging_signals(frame, "2026-03")}
    assert "payment_reliability" in signals
    assert signals["payment_reliability"].direction == "emerging"
    assert signals["payment_reliability"].share_delta_pts > 8
    assert signals["payment_reliability"].latest_count > 0
    checkout = signals.get("checkout_friction")
    assert checkout is not None and checkout.direction == "improving"


def test_emerging_signals_need_earlier_periods(frame: pd.DataFrame) -> None:
    first_period = trends.periods(frame)[0]
    only_first = frame[frame["period"] == first_period]
    assert trends.emerging_signals(only_first, first_period) == []


def test_emerging_signal_serialisation(frame: pd.DataFrame) -> None:
    signal = trends.emerging_signals(frame, "2026-03")[0]
    payload = signal.as_dict()
    for key in ["theme", "direction", "latest_share", "baseline_share", "share_delta_pts", "per_period_counts"]:
        assert key in payload
    assert payload["direction"] in {"emerging", "improving"}


def test_before_after_measures_the_checkout_change(frame: pd.DataFrame) -> None:
    comparison = trends.before_after(
        frame,
        change_date="2026-02-15",
        change_name="Checkout V2",
        product_area="Checkout",
        theme="checkout_friction",
        window_days=28,
    )
    assert comparison.verdict == "improved"
    assert comparison.share_delta_pts < 0
    assert comparison.before["complaints"] > comparison.after["complaints"]
    assert comparison.before["window_end"] == "2026-02-15"
    assert comparison.before["window_start"] == "2026-01-18"
    for window in (comparison.before, comparison.after):
        assert 0 <= window["share_of_complaints"] <= 100


def test_before_after_handles_a_date_with_no_feedback(frame: pd.DataFrame) -> None:
    comparison = trends.before_after(
        frame,
        change_date="2030-01-01",
        change_name="Future change",
        product_area="Checkout",
        window_days=14,
    )
    assert comparison.before["complaints"] == 0
    assert comparison.after["complaints"] == 0
    assert comparison.share_delta_pts == 0
    assert comparison.verdict == "no clear change"


def test_theme_series_and_trend_direction(frame: pd.DataFrame) -> None:
    counts = trends.theme_period_series(frame, "payment_reliability", "count")
    shares = trends.theme_period_series(frame, "payment_reliability", "share")
    assert len(counts) == len(trends.periods(frame))
    assert trends.trend_direction(shares) == "rising"
    assert trends.trend_direction([]) == "stable"
    assert trends.trend_direction([1.0, 1.0]) == "stable"
    assert trends.trend_direction([3.0, 1.0]) == "falling"


def test_sentiment_trend_has_shares_and_ratings(frame: pd.DataFrame) -> None:
    trend = trends.sentiment_trend(frame)
    assert {"total", "negative", "positive", "avg_rating", "negative_share"}.issubset(trend.columns)
    assert trend["negative_share"].between(0, 100).all()


def test_recent_complaints_are_negative_and_newest_first(frame: pd.DataFrame) -> None:
    recent = trends.recent_complaints(frame, "2026-03", limit=4, theme="payment_reliability")
    assert recent, "expected payment complaints in March"
    assert all(row["theme"] == "payment_reliability" for row in recent)
    dates = [row["date"] for row in recent]
    assert dates == sorted(dates, reverse=True)


def test_render_helpers_are_citable(frame: pd.DataFrame) -> None:
    brief = trends.render_period_brief(frame, "2026-01")
    assert "2026-01" in brief and "%" in brief
    series = trends.render_theme_series(frame, "checkout_friction")
    assert "checkout_friction over time" in series
