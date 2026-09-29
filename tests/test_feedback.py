"""Feedback loading, schema handling and memory-document construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.config import TAG_FEEDBACK, TAG_PRODUCT_CHANGE
from services import feedback_analyzer as analyzer
from services import trend_analyzer as trends


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    raw = analyzer.load_feedback()
    return trends.prepare(raw)


def test_dataset_has_the_documented_schema(frame: pd.DataFrame) -> None:
    for column in analyzer.FEEDBACK_COLUMNS:
        assert column in frame.columns, f"missing column {column}"
    assert len(frame) > 100
    assert frame["feedback_id"].is_unique
    assert frame["sentiment"].isin(["negative", "neutral", "positive"]).all()
    assert frame["rating"].between(1, 5).all()


def test_dataset_covers_multiple_periods_and_the_demo_areas(frame: pd.DataFrame) -> None:
    assert {"2026-01", "2026-02", "2026-03"}.issubset(set(frame["period"]))
    assert {"Checkout", "Payments"}.issubset(set(frame["product_area"]))


def test_product_changes_are_pm_recorded() -> None:
    changes = analyzer.load_product_changes()
    assert not changes.empty
    assert changes["change_name"].str.len().gt(0).all()
    assert changes["problem_targeted"].str.len().gt(0).all()


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_sentiment_from_rating_is_deterministic(rating: int, expected: str) -> None:
    assert analyzer.sentiment_from_rating(rating) == expected


def test_sentiment_from_rating_tolerates_bad_input() -> None:
    assert analyzer.sentiment_from_rating(None) == "neutral"
    assert analyzer.sentiment_from_rating("not a number") == "neutral"


def test_ensure_sentiment_repairs_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "feedback_id": ["a", "b", "c"],
            "date": ["2026-01-01"] * 3,
            "feedback_text": ["x", "y", "z"],
            "rating": [1, 3, 5],
            "sentiment": ["", "unknown", ""],
        }
    )
    fixed = analyzer.ensure_sentiment(frame)
    assert list(fixed["sentiment"]) == ["negative", "neutral", "positive"]


def test_unlabeled_rows_detects_unknown_themes(frame: pd.DataFrame) -> None:
    assert analyzer.unlabeled_rows(frame).empty, "demo dataset ships fully labelled"
    dirty = frame.copy()
    dirty.loc[0, "theme"] = "something_new"
    assert len(analyzer.unlabeled_rows(dirty)) == 1


def test_build_period_memory_carries_facts_dates_and_evidence(frame: pd.DataFrame) -> None:
    period_rows = trends.filter_period(frame, "2026-01")
    document = analyzer.build_period_feedback_memory(period_rows, period="2026-01")

    assert document.kind == TAG_FEEDBACK
    assert "2026-01" in document.content
    assert "checkout_friction" in document.content
    assert "Checkout" in document.content, "per-area breakdown must survive"
    assert '"' in document.content, "verbatim evidence should be quoted"
    assert TAG_FEEDBACK in document.tags
    assert "period:2026-01" in document.tags
    assert document.metadata["negative_count"] == str(
        (period_rows["sentiment"] == "negative").sum()
    )
    assert document.metadata["dominant_theme"] == "checkout_friction"
    assert document.document_id == "feedback-2026-01"
    # Every value Hindsight stores is a string, as its metadata type requires.
    assert all(isinstance(value, str) for value in document.metadata.values())


def test_period_memory_spans_areas_so_area_questions_stay_answerable(frame: pd.DataFrame) -> None:
    period_rows = trends.filter_period(frame, "2026-01")
    document = analyzer.build_period_feedback_memory(period_rows, period="2026-01")
    areas = set(period_rows["product_area"])
    for area in areas:
        assert area in document.content, f"{area} missing from the period memory"
    # No `area:` tag: one document covers several areas, and Hindsight propagates a
    # document's tags to every fact it extracts, so an area tag here would match all.
    assert not any(tag.startswith("area:") for tag in document.tags)


def test_period_memory_fits_in_one_hindsight_chunk(frame: pd.DataFrame) -> None:
    """One chunk means one extraction call, which is what keeps the free tier viable.

    Hindsight chunks documents at `retain_chunk_size` (3,000 characters by default)
    and runs one LLM extraction call per chunk, each costing ~1.7k tokens of fixed
    prompt overhead. Every period's document must stay inside a single chunk.
    """
    for period in trends.periods(frame):
        document = analyzer.build_period_feedback_memory(
            trends.filter_period(frame, period), period=period
        )
        assert len(document.content) <= analyzer.MAX_MEMORY_CONTENT_CHARS, (
            f"{period} memory is {len(document.content)} chars and would be chunked into "
            f"multiple extraction calls"
        )


def test_build_product_change_memory_includes_target_and_expectation() -> None:
    changes = analyzer.load_product_changes()
    document = analyzer.build_product_change_memory(changes.iloc[0])
    assert document.kind == TAG_PRODUCT_CHANGE
    assert "Checkout V2" in document.content
    assert "Release date:" in document.content
    assert changes.iloc[0]["problem_targeted"] in document.content
    assert changes.iloc[0]["expected_outcome"] in document.content
    assert document.metadata["release_date"] == "2026-02-15"


def test_representative_quotes_prefers_worst_rated(frame: pd.DataFrame) -> None:
    quotes = analyzer.representative_quotes(frame, limit=3)
    assert len(quotes) == 3
    assert all(quote["rating"] == 1 for quote in quotes)
    assert all(quote["text"].strip() for quote in quotes)


def test_append_product_change_assigns_ids_and_persists(tmp_path: Path) -> None:
    target = tmp_path / "product_changes.csv"
    analyzer.append_product_change(
        {
            "change_id": "",
            "date": "2026-04-01",
            "change_name": "Payments Retry",
            "product_area": "Payments",
            "problem_targeted": "UPI reliability",
            "expected_outcome": "Fewer failed UPI payments",
        },
        path=target,
    )
    analyzer.append_product_change(
        {
            "change_id": "",
            "date": "2026-04-20",
            "change_name": "Payments Retry 2",
            "product_area": "Payments",
            "problem_targeted": "UPI reliability",
            "expected_outcome": "Fewer failed UPI payments",
        },
        path=target,
    )
    saved = pd.read_csv(target)
    assert list(saved["change_id"]) == ["CHG-001", "CHG-002"]
    assert saved["date"].iloc[0] == "2026-04-01"


def test_load_feedback_explains_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_dataset.py"):
        analyzer.load_feedback(path=tmp_path / "nope.csv")
