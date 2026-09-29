"""Feedback understanding and the feedback half of PulseMind's memory model.

Two responsibilities:

1. Load and normalise the feedback dataset into the documented schema, deriving
   sentiment deterministically from the star rating, and use Groq only for the
   genuinely semantic job: mapping free text onto PulseMind's controlled theme
   vocabulary when a row arrives unlabelled.
2. Turn a group of feedback into the *document* that gets retained in Hindsight, so
   what the agent remembers is the same structure it reasons about (dates, product
   area, topic mix, ratings and verbatim evidence).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import (
    MEMORY_KIND_LABELS,
    PRODUCT_AREAS,
    SENTIMENTS,
    TAG_FEEDBACK,
    TAG_PRODUCT_CHANGE,
    THEMES,
    THEME_LABELS,
    Settings,
    get_settings,
)
from core.groq_client import GroqLLM
from services import trend_analyzer as trends

logger = logging.getLogger(__name__)

FEEDBACK_COLUMNS = [
    "feedback_id",
    "date",
    "customer_id",
    "channel",
    "product_area",
    "feedback_text",
    "rating",
    "sentiment",
    "theme",
]

PRODUCT_CHANGE_COLUMNS = [
    "change_id",
    "date",
    "change_name",
    "product_area",
    "problem_targeted",
    "expected_outcome",
    # Optional provenance columns, filled automatically for GitHub-detected changes:
    "source",
    "github_repo",
    "github_number",
    "github_kind",
    "github_version",
    "github_url",
]


@dataclass
class MemoryDocument:
    """Everything needed to retain one memory, built by a service layer."""

    kind: str
    content: str
    when: str
    context: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None

    @property
    def kind_label(self) -> str:
        return MEMORY_KIND_LABELS.get(self.kind, self.kind)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_csv(path: Path, required: list[str], *, dtype: Any = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python data/prepare_dataset.py demo` to "
            "generate the demo dataset (or `... kaggle` to import the real one)."
        )
    # `dtype=str` matters for files whose identifier columns hold digits next to empty
    # values: pandas would otherwise infer float64 and turn a PR number "42" into
    # "42.0" the moment it is stringified.
    frame = pd.read_csv(path, dtype=dtype)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def load_feedback(settings: Settings | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load feedback.csv into the documented PulseMind schema."""
    settings = settings or get_settings()
    frame = _read_csv(path or settings.feedback_csv, ["feedback_id", "date", "feedback_text"])
    for column in FEEDBACK_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[FEEDBACK_COLUMNS].copy()
    frame["feedback_text"] = frame["feedback_text"].fillna("").astype(str)
    frame["product_area"] = frame["product_area"].fillna("").astype(str)
    frame["theme"] = frame["theme"].fillna("").astype(str)
    frame["sentiment"] = frame["sentiment"].fillna("").astype(str).str.lower()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    return ensure_sentiment(frame)


def _clean_identifier(value: Any) -> str:
    """Repair a numeric-looking identifier that was previously stored as a float.

    Older rows (and any CSV written by a tool that lossily cast the column) can hold
    "42.0" where the real PR number is "42". Normalise at load time so both the UI and
    the retained memory show the identifier GitHub actually used.
    """
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def load_product_changes(settings: Settings | None = None, path: Path | None = None) -> pd.DataFrame:
    """Load product_changes.csv — changes the Product Manager has recorded."""
    settings = settings or get_settings()
    # Every column in this file is text; `date` is parsed explicitly below. Reading it
    # as text keeps GitHub identifiers (PR numbers, versions) exactly as written.
    frame = _read_csv(
        path or settings.product_changes_csv,
        ["change_id", "date", "change_name"],
        dtype=str,
    )
    for column in PRODUCT_CHANGE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[PRODUCT_CHANGE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in [
        "change_name",
        "product_area",
        "problem_targeted",
        "expected_outcome",
        "source",
        "github_repo",
        "github_number",
        "github_kind",
        "github_version",
        "github_url",
    ]:
        frame[column] = frame[column].fillna("").astype(str)
    frame["github_number"] = frame["github_number"].map(_clean_identifier)
    return frame


def append_product_change(
    change: dict[str, Any],
    settings: Settings | None = None,
    path: Path | None = None,
) -> Path:
    """Persist a new product change. The PM records changes; the AI never invents them."""
    settings = settings or get_settings()
    target = path or settings.product_changes_csv
    existing = load_product_changes(settings, target) if target.exists() else pd.DataFrame(
        columns=PRODUCT_CHANGE_COLUMNS
    )
    row = {column: change.get(column, "") for column in PRODUCT_CHANGE_COLUMNS}
    if not row["change_id"]:
        existing_ids = [
            int(match.group(1))
            for match in (re.match(r"CHG-(\d+)", str(value)) for value in existing.get("change_id", []))
            if match
        ]
        row["change_id"] = f"CHG-{(max(existing_ids) + 1) if existing_ids else 1:03d}"
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined.to_csv(target, index=False)
    return target


# ---------------------------------------------------------------------------
# Deterministic sentiment + semantic labelling
# ---------------------------------------------------------------------------


def sentiment_from_rating(rating: Any) -> str:
    """Deterministic sentiment from the star rating (no LLM required)."""
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return "neutral"
    if value <= 2:
        return trends.NEGATIVE
    if value == 3:
        return trends.NEUTRAL
    return trends.POSITIVE


def ensure_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/unknown sentiment deterministically from the rating."""
    out = frame.copy()
    if "rating" not in out.columns:
        out["rating"] = pd.NA
    derived = out["rating"].apply(sentiment_from_rating)
    current = out.get("sentiment", pd.Series("", index=out.index)).astype(str).str.lower()
    invalid = ~current.isin(SENTIMENTS)
    out["sentiment"] = current.where(~invalid, derived)
    return out


def unlabeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Feedback whose theme Groq still needs to extract."""
    if frame.empty:
        return frame
    known = frame["theme"].astype(str).isin(THEMES)
    return frame[~known]


def label_themes_with_llm(
    frame: pd.DataFrame,
    llm: GroqLLM,
    *,
    batch_size: int = 25,
    max_items: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    """Use Groq to map unlabelled feedback onto the controlled theme vocabulary.

    This is the one place raw customer text is interpreted rather than counted, so
    it is batched to stay inside Groq's free-tier rate limits.
    """
    out = frame.copy()
    notes: list[str] = []
    pending = unlabeled_rows(out)
    if pending.empty:
        return out, ["All feedback already carries a theme; no LLM labelling needed."]

    pending = pending.head(max_items)
    system = (
        "You classify customer feedback for a product manager. You must only use the "
        "supplied controlled vocabularies. Respond with JSON only."
    )
    labelled = 0

    for start in range(0, len(pending), batch_size):
        chunk = pending.iloc[start : start + batch_size]
        items = [
            {
                "feedback_id": str(row["feedback_id"]),
                "product_area_hint": str(row["product_area"]),
                "text": str(row["feedback_text"])[:400],
            }
            for _, row in chunk.iterrows()
        ]
        prompt = (
            "Classify each feedback item.\n\n"
            f"Allowed themes: {', '.join(THEMES)}\n"
            f"Allowed product areas: {', '.join(PRODUCT_AREAS)}\n"
            f"Allowed sentiments: {', '.join(SENTIMENTS)}\n\n"
            "For every item return: feedback_id, theme (from the allowed themes), "
            "product_area (from the allowed areas), sentiment, and a one-line "
            "rationale (max 12 words).\n\n"
            'Respond as JSON: {"items": [{"feedback_id": "...", "theme": "...", '
            '"product_area": "...", "sentiment": "...", "rationale": "..."}]}\n\n'
            f"Feedback items:\n{items}"
        )
        payload = llm.complete_json_safe(
            prompt,
            system=system,
            fallback={"items": []},
            max_tokens=2000,
            label=f"theme-extraction[{start}:{start + len(chunk)}]",
        )
        for item in payload.get("items", []) if isinstance(payload, dict) else []:
            feedback_id = str(item.get("feedback_id", ""))
            theme = str(item.get("theme", ""))
            sentiment = str(item.get("sentiment", "")).lower()
            area = str(item.get("product_area", ""))
            mask = out["feedback_id"].astype(str) == feedback_id
            if not mask.any():
                continue
            if theme in THEMES:
                out.loc[mask, "theme"] = theme
            if area in PRODUCT_AREAS:
                out.loc[mask, "product_area"] = area
            if sentiment in SENTIMENTS and pd.isna(out.loc[mask, "rating"]).all():
                out.loc[mask, "sentiment"] = sentiment
            labelled += int(mask.sum())

        notes.append(f"Groq classified {labelled} of {len(pending)} unlabelled items.")

    return out, notes


# ---------------------------------------------------------------------------
# Memory document builders
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def stringify_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Hindsight's `metadata` parameter is `dict[str, str]`, so normalise here.

    Doing it at document-build time (rather than at the client boundary) keeps every
    memory self-describing and makes the values that actually land in the bank visible
    in Memory Explorer.
    """
    return {str(key): str(value) for key, value in metadata.items()}


def representative_quotes(frame: pd.DataFrame, limit: int = 4) -> list[dict[str, Any]]:
    """Pick the clearest verbatim evidence: worst-rated, longest, de-duplicated."""
    if frame.empty:
        return []
    ranked = frame.copy()
    ranked["_len"] = ranked["feedback_text"].astype(str).str.len()
    ranked = ranked.sort_values(["rating", "_len"], ascending=[True, False])
    seen: set[str] = set()
    quotes: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        text = str(row["feedback_text"]).strip()
        fingerprint = text[:60].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        quotes.append(
            {
                "date": str(row["date"])[:10],
                "rating": int(row["rating"]) if pd.notna(row["rating"]) else None,
                "channel": str(row.get("channel", "") or "unknown"),
                "text": text,
            }
        )
        if len(quotes) >= limit:
            break
    return quotes


# Hindsight chunks documents at 3,000 characters (its default `retain_chunk_size`) and
# runs one LLM extraction call per chunk. A period document that stays under this
# therefore costs exactly one extraction call. The tests assert this limit.
MAX_MEMORY_CONTENT_CHARS = 3000


def build_period_feedback_memory(period_frame: pd.DataFrame, *, period: str) -> MemoryDocument:
    """Turn one period's feedback into a single retainable memory document.

    One document per period, rather than per product area or per review, because
    Hindsight runs an LLM fact-extraction pass over every retained document and that
    pass costs ~1.7k tokens of fixed prompt overhead *per call* (measured against the
    installed CONCISE extraction prompt) regardless of how small the document is.
    Retaining one document per product area per period paid that overhead ~7 times per
    period for no gain in recall quality. Keeping the document under one chunk keeps
    each period at exactly one extraction call.

    The document still carries everything the agent reasons about: volume, sentiment
    mix, average rating, every topic with its share, a per-area breakdown, and quoted
    customer verbatims as evidence.
    """
    neg = period_frame[period_frame["sentiment"] == trends.NEGATIVE]
    total = len(period_frame)
    negative = len(neg)
    positive = int((period_frame["sentiment"] == trends.POSITIVE).sum())
    neutral = int((period_frame["sentiment"] == trends.NEUTRAL).sum())
    avg_rating = (
        float(period_frame["rating"].mean()) if period_frame["rating"].notna().any() else 0.0
    )

    topic_lines: list[str] = []
    dominant_theme = "unknown"
    dominant_share = 0.0
    if negative:
        counts = neg["theme"].value_counts()
        dominant_theme = str(counts.index[0])
        dominant_share = float(counts.iloc[0]) / negative * 100
        for theme, count in counts.items():
            area = _dominant_area(neg, str(theme))
            topic_lines.append(
                f"  - {theme} ({THEME_LABELS.get(theme, theme)}): {int(count)} complaints, "
                f"{int(count) / negative * 100:.1f}% of complaints, area {area}"
            )

    # A per-area breakdown keeps single-area questions answerable from one memory.
    area_lines: list[str] = []
    for area, group in period_frame.groupby("product_area"):
        area_negative = int((group["sentiment"] == trends.NEGATIVE).sum())
        area_top = _dominant_area(group[group["sentiment"] == trends.NEGATIVE], None)
        area_lines.append(
            f"  - {area}: {len(group)} items, {area_negative} negative, top topic {area_top}"
        )

    channel_counts = period_frame.get("channel", pd.Series(dtype=str)).value_counts()
    channels = (
        ", ".join(f"{name} ({int(count)})" for name, count in channel_counts.items()) or "unknown"
    )

    quotes = representative_quotes(neg if not neg.empty else period_frame, limit=4)
    quote_lines = [
        f'  - {q["date"]}, rating {q["rating"]}/5, {q["channel"]}: "{q["text"]}"' for q in quotes
    ]

    date_min = str(period_frame["date"].min())[:10]
    date_max = str(period_frame["date"].max())[:10]

    content = "\n".join(
        [
            f"Customer feedback report for period {period} ({date_min} to {date_max}).",
            "",
            f"Volume: {total} feedback items in this period.",
            f"Sentiment: {negative} negative, {neutral} neutral, {positive} positive.",
            f"Average star rating: {avg_rating:.2f} out of 5.",
            "",
            "Complaint topics as a share of all complaints this period:"
            if topic_lines
            else "No negative feedback was recorded in this period.",
            *topic_lines,
            "",
            "Product areas this period:",
            *area_lines,
            "",
            "Representative customer verbatims:",
            *quote_lines,
            "",
            f"Channels represented: {channels}.",
        ]
    )

    # Deliberately no `area:` tag: this document spans several areas, and Hindsight
    # propagates a document's tags to every fact it extracts, so tagging it with
    # several areas would make area filtering match everything and mean nothing.
    # Area questions are answered semantically (the recall queries name the area) and
    # the per-area breakdown is in the text.
    tags = [TAG_FEEDBACK, f"period:{period}"]

    metadata = stringify_metadata(
        {
            "period": period,
            "date_from": date_min,
            "date_to": date_max,
            "total_feedback": total,
            "negative_count": negative,
            "positive_count": positive,
            "neutral_count": neutral,
            "avg_rating": f"{avg_rating:.2f}",
            "dominant_theme": dominant_theme,
            "dominant_theme_share": f"{dominant_share:.1f}",
            "product_areas": ", ".join(sorted(period_frame["product_area"].unique())),
            "channels": channels,
        }
    )

    return MemoryDocument(
        kind=TAG_FEEDBACK,
        content=content,
        when=date_max,
        context=f"Customer feedback collected during {period}",
        tags=tags,
        metadata=metadata,
        document_id=f"feedback-{period}",
    )


def _dominant_area(frame: pd.DataFrame, theme: str | None) -> str:
    """The product area a theme (or the frame as a whole) is dominated by."""
    if frame.empty:
        return "unknown"
    subset = frame if theme is None else frame[frame["theme"] == theme]
    if subset.empty:
        return "unknown"
    return str(subset["product_area"].mode().iloc[0])


def build_product_change_memory(change: pd.Series | dict[str, Any]) -> MemoryDocument:
    """Turn a PM-recorded product change into a retainable document.

    Rows that came from a GitHub webhook carry ``source == 'github'`` plus repository,
    PR/release number, version tag and URL fields; those are preserved verbatim so the
    memory shows the engineering origin instead of attributing the change to the PM.
    """
    row = dict(change)
    change_date = pd.Timestamp(row["date"]).date().isoformat()
    period = change_date[:7]
    area = str(row.get("product_area", "")) or "unknown"
    name = str(row.get("change_name", "unnamed change"))
    source = str(row.get("source", "") or "")
    from_github = source == "github"

    content = "\n".join(
        [
            (
                "Product change detected automatically from GitHub."
                if from_github
                else "Product change recorded by the Product Manager."
            ),
            f"Change name: {name}",
            *(
                [
                    f"Source: GitHub {'Release' if row.get('github_kind') == 'release' else 'pull request'}",
                    f"Repository: {row.get('github_repo', '')}",
                    f"Version / tag: {row.get('github_version', '')}" if row.get("github_version") else "",
                    f"PR number: #{row.get('github_number', '')}" if row.get("github_number") else "",
                    f"GitHub URL: {row.get('github_url', '')}" if row.get("github_url") else "",
                ]
                if from_github
                else []
            ),
            f"Release date: {change_date}",
            f"Product area: {area}",
            f"Problem targeted: {row.get('problem_targeted', '') or 'not specified'}",
            f"Expected outcome: {row.get('expected_outcome', '') or 'not specified'}",
            "",
            "Note: this change was explicitly recorded by the Product Manager; it was "
            "not inferred from customer feedback.",
        ]
    )

    return MemoryDocument(
        kind=TAG_PRODUCT_CHANGE,
        content=content,
        when=change_date,
        context=f"Product change in {area} released on {change_date}",
        tags=[TAG_PRODUCT_CHANGE, f"area:{_slug(area)}", f"period:{period}"]
        + (["source:github"] if from_github else []),
        metadata=stringify_metadata(
            {
                "change_id": row.get("change_id", ""),
                "change_name": name,
                "release_date": change_date,
                "product_area": area,
                "problem_targeted": row.get("problem_targeted", ""),
                "expected_outcome": row.get("expected_outcome", ""),
                "period": period,
                **(
                    {
                        "source": "github",
                        "github_repo": row.get("github_repo", ""),
                        "github_number": row.get("github_number", ""),
                        "github_kind": row.get("github_kind", ""),
                        "github_version": row.get("github_version", ""),
                        "github_url": row.get("github_url", ""),
                    }
                    if from_github
                    else {}
                ),
            }
        ),
        document_id=f"change-{row.get('change_id', _slug(name))}",
    )


def theme_mix(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Tidy theme breakdown for the UI (negative feedback only)."""
    neg = trends.negatives(frame)
    if neg.empty:
        return []
    counts = neg["theme"].value_counts()
    total = int(len(neg))
    return [
        {
            "theme": str(theme),
            "label": THEME_LABELS.get(str(theme), str(theme)),
            "count": int(count),
            "share": round(int(count) / total * 100, 1),
        }
        for theme, count in counts.items()
    ]


def load_all(settings: Settings | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load feedback + product changes, prepared for analysis."""
    settings = settings or get_settings()
    raw_feedback = load_feedback(settings)
    changes = load_product_changes(settings)
    return raw_feedback, trends.prepare(raw_feedback), changes
