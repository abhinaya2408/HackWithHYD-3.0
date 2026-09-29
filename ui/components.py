"""Shared Streamlit components.

Small presentational helpers used by more than one page, so the six pages stay
readable. Nothing here invents data: every function takes real objects returned by
the agent (analytics frames, Hindsight memory hits/records, insights) and renders them.

The visual language comes from :mod:`ui.theme` (the Airbnb ``DESIGN.md`` system), and
those helpers are re-exported here so pages can keep importing a single ``ui`` module.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import MEMORY_KIND_LABELS, THEME_LABELS
from core.hindsight_client import MemoryHit, MemoryRecord
from ui.theme import (
    CHART_COLORS,
    COLORS,
    FONT_STACK,
    ROUNDED,
    SENTIMENT_COLORS,
    SHADOW,
    apply_theme,
    badge,
    badge_row,
    card,
    empty_state,
    key_values,
    page_header,
    retention_badge,
    section,
    source_badge,
)

KIND_ICONS = {
    "feedback": "💬",
    "product_change": "🚀",
    "outcome": "📈",
    "emerging_issue": "⚠️",
    "product_manager_decision": "✅",
    "unknown": "•",
}

# One shared Plotly configuration so every chart reads as the same product.
PLOTLY_LAYOUT = dict(
    margin=dict(l=10, r=10, t=34, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=-0.28, title_text=""),
    font=dict(family=FONT_STACK, size=12, color=COLORS["ink"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=CHART_COLORS,
    title_font=dict(size=15, color=COLORS["ink"]),
    hoverlabel=dict(font=dict(family=FONT_STACK, size=12)),
)

PLOTLY_AXES = dict(
    xaxis=dict(showgrid=False, zeroline=False, linecolor=COLORS["hairline"],
               tickfont=dict(color=COLORS["muted"], size=11)),
    yaxis=dict(gridcolor=COLORS["hairline-soft"], zeroline=False,
               tickfont=dict(color=COLORS["muted"], size=11)),
)


def kind_label(kind: str) -> str:
    return MEMORY_KIND_LABELS.get(kind, kind.replace("_", " ").title())


def theme_label(theme: str) -> str:
    return THEME_LABELS.get(theme, theme)


def memory_card(item: MemoryHit | MemoryRecord, *, expanded: bool = False) -> None:
    """Render one real Hindsight memory (recalled hit or listed memory unit)."""
    kind = getattr(item, "kind", "unknown")
    when = (
        getattr(item, "occurred_start", None)
        or getattr(item, "metadata", {}).get("memory_date")
        or getattr(item, "metadata", {}).get("decision_date")
        or "date not stored"
    )
    title = f"{KIND_ICONS.get(kind, '•')} {kind_label(kind)} · {str(when)[:10]}"
    with st.expander(title, expanded=expanded):
        badges = [badge(kind_label(kind), "primary")]
        if getattr(item, "fact_type", None):
            badges.append(badge(f"fact_type: {item.fact_type}", "muted"))
        state = getattr(item, "state", None)
        if state:
            badges.append(badge(f"state: {state}", "muted"))
        badge_row(badges)

        st.markdown(
            f'<div class="pm-answer">{item.text}</div>', unsafe_allow_html=True
        )
        st.markdown(f"**Memory id:** `{item.id}`")

        tags = getattr(item, "tags", None) or []
        if tags:
            st.caption("tags: " + ", ".join(f"`{tag}`" for tag in tags))
        meta = getattr(item, "metadata", None) or {}
        if meta:
            st.caption("metadata: " + ", ".join(f"{key}={value}" for key, value in meta.items()))
        scores = getattr(item, "scores", None)
        if scores:
            st.caption(
                "retrieval scores: "
                + ", ".join(
                    f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in scores.items()
                )
            )


def evidence_list(items: Iterable[MemoryHit | MemoryRecord], *, empty_message: str) -> int:
    """Render recalled memories and return how many were shown."""
    items = list(items)
    if not items:
        empty_state("Nothing to show here yet", empty_message)
        return 0
    for item in items:
        memory_card(item)
    return len(items)


def metric_row(metrics: list[tuple[str, Any, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def theme_share_chart(share_matrix: pd.DataFrame, title: str = "Complaint share by topic over time") -> None:
    if share_matrix is None or share_matrix.empty:
        empty_state("No complaints to chart", "Charts appear once complaints are loaded.")
        return
    long = share_matrix.reset_index().melt(id_vars="theme", var_name="period", value_name="share")
    long["topic"] = long["theme"].map(theme_label)
    figure = px.line(
        long,
        x="period",
        y="share",
        color="topic",
        markers=True,
        title=title,
        labels={"share": "% of complaints", "period": "Period"},
    )
    figure.update_traces(line=dict(width=2.5))
    figure.update_layout(**PLOTLY_LAYOUT, **PLOTLY_AXES)
    st.plotly_chart(figure, width="stretch")


def theme_count_chart(count_matrix: pd.DataFrame, title: str = "Complaints per topic over time") -> None:
    if count_matrix is None or count_matrix.empty:
        empty_state("No complaints to chart", "Charts appear once complaints are loaded.")
        return
    long = count_matrix.reset_index().melt(id_vars="theme", var_name="period", value_name="count")
    long["topic"] = long["theme"].map(theme_label)
    figure = px.bar(
        long,
        x="period",
        y="count",
        color="topic",
        barmode="group",
        title=title,
        labels={"count": "Complaints", "period": "Period"},
    )
    figure.update_layout(**PLOTLY_LAYOUT, **PLOTLY_AXES)
    st.plotly_chart(figure, width="stretch")


def sentiment_chart(sentiment: pd.DataFrame, title: str = "Feedback volume & average rating") -> None:
    if sentiment is None or sentiment.empty:
        empty_state("No feedback to chart", "Charts appear once feedback is loaded.")
        return
    frame = sentiment.reset_index()
    figure = go.Figure()
    for column, colour in (
        ("negative", SENTIMENT_COLORS["negative"]),
        ("neutral", SENTIMENT_COLORS["neutral"]),
        ("positive", SENTIMENT_COLORS["positive"]),
    ):
        figure.add_bar(x=frame["period"], y=frame[column], name=column, marker_color=colour)
    figure.add_scatter(
        x=frame["period"],
        y=frame["avg_rating"],
        name="avg rating",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=COLORS["ink"], width=3),
        marker=dict(size=8),
    )
    figure.update_layout(
        title=title,
        barmode="stack",
        yaxis=dict(title="Feedback items", gridcolor=COLORS["hairline-soft"], zeroline=False),
        yaxis2=dict(title="Avg rating", overlaying="y", side="right", range=[0, 5],
                    showgrid=False, zeroline=False),
        **PLOTLY_LAYOUT,
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor=COLORS["hairline"])
    st.plotly_chart(figure, width="stretch")


def before_after_chart(outcome: dict[str, Any]) -> None:
    """Grouped bar comparing the two measurement windows of a product change."""
    before, after = outcome["before"], outcome["after"]
    figure = go.Figure(
        data=[
            go.Bar(
                name="Before",
                x=["Complaints", "Share of all complaints (%)", "Avg rating"],
                y=[
                    before["complaints"],
                    before["share_of_complaints"],
                    before["avg_rating"],
                ],
                marker_color=SENTIMENT_COLORS["negative"],
            ),
            go.Bar(
                name="After",
                x=["Complaints", "Share of all complaints (%)", "Avg rating"],
                y=[after["complaints"], after["share_of_complaints"], after["avg_rating"]],
                marker_color=SENTIMENT_COLORS["positive"],
            ),
        ]
    )
    figure.update_layout(
        title=f"{outcome['change_name']}: {before['window_start']} → {before['window_end']} vs "
        f"{after['window_start']} → {after['window_end']}",
        barmode="group",
        **PLOTLY_LAYOUT,
        **PLOTLY_AXES,
    )
    st.plotly_chart(figure, width="stretch")


def feedback_table(frame: pd.DataFrame, *, limit: int = 300) -> None:
    if frame is None or frame.empty:
        empty_state(
            "No feedback matches these filters",
            "Widen the filters or clear the search box to see feedback records again.",
        )
        return
    view = frame.head(limit).copy()
    view["topic"] = view["theme"].map(theme_label)
    columns = [
        column
        for column in [
            "feedback_id",
            "date",
            "product_area",
            "topic",
            "sentiment",
            "rating",
            "channel",
            "feedback_text",
        ]
        if column in view.columns
    ]
    st.dataframe(view[columns], width="stretch", hide_index=True)


def setup_banner(problems: list[str]) -> None:
    if not problems:
        return
    st.warning(
        "**PulseMind is not fully configured yet**\n\n"
        + "\n".join(f"- {problem}" for problem in problems)
        + "\n\nCopy `.env.example` to `.env`, add your free Groq key from "
        "https://console.groq.com/keys, then reload this page."
    )


__all__ = [
    "CHART_COLORS",
    "COLORS",
    "KIND_ICONS",
    "PLOTLY_LAYOUT",
    "ROUNDED",
    "SHADOW",
    "apply_theme",
    "badge",
    "badge_row",
    "before_after_chart",
    "card",
    "empty_state",
    "evidence_list",
    "feedback_table",
    "key_values",
    "kind_label",
    "memory_card",
    "metric_row",
    "page_header",
    "retention_badge",
    "section",
    "sentiment_chart",
    "setup_banner",
    "source_badge",
    "theme_count_chart",
    "theme_label",
    "theme_share_chart",
]
