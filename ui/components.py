"""Shared Streamlit components for every PulseMind page.

Three groups live here:

* **Navigation** — the pill nav (icons + labels, active/inactive/hover/focus) and the
  brand/sidebar chrome. It is the only place the sidebar structure is defined.
* **States** — ``guard``/``error_state``/``warning_state``/``empty_state``/``skeleton``.
  Every page loads its data inside a guard, so a failure renders an on-brand message with
  a Retry button and a short reference, never a Python traceback. The full error is
  logged server-side through :mod:`logging`.
* **Presentation** — headers, badges, cards, metric rows, KPI tables and the Plotly
  charts (which read the active theme's concrete colours, since Plotly cannot read CSS
  custom properties).

Nothing here invents data: every function takes real objects returned by the agent.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from html import escape
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import MEMORY_KIND_LABELS, THEME_LABELS
from core.hindsight_client import MemoryHit, MemoryRecord
from ui import theme as design
from ui.theme import (
    CHART_COLORS,
    COLORS,
    FONT_STACK,
    ROUNDED,
    SENTIMENT_COLORS,
    SHADOW,
    apply_theme,
    chart_colors,
    colors,
    icon,
    sentiment_colors,
    svg,
)

logger = logging.getLogger("pulsemind.ui")

# ---------------------------------------------------------------------------
# Icons (line style; the nav uses Streamlit's matching Material Symbols)
# ---------------------------------------------------------------------------

KIND_ICONS: dict[str, str] = {
    "feedback": ":material/forum:",
    "product_change": ":material/rocket_launch:",
    "outcome": ":material/trending_up:",
    "emerging_issue": ":material/warning:",
    "product_manager_decision": ":material/check_circle:",
    "unknown": ":material/dot:",
}

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGE_KEY = "pm_page"
DEFAULT_PAGE = "Executive Dashboard"
SETTINGS_PAGE = "Settings"

NAV_ITEMS: list[tuple[str, str]] = [
    ("Executive Dashboard", ":material/space_dashboard:"),
    ("Feedback Explorer", ":material/forum:"),
    ("Product Changes", ":material/rocket_launch:"),
    ("Insights", ":material/lightbulb:"),
    ("Memory Explorer", ":material/database:"),
    ("Ask PulseMind", ":material/chat_bubble:"),
]
SETTINGS_ITEM: tuple[str, str] = (SETTINGS_PAGE, ":material/settings:")


def current_page() -> str:
    """The page label currently selected in the sidebar."""
    page = str(st.session_state.get(PAGE_KEY) or DEFAULT_PAGE)
    return page


def _select_page(label: str) -> None:
    st.session_state[PAGE_KEY] = label


def nav_button(label: str, material_icon: str, *, active: bool, key: str) -> None:
    """One pill nav item.

    Active items are rendered as primary buttons (filled accent pill, bolder label);
    inactive ones as quiet secondaries. Both states, plus hover and focus-visible, are
    styled from the theme tokens in :mod:`ui.theme`.
    """
    st.button(
        label,
        key=key,
        icon=material_icon,
        type="primary" if active else "secondary",
        width="stretch",
        on_click=_select_page,
        args=(label,),
    )


def sidebar_nav(pages: Sequence[str]) -> None:
    """Render the main navigation. Items keep the order given in ``pages``."""
    icons = dict(NAV_ITEMS + [SETTINGS_ITEM])
    active_page = current_page()
    with st.container(key="pm_nav_container"):
        for label in pages:
            nav_button(
                label,
                icons.get(label, ":material/dot:"),
                active=label == active_page,
                key=f"pm_nav_item_{label.lower().replace(' ', '_')}",
            )


def sidebar_bottom(pages: Sequence[str]) -> None:
    """The Settings entry, pinned to the bottom of the sidebar."""
    active_page = current_page()
    # Container keys must stay distinct from the button keys below them: Streamlit
    # registers both in the same element-id namespace.
    with st.container(key="pm_nav_settings_container"):
        for label in pages:
            nav_button(
                label,
                SETTINGS_ITEM[1],
                active=label == active_page,
                key=f"pm_nav_item_{label.lower()}",
            )


def brand(mark: str = "PM", name: str = "PulseMind", tagline: str = "Product intelligence, with memory") -> None:
    """The wordmark block at the top of the sidebar."""
    st.markdown(
        '<div class="pm-brand">'
        f'<div class="pm-brand-mark"><span style="font-size:13px;font-weight:600">{escape(mark)}</span></div>'
        "<div>"
        f'<div class="pm-brand-name">{escape(name)}</div>'
        f'<div class="pm-brand-tag">{escape(tagline)}</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def nav_label(text: str) -> None:
    """A quiet uppercase label above a nav group."""
    st.markdown(f'<div class="pm-nav-label">{escape(text)}</div>', unsafe_allow_html=True)


_MEMORY_PROBE_TTL = 60.0
_MEMORY_PROBE_KEY = "pm_memory_probe"


def memory_state(agent: Any) -> tuple[str, str, str]:
    """What we actually know about memory, as ``(dot_class, label, meta)``.

    The label says "Connected" only after a real round-trip to the backend
    (``HindsightMemory.check_connection``). It never claims more than it checked:
    an unstarted client reports "Not connected", and a started client whose probe
    failed reports "Connection problem" rather than implying it is reachable. The
    probe is cached briefly so ordinary widget interactions do not re-probe it on
    every rerun.
    """
    try:
        started = bool(agent.memory.is_started)
    except Exception:  # noqa: BLE001 - status chrome must never break a page
        started = False
    if not started:
        return "pm-dot--idle", "Not connected", ""

    probe = getattr(agent.memory, "check_connection", None)
    if not callable(probe):
        # Better to admit we cannot tell than to imply a check we did not make.
        return "pm-dot--idle", "Status unavailable", ""

    now = time.time()
    cached = st.session_state.get(_MEMORY_PROBE_KEY)
    if isinstance(cached, tuple) and len(cached) == 3 and now - float(cached[0]) < _MEMORY_PROBE_TTL:
        _, reachable, total = cached
    else:
        try:
            reachable = bool(probe())
        except Exception:  # noqa: BLE001 - a probe must never break a page
            logger.debug("Memory probe raised", exc_info=True)
            reachable = False
        total = None
        if reachable:
            try:
                total = int(agent.memory_overview()["total"])
            except Exception:  # noqa: BLE001 - the count is a nicety, not the status
                logger.debug("Memory count unavailable", exc_info=True)
        st.session_state[_MEMORY_PROBE_KEY] = (now, reachable, total)
        _, reachable, total = st.session_state[_MEMORY_PROBE_KEY]

    if not reachable:
        return "pm-dot--idle", "Connection problem", ""
    return "pm-dot--ok", "Connected", f"{total:,} memories" if total is not None else ""


def connection_status(agent: Any) -> None:
    """A neutral connectivity indicator: a dot and one word, nothing technical."""
    dot, label, meta = memory_state(agent)
    st.markdown(
        f'<div class="pm-conn"><span class="pm-dot {dot}"></span>'
        f'<span>{escape(label)}</span>'
        + (f'<span class="pm-conn-meta">{escape(meta)}</span>' if meta else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def status_indicator(state: tuple[str, str, str]) -> None:
    """The same status, inline (used at the top of a page instead of the sidebar).

    Takes the tuple from :func:`memory_state` so both places always agree on the
    wording, and carries no technical detail — no endpoints, banks or model names.
    """
    dot, label, meta = state
    st.markdown(
        f'<div class="pm-status-inline"><span class="pm-dot {dot}"></span>'
        f'<span>{escape(label)}</span>'
        + (f'<span class="pm-status-meta">{escape(meta)}</span>' if meta else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Loading / empty / error states
# ---------------------------------------------------------------------------


def new_reference() -> str:
    """A short, non-sensitive id that ties a user-visible message to a log entry."""
    return uuid.uuid4().hex[:6]


def log_exception(section: str, exc: BaseException, reference: str | None = None) -> str:
    """Log a full traceback server-side and return the reference shown to the user.

    ``exc_info=exc`` is used rather than ``logger.exception`` because this is also called
    outside an ``except`` block (when a failure arrives as a value), where
    ``logger.exception`` would log ``NoneType: None`` instead of the stack.
    """
    reference = reference or new_reference()
    logger.error("[%s] %s raised %s", reference, section, type(exc).__name__, exc_info=exc)
    return reference


def log_failure(section: str, detail: str, reference: str | None = None) -> str:
    """Log a failure that arrives as a value (no live exception) and return the reference."""
    reference = reference or new_reference()
    logger.error("[%s] %s: %s", reference, section, detail)
    return reference


def _state_block(
    kind: str,
    title: str,
    body: str = "",
    *,
    reference: str | None = None,
    icon_name: str = "alert",
) -> None:
    css = {"empty": "pm-empty", "error": "pm-error", "warning": "pm-warning"}.get(kind, "pm-empty")
    html = [
        f'<div class="{css}">',
        f'<span class="pm-state-icon">{icon(icon_name, 20)}</span>',
        "<div>",
        f'<div class="pm-state-title">{escape(title)}</div>',
    ]
    if body:
        html.append(f'<div class="pm-state-body">{escape(body)}</div>')
    if reference:
        html.append(f'<div class="pm-state-ref">Reference: {escape(reference)}</div>')
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def empty_state(title: str, body: str = "") -> None:
    """A friendly empty state instead of a bare info box."""
    _state_block("empty", title, body, icon_name="inbox")


def warning_state(title: str, body: str = "") -> None:
    """An inline validation / partial-failure notice (amber, not red)."""
    _state_block("warning", title, body, icon_name="alert")


def error_state(
    exc: BaseException | None = None,
    *,
    section: str = "page",
    title: str = "We couldn't load this right now",
    body: str = "Something went wrong on our side. Please try again in a moment.",
    detail: str | None = None,
    reference: str | None = None,
    retry_key: str | None = None,
    retry_label: str = "Try again",
) -> str:
    """Render an on-brand failure with a Retry button; never a traceback.

    The exception itself is logged in full (with its stack) server-side, and only the
    short reference travels to the browser. Clicking Retry triggers the normal rerun, so
    the caller's loading code runs again.
    """
    if exc is not None:
        reference = log_exception(section, exc, reference)
    elif detail is not None:
        reference = log_failure(section, detail, reference)
    _state_block("error", title, body, reference=reference)
    if retry_key:
        st.button(retry_label, key=retry_key, icon=":material/refresh:")
    return reference or ""


def skeleton(lines: int = 3, label: str | None = None) -> None:
    """A lightweight skeleton placeholder for content that is still loading."""
    bars = "".join('<div class="pm-skeleton-bar"></div>' for _ in range(max(1, lines)))
    caption = f'<div class="pm-muted-text">{escape(label)}</div>' if label else ""
    st.markdown(f'<div class="pm-skeleton">{bars}{caption}</div>', unsafe_allow_html=True)


@contextmanager
def loading(label: str = "Loading…") -> Iterator[None]:
    """A themed spinner for a short blocking operation."""
    with st.spinner(label):
        yield


@contextmanager
def guard(section: str, *, retry_key: str | None = None, title: str | None = None) -> Iterator[None]:
    """Run a block of page code and degrade gracefully if it raises.

    ``st.rerun``/``st.stop`` are not swallowed: Streamlit raises those as
    ``BaseException`` subclasses specifically so user code cannot catch them.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - this is the whole point of the guard
        error_state(
            exc,
            section=section,
            title=title or "We couldn't load this right now",
            retry_key=retry_key or f"retry_{section.lower().replace(' ', '_')}",
        )


def retry_button(label: str = "Reload data", key: str = "pm_retry") -> bool:
    """A standalone retry control for banner-level failures."""
    return st.button(label, key=key, icon=":material/refresh:")


# ---------------------------------------------------------------------------
# Presentational primitives
# ---------------------------------------------------------------------------

_TONES = {
    "github": "pm-badge--github",
    "pm": "pm-badge--pm",
    "primary": "pm-badge--accent",
    "accent": "pm-badge--accent",
    "soft": "pm-badge--primary",
    "success": "pm-badge--success",
    "warning": "pm-badge--warning",
    "muted": "pm-badge--muted",
    "error": "pm-badge--error",
    "new": "pm-badge--new",
}


def badge(text: str, tone: str = "muted", *, icon_name: str | None = None, icon_size: int = 13) -> str:
    """A pill badge. ``tone`` is one of the design-system badge variants."""
    css = _TONES.get(tone, _TONES["muted"])
    glyph = f"{icon(icon_name, icon_size)} " if icon_name else ""
    return f'<span class="pm-badge {css}">{glyph}{escape(str(text))}</span>'


def badge_row(badges: Sequence[str]) -> None:
    """Render a wrapping row of badges (already-built HTML or plain text)."""
    items = [item if item.strip().startswith("<") else badge(item) for item in badges if item]
    if not items:
        return
    st.markdown(f'<div class="pm-badge-row">{"".join(items)}</div>', unsafe_allow_html=True)


def source_badge(change) -> str:
    """GitHub vs Product-Manager provenance badge for one product change."""
    source = str((change.get("source", "") if hasattr(change, "get") else "") or "")
    if source == "github":
        kind = str((change.get("github_kind", "") if hasattr(change, "get") else "") or "").lower()
        label = "GitHub release" if kind == "release" else "GitHub pull request"
        return badge(label, "github", icon_name="change")
    return badge("Recorded by Product Manager", "pm", icon_name="user")


def retention_badge(retained: bool, count: int = 0) -> str:
    """Memory retention status for a product change."""
    if retained:
        suffix = f" · {count} unit{'s' if count != 1 else ''}" if count else ""
        return badge(f"Retained in memory{suffix}", "success", icon_name="check")
    return badge("Not retained yet", "warning", icon_name="dot")


def page_header(title: str, subtitle: str | None = None, eyebrow: str = "PulseMind") -> None:
    """The one consistent page header used by every page."""
    parts = ['<div class="pm-page-header">']
    if eyebrow:
        parts.append(f'<div class="pm-eyebrow">{escape(eyebrow)}</div>')
    parts.append(f'<h1 class="pm-page-title">{escape(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="pm-page-subtitle">{escape(subtitle)}</p>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    """A section heading with an optional muted description."""
    html = [f'<div class="pm-section-title">{escape(title)}</div>']
    if subtitle:
        html.append(f'<p class="pm-section-sub">{escape(subtitle)}</p>')
    st.markdown("".join(html), unsafe_allow_html=True)


def key_values(pairs: Sequence[tuple[str, str]]) -> None:
    """A compact key/value grid for metadata blocks."""
    cells = []
    for key, value in pairs:
        if value in (None, "", "—"):
            continue
        cells.append(
            '<div><div class="pm-kv-key">'
            + escape(str(key))
            + '</div><div class="pm-kv-val">'
            + escape(str(value))
            + "</div></div>"
        )
    if not cells:
        return
    st.markdown(f'<div class="pm-kv">{"".join(cells)}</div>', unsafe_allow_html=True)


@contextmanager
def card() -> Iterator[None]:
    """A design-system card (Streamlit's bordered container, styled above)."""
    with st.container(border=True):
        yield


def avatar(name: str, email: str = "", *, large: bool = False) -> None:
    """The profile avatar with an initials fallback."""
    from ui.state import initials  # local import keeps the module dependency one-way

    size = " pm-avatar--lg" if large else ""
    st.markdown(
        f'<div class="pm-avatar{size}">{escape(initials(name, email))}</div>',
        unsafe_allow_html=True,
    )


def profile_line(name: str, email: str, role: str = "") -> None:
    """The compact profile summary used on the Settings header."""
    from ui.state import initials

    display = name or "Your profile"
    meta = " · ".join(part for part in (email, role) if part) or "Add your details"
    st.markdown(
        '<div class="pm-profile">'
        f'<div class="pm-avatar pm-avatar--lg">{escape(initials(name, email))}</div>'
        "<div>"
        f'<div class="pm-profile-name">{escape(display)}</div>'
        f'<div class="pm-profile-meta">{escape(meta)}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_row(metrics: list[tuple[str, Any, str | None]]) -> None:
    """A clean, evenly spaced metric card grid."""
    if not metrics:
        return
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


# ---------------------------------------------------------------------------
# Charts (Plotly needs concrete colours, so they follow the active theme)
# ---------------------------------------------------------------------------


def _layout(*, title: str | None = None) -> dict:
    palette = colors()
    layout: dict[str, Any] = dict(
        margin=dict(l=10, r=10, t=40 if title else 18, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, title_text=""),
        font=dict(family=FONT_STACK, size=12, color=palette["ink"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=chart_colors(),
        hoverlabel=dict(
            font=dict(family=FONT_STACK, size=12, color=palette["ink"]),
            bgcolor=palette["canvas"],
            bordercolor=palette["hairline"],
        ),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color=palette["ink"]))
    return layout


def _axes() -> dict:
    palette = colors()
    return dict(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=palette["hairline"],
            tickfont=dict(color=palette["muted"], size=11),
        ),
        yaxis=dict(
            gridcolor=palette["hairline-soft"],
            zeroline=False,
            tickfont=dict(color=palette["muted"], size=11),
        ),
    )


def _render_chart(figure: go.Figure) -> None:
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def theme_share_chart(share_matrix: pd.DataFrame, title: str = "Complaint share by topic over time") -> None:
    if share_matrix is None or share_matrix.empty:
        empty_state("No complaints to chart yet", "Charts appear once complaints are loaded.")
        return
    long = share_matrix.reset_index().melt(id_vars="theme", var_name="period", value_name="share")
    long["topic"] = long["theme"].map(theme_label)
    figure = px.line(
        long,
        x="period",
        y="share",
        color="topic",
        markers=True,
        labels={"share": "% of complaints", "period": "Period"},
    )
    figure.update_traces(line=dict(width=2.5))
    figure.update_layout(**_layout(title=title), **_axes())
    _render_chart(figure)


def theme_count_chart(count_matrix: pd.DataFrame, title: str = "Complaints per topic over time") -> None:
    if count_matrix is None or count_matrix.empty:
        empty_state("No complaints to chart yet", "Charts appear once complaints are loaded.")
        return
    long = count_matrix.reset_index().melt(id_vars="theme", var_name="period", value_name="count")
    long["topic"] = long["theme"].map(theme_label)
    figure = px.bar(
        long,
        x="period",
        y="count",
        color="topic",
        barmode="group",
        labels={"count": "Complaints", "period": "Period"},
    )
    figure.update_layout(**_layout(title=title), **_axes())
    _render_chart(figure)


def sentiment_chart(sentiment: pd.DataFrame, title: str = "Feedback volume & average rating") -> None:
    if sentiment is None or sentiment.empty:
        empty_state("No feedback to chart yet", "Charts appear once feedback is loaded.")
        return
    palette = colors()
    sentiment_palette = sentiment_colors()
    frame = sentiment.reset_index()
    figure = go.Figure()
    for column, colour in (
        ("negative", sentiment_palette["negative"]),
        ("neutral", sentiment_palette["neutral"]),
        ("positive", sentiment_palette["positive"]),
    ):
        figure.add_bar(x=frame["period"], y=frame[column], name=column, marker_color=colour)
    figure.add_scatter(
        x=frame["period"],
        y=frame["avg_rating"],
        name="avg rating",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=palette["ink"], width=3),
        marker=dict(size=8),
    )
    figure.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(size=15, color=palette["ink"])),
        yaxis=dict(title="Feedback items", gridcolor=palette["hairline-soft"], zeroline=False),
        yaxis2=dict(
            title="Avg rating",
            overlaying="y",
            side="right",
            range=[0, 5],
            showgrid=False,
            zeroline=False,
        ),
        **_layout(),
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor=palette["hairline"])
    _render_chart(figure)


def before_after_chart(outcome: dict[str, Any]) -> None:
    """Grouped bar comparing the two measurement windows of a product change."""
    before, after = outcome["before"], outcome["after"]
    sentiment_palette = sentiment_colors()
    figure = go.Figure(
        data=[
            go.Bar(
                name="Before",
                x=["Complaints", "Share of all complaints (%)", "Avg rating"],
                y=[before["complaints"], before["share_of_complaints"], before["avg_rating"]],
                marker_color=sentiment_palette["negative"],
            ),
            go.Bar(
                name="After",
                x=["Complaints", "Share of all complaints (%)", "Avg rating"],
                y=[after["complaints"], after["share_of_complaints"], after["avg_rating"]],
                marker_color=sentiment_palette["positive"],
            ),
        ]
    )
    figure.update_layout(
        **_layout(
            title=f"{outcome['change_name']}: {before['window_start']} → {before['window_end']} "
            f"vs {after['window_start']} → {after['window_end']}"
        ),
        barmode="group",
        **_axes(),
    )
    _render_chart(figure)


# ---------------------------------------------------------------------------
# Domain-specific blocks
# ---------------------------------------------------------------------------


def kind_label(kind: str) -> str:
    return MEMORY_KIND_LABELS.get(kind, kind.replace("_", " ").title())


def theme_label(theme: str) -> str:
    return THEME_LABELS.get(theme, theme)


def memory_card(item: MemoryHit | MemoryRecord, *, expanded: bool = False) -> None:
    """Render one real memory (a recalled hit or a listed memory unit)."""
    kind = getattr(item, "kind", "unknown")
    metadata = getattr(item, "metadata", None) or {}
    when = (
        getattr(item, "occurred_start", None)
        or metadata.get("memory_date")
        or metadata.get("decision_date")
        or "date not stored"
    )
    title = f"{kind_label(kind)} · {str(when)[:10]}"
    with st.expander(title, expanded=expanded, icon=KIND_ICONS.get(kind, KIND_ICONS["unknown"])):
        badges = [badge(kind_label(kind), "soft")]
        if getattr(item, "fact_type", None):
            badges.append(badge(str(item.fact_type), "muted"))
        state = getattr(item, "state", None)
        if state:
            badges.append(badge(f"state: {state}", "muted"))
        badge_row(badges)

        st.markdown(f'<div class="pm-answer">{item.text}</div>', unsafe_allow_html=True)

        tags = getattr(item, "tags", None) or []
        scores = getattr(item, "scores", None) or {}
        memory_id = getattr(item, "id", None)
        if tags or metadata or scores or memory_id:
            with st.expander("Technical details"):
                if memory_id:
                    st.caption(f"Memory id: `{memory_id}`")
                if tags:
                    st.caption("tags: " + ", ".join(f"`{tag}`" for tag in tags))
                if metadata:
                    st.caption("metadata: " + ", ".join(f"{key}={value}" for key, value in metadata.items()))
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
    """Non-technical setup notice (no model names, endpoints, banks or ports)."""
    if not problems:
        return
    warning_state(
        "PulseMind is not fully configured yet",
        "Add the missing values to your .env file and reload the page. "
        + " ".join(problems),
    )


__all__ = [
    "CHART_COLORS",
    "COLORS",
    "DEFAULT_PAGE",
    "KIND_ICONS",
    "NAV_ITEMS",
    "PAGE_KEY",
    "ROUNDED",
    "SETTINGS_PAGE",
    "SHADOW",
    "apply_theme",
    "avatar",
    "badge",
    "badge_row",
    "before_after_chart",
    "brand",
    "card",
    "connection_status",
    "current_page",
    "design",
    "empty_state",
    "error_state",
    "evidence_list",
    "feedback_table",
    "guard",
    "icon",
    "key_values",
    "kind_label",
    "loading",
    "log_exception",
    "log_failure",
    "memory_card",
    "memory_state",
    "metric_row",
    "nav_label",
    "nav_button",
    "new_reference",
    "page_header",
    "profile_line",
    "retention_badge",
    "retry_button",
    "section",
    "sentiment_chart",
    "setup_banner",
    "sidebar_bottom",
    "sidebar_nav",
    "skeleton",
    "source_badge",
    "status_indicator",
    "svg",
    "theme_count_chart",
    "theme_label",
    "theme_share_chart",
    "warning_state",
]
