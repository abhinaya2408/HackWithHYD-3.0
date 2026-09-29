"""PulseMind's design system — an original language built from two references.

* ``DESIGN-airbnb.md`` supplies the layout skeleton: white canvas, 8px-based spacing,
  rounded interactive elements, 14px cards, hairline borders, marketplace density.
* ``DESIGN-apple.md`` supplies the typographic voice and motion: a near-black ink on
  white, 600-weight display type with tight negative tracking, 17px body text, a
  single accent colour, flat chrome and restrained 120–300ms easing.

It is neither a clone: tokens are recombined into PulseMind's own system. Nothing
here changes behaviour — pages call the same helpers and render the same data.
"""

from __future__ import annotations

from contextlib import contextmanager
from html import escape
from typing import Iterator, Sequence

import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    # The single accent (Airbnb's "one voltage" rule, PulseMind's own coral).
    "primary": "#e0455e",
    "primary-active": "#c22c46",
    "primary-disabled": "#f5c2cb",
    "primary-soft": "#fdf0f2",
    # Ink & text (Apple's near-black instead of pure black).
    "ink": "#1d1d1f",
    "body": "#3a3a3c",
    "muted": "#6e6e73",
    "muted-soft": "#a1a1a6",
    # Surfaces (Apple's white + parchment pair).
    "canvas": "#ffffff",
    "parchment": "#f5f5f7",
    "surface-soft": "#f5f5f7",
    "surface-strong": "#ebebed",
    # Hairlines (Airbnb border grammar, Apple's softness).
    "hairline": "#e0e0e0",
    "hairline-soft": "#f0f0f0",
    "border-strong": "#c7c7cc",
    # Semantic — kept conventional and readable, never decorative.
    "success": "#1d7a45",
    "success-soft": "#e9f6ee",
    "warning": "#8a5a00",
    "warning-soft": "#fdf4e3",
    "error": "#b62020",
    "error-soft": "#fdecec",
}

ROUNDED: dict[str, str] = {
    "xs": "5px",
    "sm": "8px",
    "md": "14px",
    "lg": "18px",
    "xl": "28px",
    "full": "9999px",
}

# Apple's stack first (resolves to real SF Pro on Apple devices); Inter is the
# documented open-source substitute elsewhere.
FONT_STACK = (
    "system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', "
    "'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
)

# Apple allows exactly one shadow, on product imagery only. UI chrome stays flat;
# the single soft tier below is reserved for dropdown/popover-class surfaces.
SHADOW = "0 2px 12px rgba(0, 0, 0, 0.08)"

# Motion tokens (Apple): fast interactions, normal transitions, large entrances.
MOTION = {
    "fast": "150ms cubic-bezier(0.25, 0.1, 0.25, 1)",
    "normal": "240ms cubic-bezier(0.25, 0.1, 0.25, 1)",
    "slow": "420ms cubic-bezier(0.32, 0.72, 0, 1)",
}

# Optional accent colours for charts, derived from the system palette.
CHART_COLORS = ["#e0455e", "#1d1d1f", "#6e6e73", "#a1a1a6", "#c22c46", "#c7c7cc"]

# Semantic colours that must stay conventional for charts (negative/neutral/positive).
SENTIMENT_COLORS = {"negative": "#b62020", "neutral": "#c7c7cc", "positive": "#1d7a45"}


def _css_variables() -> str:
    lines = [f"  --pm-{name}: {value};" for name, value in COLORS.items()]
    lines += [f"  --pm-radius-{name}: {value};" for name, value in ROUNDED.items()]
    lines += [f"  --pm-motion-{name}: {value};" for name, value in MOTION.items()]
    lines.append(f"  --pm-font: {FONT_STACK};")
    lines.append(f"  --pm-shadow: {SHADOW};")
    return ":root {\n" + "\n".join(lines) + "\n}"


# Plain (non f-) string so CSS braces never need escaping; every value is a var().
_STYLES = """
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
    font-family: var(--pm-font);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

.stApp { background: var(--pm-canvas); color: var(--pm-ink); }
[data-testid="stAppViewContainer"] { background: var(--pm-canvas); }
[data-testid="stHeader"] { background: rgba(255, 255, 255, 0.85); backdrop-filter: saturate(180%) blur(20px); }

/* Generous page frame (Airbnb), capped near its ~1280px content width. */
.block-container, [data-testid="stAppViewBlockContainer"] {
    max-width: 1200px;
    padding: 3rem 2.5rem 5rem 2.5rem;
}
@media (max-width: 744px) {
    .block-container, [data-testid="stAppViewBlockContainer"] { padding: 1.5rem 1rem 3rem 1rem; }
}

/* ---- Typography (Apple): 600 display, tight tracking; 17px body; no 700 body ---- */
h1, .pm-page-title {
    font-size: 40px !important;
    font-weight: 600 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.6px !important;
    color: var(--pm-ink) !important;
    margin: 0 0 6px 0 !important;
}
h2 { font-size: 34px !important; font-weight: 600 !important; line-height: 1.2 !important; letter-spacing: -0.4px !important; color: var(--pm-ink) !important; }
h3 { font-size: 24px !important; font-weight: 600 !important; letter-spacing: -0.3px !important; color: var(--pm-ink) !important; }
h4, h5, h6 { font-weight: 600 !important; color: var(--pm-ink) !important; }
p, li, .stMarkdown p { color: var(--pm-body); font-size: 17px; line-height: 1.47; letter-spacing: -0.1px; }
strong, b { color: var(--pm-ink); font-weight: 600; }
a, a:visited { color: var(--pm-primary); text-decoration: none; transition: color var(--pm-motion-fast); }
a:hover { color: var(--pm-primary-active); text-decoration: underline; }
hr, [data-testid="stDivider"] hr { border-color: var(--pm-hairline-soft); margin: 2rem 0; }
code { border-radius: var(--pm-radius-xs); font-size: 13px; }
pre { border-radius: var(--pm-radius-sm) !important; }
[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--pm-muted) !important;
    font-size: 14px !important;
    line-height: 1.43 !important;
}

/* ---- PulseMind header block ---- */
.pm-eyebrow {
    font-size: 12px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--pm-primary); margin-bottom: 8px;
}
.pm-page-subtitle { color: var(--pm-muted); font-size: 17px; line-height: 1.47; margin: 0 0 10px 0; max-width: 860px; }
.pm-page-header { padding-bottom: 14px; border-bottom: 1px solid var(--pm-hairline-soft); margin-bottom: 28px; }
.pm-section-title { font-size: 24px; font-weight: 600; color: var(--pm-ink); margin: 10px 0 4px 0; letter-spacing: -0.3px; }
.pm-section-sub { color: var(--pm-muted); font-size: 15px; margin: 0 0 14px 0; line-height: 1.45; }

/* ---- Badges / pills ---- */
.pm-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 11px; border-radius: var(--pm-radius-full);
    font-size: 12px; font-weight: 500; line-height: 1.6;
    border: 1px solid transparent; white-space: nowrap;
}
.pm-badge--github { background: var(--pm-ink); color: #fff; }
.pm-badge--pm { background: var(--pm-surface-soft); color: var(--pm-ink); border-color: var(--pm-hairline); }
.pm-badge--primary { background: var(--pm-primary); color: #fff; }
.pm-badge--success { background: var(--pm-success-soft); color: var(--pm-success); border-color: #c4e6d1; }
.pm-badge--warning { background: var(--pm-warning-soft); color: var(--pm-warning); border-color: #f0dcb0; }
.pm-badge--muted { background: var(--pm-surface-strong); color: var(--pm-muted); }
.pm-badge--error { background: var(--pm-error-soft); color: var(--pm-error); border-color: #f4c7c7; }
.pm-badge--new {
    background: var(--pm-canvas); color: var(--pm-ink); border-color: var(--pm-ink);
    font-size: 9px; letter-spacing: 0.32px; text-transform: uppercase; padding: 2px 7px; font-weight: 600;
}
.pm-badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 12px 0; }

/* ---- Cards (Airbnb): flat surfaces first, one restrained shadow tier ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-md) !important;
    background: var(--pm-canvas);
    box-shadow: none;
    overflow: hidden;
    transition: border-color var(--pm-motion-fast), box-shadow var(--pm-motion-normal);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--pm-border-strong); }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: none; }
[data-testid="stExpander"] {
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-md) !important;
    background: var(--pm-canvas);
    overflow: hidden;
}
[data-testid="stExpander"] details { border: none !important; }
[data-testid="stExpander"] summary { padding: 13px 16px !important; font-weight: 500; color: var(--pm-ink); transition: background var(--pm-motion-fast); }
[data-testid="stExpander"] summary:hover { background: var(--pm-surface-soft); }

/* ---- Metrics: flat, typography-led (Apple restraint) ---- */
[data-testid="stMetric"] {
    background: var(--pm-canvas);
    border: 1px solid var(--pm-hairline-soft);
    border-radius: var(--pm-radius-md);
    padding: 16px 18px;
    box-shadow: none;
}
[data-testid="stMetricLabel"] p { color: var(--pm-muted) !important; font-size: 13px !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: var(--pm-ink) !important; font-size: 30px !important; font-weight: 600 !important; letter-spacing: -0.5px; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* ---- Buttons: pill primaries (Apple CTA grammar), quiet secondaries ---- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    border-radius: var(--pm-radius-full) !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    min-height: 44px;
    padding: 0.55rem 1.3rem !important;
    transition: transform var(--pm-motion-fast), background var(--pm-motion-fast),
                border-color var(--pm-motion-fast), color var(--pm-motion-fast);
}
.stButton > button, [data-testid="stBaseButton-secondary"] {
    background: var(--pm-canvas) !important;
    color: var(--pm-ink) !important;
    border: 1px solid var(--pm-hairline) !important;
}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {
    background: var(--pm-surface-soft) !important;
    border-color: var(--pm-ink) !important;
    color: var(--pm-ink) !important;
}
.stButton > button:active, [data-testid="stBaseButton-secondary"]:active,
[data-testid="stBaseButton-primary"]:active {
    transform: scale(0.97);
}
button[kind="primary"], [data-testid="stBaseButton-primary"], .stFormSubmitButton > button[kind="primary"] {
    background: var(--pm-primary) !important;
    color: #fff !important;
    border: 1px solid var(--pm-primary) !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: var(--pm-primary-active) !important;
    border-color: var(--pm-primary-active) !important;
    color: #fff !important;
}
button:disabled, button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled {
    background: var(--pm-primary-disabled) !important;
    border-color: var(--pm-primary-disabled) !important;
    color: #fff !important;
    cursor: not-allowed;
    transform: none;
}
.stButton > button:focus-visible, [data-testid="stBaseButton-primary"]:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible {
    outline: 2px solid var(--pm-primary); outline-offset: 2px;
}

/* ---- Inputs: 8px radius (Airbnb), ink focus ring (Apple: no glow) ---- */
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
    border-radius: var(--pm-radius-sm) !important;
    border-color: var(--pm-hairline) !important;
    background: var(--pm-canvas) !important;
    transition: border-color var(--pm-motion-fast), box-shadow var(--pm-motion-fast);
}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--pm-ink) !important;
    box-shadow: 0 0 0 1px var(--pm-ink) !important;
}
input, textarea { color: var(--pm-ink) !important; }
::placeholder { color: var(--pm-muted-soft) !important; }
label, .stSelectbox label, .stTextInput label {
    font-size: 14px !important; font-weight: 500 !important; color: var(--pm-body) !important;
}
[data-testid="stWidgetLabel"] p { font-size: 14px !important; font-weight: 500 !important; color: var(--pm-body) !important; }
[data-baseweb="tag"] { background: var(--pm-surface-strong) !important; border-radius: var(--pm-radius-full) !important; }

/* Slider, checkbox, toggle accents pick up the single accent colour. */
[data-baseweb="slider"] [role="slider"] { border-color: var(--pm-primary) !important; background: var(--pm-primary) !important; }
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--pm-primary) !important; }
[data-testid="stCheckbox"] input, [data-testid="stToggle"] input { accent-color: var(--pm-primary); }

/* ---- Tables ---- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-md) !important;
    overflow: hidden;
}

/* ---- Alerts: rounded, semantic, readable ---- */
[data-testid="stAlert"], [data-baseweb="notification"], [data-testid="stAlertContainer"] {
    border-radius: var(--pm-radius-md) !important;
}
[data-testid="stAlert"] p { font-size: 15px !important; }

/* ---- Sidebar: quiet navigation, active item tinted with the accent ---- */
[data-testid="stSidebar"] {
    background: var(--pm-parchment);
    border-right: 1px solid var(--pm-hairline-soft);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.75rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
.pm-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.pm-brand-mark {
    width: 38px; height: 38px; border-radius: var(--pm-radius-full);
    background: var(--pm-primary); color: #fff; display: flex; align-items: center;
    justify-content: center; font-size: 19px;
}
.pm-brand-name { font-size: 19px; font-weight: 600; color: var(--pm-ink); line-height: 1.1; letter-spacing: -0.3px; }
.pm-brand-tag { font-size: 12px; color: var(--pm-muted); }
.pm-nav-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.7px; text-transform: uppercase;
    color: var(--pm-muted-soft); margin: 16px 0 4px 2px;
}
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex; align-items: center;
    padding: 10px 14px !important;
    border-radius: var(--pm-radius-sm) !important;
    color: var(--pm-body);
    font-size: 15px; font-weight: 400;
    cursor: pointer;
    transition: background var(--pm-motion-fast), color var(--pm-motion-fast);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(0, 0, 0, 0.04); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: var(--pm-canvas);
    color: var(--pm-primary);
    font-weight: 600;
    box-shadow: inset 0 0 0 1px var(--pm-hairline);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] .pm-side-card {
    background: var(--pm-canvas);
    border: 1px solid var(--pm-hairline-soft);
    border-radius: var(--pm-radius-md);
    padding: 14px 16px;
}

/* ---- Progress + tabs ---- */
[data-testid="stProgress"] > div > div > div { background: var(--pm-primary) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--pm-hairline); }
.stTabs [aria-selected="true"] { color: var(--pm-primary) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--pm-primary) !important; }

/* ---- Custom content helpers ---- */
.pm-empty {
    background: var(--pm-surface-soft);
    border: 1px dashed var(--pm-hairline);
    border-radius: var(--pm-radius-md);
    padding: 22px 24px;
    color: var(--pm-muted);
}
.pm-empty-title { color: var(--pm-ink); font-weight: 600; font-size: 16px; margin-bottom: 3px; }
.pm-kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px 24px; margin: 6px 0 14px 0; }
.pm-kv-key { font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: var(--pm-muted-soft); }
.pm-kv-val { font-size: 14px; color: var(--pm-ink); font-weight: 500; word-break: break-word; }
.pm-change-title { font-size: 22px; font-weight: 600; color: var(--pm-ink); letter-spacing: -0.3px; margin-bottom: 2px; }
.pm-answer { font-size: 17px; line-height: 1.55; color: var(--pm-body); letter-spacing: -0.1px; }
.pm-question-echo { font-size: 20px; font-weight: 600; color: var(--pm-ink); letter-spacing: -0.2px; }

/* Respect reduced-motion preferences. */
@media (prefers-reduced-motion: reduce) {
    * { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}
"""


def apply_theme() -> None:
    """Inject the design system into the current Streamlit page.

    Called once per rerun (Streamlit re-executes the script top-to-bottom), so the
    styles are always present for whichever session is rendering.
    """
    st.markdown(
        f"<style>{_css_variables()}{_STYLES}</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Small presentational helpers (return HTML so pages stay declarative)
# ---------------------------------------------------------------------------

_TONES = {
    "github": "pm-badge--github",
    "pm": "pm-badge--pm",
    "primary": "pm-badge--primary",
    "success": "pm-badge--success",
    "warning": "pm-badge--warning",
    "muted": "pm-badge--muted",
    "error": "pm-badge--error",
    "new": "pm-badge--new",
}


def badge(text: str, tone: str = "muted") -> str:
    """A pill badge. ``tone`` is one of the design-system badge variants."""
    css = _TONES.get(tone, _TONES["muted"])
    return f'<span class="pm-badge {css}">{escape(str(text))}</span>'


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
        return badge(f"◆ {label}", "github")
    return badge("● Recorded by Product Manager", "pm")


def retention_badge(retained: bool, count: int = 0) -> str:
    """Hindsight retention status for a product change."""
    if retained:
        suffix = f" · {count} memory unit{'s' if count != 1 else ''}" if count else ""
        return badge(f"✓ Retained in Hindsight{suffix}", "success")
    return badge("○ Not yet retained in Hindsight", "warning")


def page_header(title: str, subtitle: str | None = None, eyebrow: str = "PulseMind") -> None:
    """The one consistent page header used by every page."""
    parts = [
        '<div class="pm-page-header">',
        f'<div class="pm-eyebrow">{escape(eyebrow)}</div>',
        f'<h1 class="pm-page-title">{escape(title)}</h1>',
    ]
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


def empty_state(title: str, body: str = "") -> None:
    """A friendly, styled empty state instead of a bare ``st.info``."""
    html = [
        '<div class="pm-empty">',
        f'<div class="pm-empty-title">{escape(title)}</div>',
    ]
    if body:
        html.append(f"<div>{escape(body)}</div>")
    html.append("</div>")
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


__all__ = [
    "CHART_COLORS",
    "COLORS",
    "FONT_STACK",
    "MOTION",
    "ROUNDED",
    "SENTIMENT_COLORS",
    "SHADOW",
    "apply_theme",
    "badge",
    "badge_row",
    "card",
    "empty_state",
    "key_values",
    "page_header",
    "retention_badge",
    "section",
    "source_badge",
]
