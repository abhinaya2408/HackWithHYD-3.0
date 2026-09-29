"""PulseMind's design system — the single source of truth for every visual token.

``DESIGN-airbnb.md`` (project root) drives this file: the light theme is that system's
palette, type scale, spacing scale, radii, the single shadow tier and component rules,
transcribed one-to-one. The design file documents **no dark mode**, so the dark theme is
derived from the *same token names* — every component reads ``var(--pm-*)``, which means
switching themes re-skins the entire app (sidebar, inputs, dropdown menus, cards, charts
labels) without a single component-level branch.

If Cereal / Circular are unavailable the design file recommends Inter, so Inter is the
first family in the stack and the rest is the documented fallback chain.

Nothing behavioural lives here: no data access, no widget construction beyond the
``<style>`` injection.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

MODES: tuple[str, ...] = ("light", "dark", "system")
MODE_LABELS: dict[str, str] = {"light": "Light", "dark": "Dark", "system": "System"}
MODE_ICONS: dict[str, str] = {
    "light": ":material/light_mode:",
    "dark": ":material/dark_mode:",
    "system": ":material/brightness_auto:",
}


def normalise_mode(value: str | None) -> str:
    """Coerce anything to one of the three supported modes."""
    mode = str(value or "").strip().lower()
    return mode if mode in MODES else "light"


# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------
# Light: DESIGN-airbnb.md verbatim. `primary-soft`, `success*`, `warning*` and `error-soft`
# are derived tints — the design file documents their text colours but not their fills.
# `on-primary` stays #ffffff in both themes.

LIGHT: dict[str, str] = {
    # `primary` is the brand accent: dots, chart series, decoration, non-text fills.
    "primary": "#ff385c",
    "primary-active": "#e00b41",
    "primary-disabled": "#ffd1da",
    "primary-soft": "#fff3f5",
    # `cta` / `link` are the two tokens that carry *text*, so they use the design
    # file's deeper Rausch (#e00b41, also `button-primary-active`). White on #ff385c is
    # only 3.52:1 and fails WCAG AA for normal text; white on #e00b41 is 4.90:1 and the
    # hover tone below is 5.61:1. `primary` itself is unchanged everywhere it is used
    # as an accent, so the design system's "one voltage" rule still holds.
    "cta": "#e00b41",
    "cta-hover": "#c22c46",
    "link": "#e00b41",
    "ink": "#222222",
    "body": "#3f3f3f",
    "muted": "#6a6a6a",
    "muted-soft": "#929292",
    "canvas": "#ffffff",
    "surface-card": "#ffffff",
    "surface-soft": "#f7f7f7",
    "surface-strong": "#f2f2f2",
    "hairline": "#dddddd",
    "hairline-soft": "#ebebeb",
    "border-strong": "#c1c1c1",
    "on-primary": "#ffffff",
    "on-dark": "#ffffff",
    "error": "#c13515",
    "error-soft": "#fdecea",
    "success": "#0f7b3d",
    "success-soft": "#e9f6ee",
    "warning": "#8a5a00",
    "warning-soft": "#fdf4e3",
    "scrim": "rgba(0, 0, 0, 0.5)",
}

# Dark: same token names, values derived for a #121212 canvas. Every pair below clears
# WCAG AA (4.5:1) for body text against `canvas` / the matching `*-soft` fill.
DARK: dict[str, str] = {
    "primary": "#ff385c",
    "primary-active": "#ff5c78",
    "primary-disabled": "#5c2b34",
    "primary-soft": "#2a1319",
    # Same reasoning as light: #ff385c is 3.52:1 behind white text. In dark mode the
    # hover tone also stays on the darker side (5.61:1) instead of lightening, because
    # lightening the fill is what would break AA.
    "cta": "#e00b41",
    "cta-hover": "#c22c46",
    # As *text* on the dark canvas, #ff385c is 5.33:1, so links keep the brand voltage.
    "link": "#ff385c",
    "ink": "#f5f5f5",
    "body": "#d6d6d6",
    "muted": "#a8a8a8",
    "muted-soft": "#8f8f8f",
    "canvas": "#121212",
    "surface-card": "#1a1a1a",
    "surface-soft": "#1c1c1c",
    "surface-strong": "#262626",
    "hairline": "#333333",
    "hairline-soft": "#2a2a2a",
    "border-strong": "#4d4d4d",
    "on-primary": "#ffffff",
    "on-dark": "#ffffff",
    "error": "#ff9c8a",
    "error-soft": "#3a1a14",
    "success": "#5fd39a",
    "success-soft": "#10281d",
    "warning": "#e8b75d",
    "warning-soft": "#2e2410",
    "scrim": "rgba(0, 0, 0, 0.7)",
}

THEMES: dict[str, dict[str, str]] = {"light": LIGHT, "dark": DARK}

# Back-compat alias: components that need a concrete hex (Plotly, which cannot read CSS
# variables) go through `colors()` instead, but this keeps the light palette addressable.
COLORS: dict[str, str] = LIGHT

CHART_COLORS: list[str] = ["#ff385c", "#222222", "#6a6a6a", "#929292", "#e00b41", "#c1c1c1"]
CHART_COLORS_DARK: list[str] = ["#ff385c", "#f5f5f5", "#a8a8a8", "#8f8f8f", "#ff5c78", "#4d4d4d"]

SENTIMENT_COLORS: dict[str, str] = {"negative": "#c13515", "neutral": "#c1c1c1", "positive": "#0f7b3d"}
SENTIMENT_COLORS_DARK: dict[str, str] = {"negative": "#ff9c8a", "neutral": "#4d4d4d", "positive": "#5fd39a"}

# ---------------------------------------------------------------------------
# Shape, space, type
# ---------------------------------------------------------------------------

ROUNDED: dict[str, str] = {
    "none": "0px",
    "xs": "4px",
    "sm": "8px",
    "md": "14px",
    "lg": "20px",
    "xl": "32px",
    "full": "9999px",
}

SPACING: dict[str, str] = {
    "xxs": "2px",
    "xs": "4px",
    "sm": "8px",
    "md": "12px",
    "base": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
    "section": "64px",
}

# Cereal/Circular are licensed; Inter is the design file's documented open-source
# substitute and is named first so an installed Inter is always preferred.
FONT_STACK = (
    "'Inter', 'Airbnb Cereal VF', Circular, -apple-system, BlinkMacSystemFont, "
    "system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
)

# The one shadow tier in the whole system.
SHADOW = "rgba(0, 0, 0, 0.02) 0 0 0 1px, rgba(0, 0, 0, 0.04) 0 2px 6px 0, rgba(0, 0, 0, 0.1) 0 4px 8px 0"
SHADOW_DARK = "rgba(0, 0, 0, 0.4) 0 0 0 1px, rgba(0, 0, 0, 0.45) 0 2px 6px 0, rgba(0, 0, 0, 0.6) 0 4px 8px 0"

MOTION: dict[str, str] = {
    "fast": "140ms cubic-bezier(0.2, 0, 0.2, 1)",
    "normal": "220ms cubic-bezier(0.2, 0, 0.2, 1)",
}

MAX_CONTENT_WIDTH = "1240px"

# ---------------------------------------------------------------------------
# Icons (inline SVG, 1.6px stroke — Lucide/Material line style)
# ---------------------------------------------------------------------------

_STROKE = (
    'fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"'
)


ICON_PATHS: dict[str, str] = {
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    "feedback": '<path d="M21 12a8 8 0 0 1-8 8H8l-5 3 1.4-4.6A8 8 0 1 1 21 12Z"/><path d="M8.5 12h7"/>',
    "change": '<path d="M5 15c0-3.9 3.1-7 7-7h7"/><path d="m16 5 3 3-3 3"/><path d="M19 9a7 7 0 0 1-7 7H5"/><path d="m8 19-3-3 3-3"/>',
    "insight": '<path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.5.4.8 1 .9 1.7h5.4c.1-.7.4-1.3.9-1.7A6 6 0 0 0 12 3Z"/>',
    "memory": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>',
    "ask": '<path d="M12 21a9 9 0 1 0-7.9-4.7L3 21l4.9-1.1A9 9 0 0 0 12 21Z"/><path d="M9.6 9.6a2.4 2.4 0 1 1 3.3 2.2c-.6.3-.9.8-.9 1.4v.3"/><path d="M12 16.5h.01"/>',
    "settings": '<path d="M4 7h10"/><path d="M18 7h2"/><circle cx="16" cy="7" r="2"/><path d="M4 17h4"/><path d="M12 17h8"/><circle cx="10" cy="17" r="2"/>',
    "alert": '<path d="M10.3 4.3 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "inbox": '<path d="M3 12h5l1.5 2.5h5L16 12h5"/><path d="M5.4 5.6 3 12v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5l-2.4-6.4A2 2 0 0 0 16.7 4H7.3a2 2 0 0 0-1.9 1.6Z"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>',
    "sparkle": '<path d="M12 3.5 13.7 9l5.5 1.7-5.5 1.7L12 18l-1.7-5.6L4.8 10.7 10.3 9Z"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "user": '<circle cx="12" cy="8.5" r="3.5"/><path d="M4.5 20a7.5 7.5 0 0 1 15 0"/>',
    "dot": '<circle cx="12" cy="12" r="5"/>',
}


def svg(paths: str, size: int = 20, view_box: str = "0 0 24 24") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="{view_box}" {_STROKE} aria-hidden="true">{paths}</svg>'
    )


ICONS: dict[str, str] = {name: svg(paths) for name, paths in ICON_PATHS.items()}


# ---------------------------------------------------------------------------
# Active mode
# ---------------------------------------------------------------------------


def resolved_mode(mode: str | None = None) -> str:
    """Resolve a mode to a concrete ``"light"`` or ``"dark"``.

    ``"system"`` cannot be observed server-side, so it is resolved from Streamlit's own
    theme context when available and otherwise defaults to light. (The CSS for
    ``"system"`` additionally carries a ``prefers-color-scheme`` block, so the chrome
    itself still follows the operating system.)
    """
    mode = normalise_mode(mode)
    if mode != "system":
        return mode
    try:
        detected = str(st.context.theme.type or "").strip().lower()
    except Exception:  # noqa: BLE001 - context is unavailable outside a script run
        detected = ""
    return "dark" if detected == "dark" else "light"


def colors(mode: str | None = None) -> dict[str, str]:
    """Concrete hex tokens for the active theme (for Plotly, which cannot read CSS vars)."""
    return THEMES[resolved_mode(mode)]


def chart_colors(mode: str | None = None) -> list[str]:
    return CHART_COLORS_DARK if resolved_mode(mode) == "dark" else CHART_COLORS


def sentiment_colors(mode: str | None = None) -> dict[str, str]:
    return SENTIMENT_COLORS_DARK if resolved_mode(mode) == "dark" else SENTIMENT_COLORS


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def _token_lines(tokens: dict[str, str]) -> str:
    return "\n".join(f"  --pm-{name}: {value};" for name, value in tokens.items())


def _root_block(tokens: dict[str, str], scheme: str) -> str:
    shape = "\n".join(f"  --pm-radius-{name}: {value};" for name, value in ROUNDED.items())
    space = "\n".join(f"  --pm-space-{name}: {value};" for name, value in SPACING.items())
    motion = "\n".join(f"  --pm-motion-{name}: {value};" for name, value in MOTION.items())
    shadow = SHADOW_DARK if scheme == "dark" else SHADOW
    return (
        ":root {\n"
        f"  color-scheme: {scheme};\n"
        f"{_token_lines(tokens)}\n"
        f"{shape}\n{space}\n{motion}\n"
        f"  --pm-font: {FONT_STACK};\n"
        f"  --pm-shadow: {shadow};\n"
        f"  --pm-content-width: {MAX_CONTENT_WIDTH};\n"
        "}"
    )


def _variables(mode: str) -> str:
    """Emit the CSS custom properties for the requested mode.

    ``system`` emits the light tokens on ``:root`` plus a dark override inside
    ``@media (prefers-color-scheme: dark)`` so the browser preference wins without a
    server round-trip.
    """
    mode = normalise_mode(mode)
    if mode == "system":
        return (
            _root_block(LIGHT, "light")
            + "\n@media (prefers-color-scheme: dark) {\n"
            + _root_block(DARK, "dark").replace(":root {", ":root {", 1)
            + "\n}"
        )
    resolved = resolved_mode(mode)
    return _root_block(THEMES[resolved], resolved)


# Plain (non f-) string: every value is a var(), so no brace escaping is ever needed and
# both themes flow through the identical rule set.
_STYLES = """
/* ---------------------------------------------------------------- reset */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
    font-family: var(--pm-font);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--pm-canvas);
    color: var(--pm-ink);
}
[data-testid="stHeader"] { background: transparent; }

/* ------------------------------------------------------- hide chrome */
#MainMenu, [data-testid="stToolbar"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stHeaderActionElements"], footer, .stDeployButton,
[data-testid="stException"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}
[data-testid="stSidebarCollapseButton"] { color: var(--pm-muted); }

/* ---------------------------------------------------------- container */
.block-container, [data-testid="stAppViewBlockContainer"], .stMainBlockContainer {
    max-width: var(--pm-content-width);
    padding: 2.5rem 2rem 4.5rem 2rem;
}
[data-testid="stVerticalBlock"] { gap: var(--pm-space-base); }
hr, [data-testid="stDivider"] hr {
    border-color: var(--pm-hairline-soft); margin: var(--pm-space-lg) 0;
}

/* --------------------------------------------------------- typography */
h1, .pm-page-title {
    font-size: 28px !important; font-weight: 600 !important; line-height: 1.25 !important;
    letter-spacing: -0.4px !important; color: var(--pm-ink) !important;
    margin: 0 0 var(--pm-space-sm) 0 !important;
}
h2 { font-size: 22px !important; font-weight: 600 !important; line-height: 1.3 !important;
     letter-spacing: -0.3px !important; color: var(--pm-ink) !important; }
h3 { font-size: 20px !important; font-weight: 600 !important; letter-spacing: -0.2px !important;
     color: var(--pm-ink) !important; }
h4, h5, h6 { font-weight: 600 !important; color: var(--pm-ink) !important; }
p, li, .stMarkdown p, [data-testid="stMarkdownContainer"] p {
    color: var(--pm-body); font-size: 16px; line-height: 1.55;
}
strong, b { color: var(--pm-ink); font-weight: 600; }
a, a:visited { color: var(--pm-link); text-decoration: none; }
a:hover { text-decoration: underline; }
[data-testid="stCaptionContainer"], .stCaption, small,
[data-testid="stCaptionContainer"] p {
    color: var(--pm-muted) !important; font-size: 13px !important; line-height: 1.45 !important;
}
.pm-page-header {
    padding-bottom: var(--pm-space-base); border-bottom: 1px solid var(--pm-hairline-soft);
    margin-bottom: var(--pm-space-lg);
}
.pm-page-subtitle { color: var(--pm-muted); font-size: 16px; line-height: 1.55; margin: 0; max-width: 78ch; }
.pm-section-title { font-size: 20px; font-weight: 600; color: var(--pm-ink); margin: 4px 0 var(--pm-space-xs) 0; letter-spacing: -0.2px; }
.pm-section-sub { color: var(--pm-muted); font-size: 14px; margin: 0 0 var(--pm-space-base) 0; line-height: 1.5; }
.pm-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--pm-muted); margin-bottom: var(--pm-space-sm);
}

/* -------------------------------------------------------------- badges */
.pm-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 11px; border-radius: var(--pm-radius-full);
    font-size: 12px; font-weight: 500; line-height: 1.6;
    border: 1px solid transparent; white-space: nowrap;
}
.pm-badge--github { background: var(--pm-ink); color: var(--pm-canvas); }
.pm-badge--pm { background: var(--pm-surface-strong); color: var(--pm-ink); border-color: var(--pm-hairline); }
.pm-badge--primary { background: var(--pm-primary-soft); color: var(--pm-primary-active); border-color: color-mix(in srgb, var(--pm-primary) 28%, transparent); }
.pm-badge--success { background: var(--pm-success-soft); color: var(--pm-success); border-color: color-mix(in srgb, var(--pm-success) 30%, transparent); }
.pm-badge--warning { background: var(--pm-warning-soft); color: var(--pm-warning); border-color: color-mix(in srgb, var(--pm-warning) 30%, transparent); }
.pm-badge--error { background: var(--pm-error-soft); color: var(--pm-error); border-color: color-mix(in srgb, var(--pm-error) 30%, transparent); }
.pm-badge--muted { background: var(--pm-surface-soft); color: var(--pm-muted); border-color: var(--pm-hairline); }
.pm-badge--accent { background: var(--pm-cta); color: var(--pm-on-primary); }
.pm-badge--new {
    background: var(--pm-canvas); color: var(--pm-ink); border-color: var(--pm-ink);
    font-size: 10px; letter-spacing: 0.32px; text-transform: uppercase; padding: 2px 7px; font-weight: 600;
}
.pm-badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 var(--pm-space-md) 0; }

/* --------------------------------------------------------------- cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-md) !important;
    background: var(--pm-surface-card);
    overflow: hidden;
    transition: box-shadow var(--pm-motion-normal), border-color var(--pm-motion-fast);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: var(--pm-shadow); }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] { box-shadow: none; }

/* -------------------------------------------------------------- metric */
[data-testid="stMetric"] {
    background: var(--pm-surface-card);
    border: 1px solid var(--pm-hairline);
    border-radius: var(--pm-radius-md);
    padding: var(--pm-space-base) var(--pm-space-base);
}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
    color: var(--pm-muted) !important; font-size: 12px !important; font-weight: 500 !important;
}
[data-testid="stMetricValue"] { color: var(--pm-ink) !important; font-size: 26px !important;
    font-weight: 600 !important; letter-spacing: -0.4px; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* -------------------------------------------------------------- buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
    border-radius: var(--pm-radius-sm) !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    min-height: 44px;
    padding: 0 var(--pm-space-base) !important;
    transition: background var(--pm-motion-fast), border-color var(--pm-motion-fast),
                color var(--pm-motion-fast), box-shadow var(--pm-motion-fast);
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
/* Filled controls use `--pm-cta`, not `--pm-primary`: the deeper Rausch keeps the
   white label at 4.90:1 (hover 5.61:1) in both themes, where #ff385c would be 3.52:1. */
button[kind="primary"], [data-testid="stBaseButton-primary"], .stFormSubmitButton > button[kind="primary"] {
    background: var(--pm-cta) !important;
    color: var(--pm-on-primary) !important;
    border: 1px solid var(--pm-cta) !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: var(--pm-cta-hover) !important;
    border-color: var(--pm-cta-hover) !important;
    color: var(--pm-on-primary) !important;
}
button:disabled, .stButton > button:disabled,
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled {
    background: var(--pm-surface-strong) !important;
    border-color: var(--pm-hairline) !important;
    color: var(--pm-muted) !important;
    cursor: not-allowed;
    opacity: 1 !important;
}
button:focus-visible, .stButton > button:focus-visible,
[data-testid="stBaseButton-primary"]:focus-visible, [data-testid="stBaseButton-secondary"]:focus-visible {
    outline: 2px solid var(--pm-ink); outline-offset: 2px;
}
[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"],
[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
    font-size: 20px; margin-right: 6px; vertical-align: middle;
}

/* --------------------------------------------------------------- forms */
/* One input grammar everywhere: same surface, same hairline, same focus colour,
   same placeholder colour — per DESIGN-airbnb.md `text-input`. */
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stDateInput"] > div, [data-baseweb="input"], [data-baseweb="textarea"],
[data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-testid="stTextAreaRootElement"], [data-testid="stFileUploaderDropzone"] {
    background: var(--pm-canvas) !important;
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-sm) !important;
    color: var(--pm-ink) !important;
    transition: border-color var(--pm-motion-fast), box-shadow var(--pm-motion-fast);
}
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stDateInput"] > div:focus-within, [data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within, [data-baseweb="base-input"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--pm-ink) !important;
    box-shadow: inset 0 0 0 1px var(--pm-ink) !important;
}
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-testid="stNumberInputField"], [data-testid="stDateInputField"] {
    color: var(--pm-ink) !important;
    background: transparent !important;
    -webkit-text-fill-color: var(--pm-ink);
    caret-color: var(--pm-primary);
}
input::placeholder, textarea::placeholder,
[data-baseweb="input"] input::placeholder, [data-baseweb="textarea"] textarea::placeholder {
    color: var(--pm-muted) !important;
    -webkit-text-fill-color: var(--pm-muted);
    opacity: 1;
}
label, .stSelectbox label, .stTextInput label, .stMultiSelect label,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] {
    font-size: 14px !important; font-weight: 500 !important; color: var(--pm-body) !important;
}
/* Selectbox / multiselect closed value + dropdown menu (rendered in a body portal). */
[data-baseweb="select"] span, [data-baseweb="select"] div[aria-selected] {
    color: var(--pm-ink) !important;
}
[data-baseweb="select"] svg { fill: var(--pm-muted) !important; color: var(--pm-muted) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="menu"], ul[role="listbox"], div[role="listbox"] {
    background: var(--pm-canvas) !important;
    color: var(--pm-ink) !important;
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-sm) !important;
    box-shadow: var(--pm-shadow) !important;
}
li[role="option"], [role="option"] {
    background: var(--pm-canvas) !important;
    color: var(--pm-ink) !important;
}
li[role="option"]:hover, [role="option"]:hover, [role="option"][data-highlighted="true"] {
    background: var(--pm-surface-soft) !important;
    color: var(--pm-ink) !important;
}
li[role="option"][aria-selected="true"], [role="option"][aria-selected="true"] {
    background: var(--pm-surface-strong) !important;
    color: var(--pm-ink) !important;
    font-weight: 600;
}
[data-baseweb="tag"] {
    background: var(--pm-surface-strong) !important;
    color: var(--pm-ink) !important;
    border-radius: var(--pm-radius-full) !important;
}
[data-baseweb="tag"] span { color: var(--pm-ink) !important; }
[data-baseweb="tag"] svg { fill: var(--pm-muted) !important; }
/* Slider, toggle, checkbox, radio labels. */
[data-testid="stSlider"] [role="slider"] { border-color: var(--pm-primary) !important; }
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--pm-primary) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--pm-muted) !important; }
[data-testid="stCheckbox"] input, [data-testid="stToggle"] input,
[data-testid="stRadio"] input { accent-color: var(--pm-primary); }
[data-testid="stCheckbox"] label, [data-testid="stToggle"] label, [data-testid="stRadio"] label,
[data-baseweb="radio"] div, [data-baseweb="checkbox"] div { color: var(--pm-body) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--pm-surface-soft) !important; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {
    color: var(--pm-muted) !important;
}

/* ------------------------------------------------- tables & containers */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-md) !important;
    overflow: hidden;
}
[data-testid="stTable"] td, [data-testid="stTable"] th { color: var(--pm-ink) !important; }
[data-testid="stExpander"] {
    border: 1px solid var(--pm-hairline) !important;
    border-radius: var(--pm-radius-md) !important;
    background: var(--pm-surface-card);
    overflow: hidden;
}
[data-testid="stExpander"] details { border: none !important; background: transparent !important; }
[data-testid="stExpander"] summary {
    padding: 13px var(--pm-space-base) !important; font-weight: 500; color: var(--pm-ink) !important;
    background: transparent; transition: background var(--pm-motion-fast);
}
[data-testid="stExpander"] summary:hover { background: var(--pm-surface-soft); }
[data-testid="stExpander"] summary svg { fill: var(--pm-muted) !important; color: var(--pm-muted) !important; }
[data-testid="stExpander"] p { color: var(--pm-body); }
code, .stCode, [data-testid="stCode"] {
    color: var(--pm-ink) !important;
    background: var(--pm-surface-strong) !important;
    border: 1px solid var(--pm-hairline);
    border-radius: var(--pm-radius-xs);
    font-size: 12.5px;
}
pre, [data-testid="stCode"] pre, .stCode pre {
    background: var(--pm-surface-soft) !important;
    border: 1px solid var(--pm-hairline);
    border-radius: var(--pm-radius-sm) !important;
}
[data-testid="stCode"] code, pre code { background: transparent !important; border: none !important; color: var(--pm-ink) !important; }

/* ------------------------------------------------------- tabs & alerts */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--pm-hairline); }
.stTabs [data-baseweb="tab"] { color: var(--pm-muted) !important; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--pm-ink) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--pm-ink) !important; }
[data-testid="stAlert"], [data-baseweb="notification"], [data-testid="stAlertContainer"] {
    border-radius: var(--pm-radius-md) !important;
    border: 1px solid var(--pm-hairline);
    background: var(--pm-surface-soft);
    color: var(--pm-body) !important;
}
[data-testid="stAlert"] p { font-size: 15px !important; color: var(--pm-body) !important; }
[data-testid="stNotification"] { background: var(--pm-canvas); color: var(--pm-ink); }
[data-testid="stSpinner"] i, [data-testid="stSpinner"] svg { color: var(--pm-primary) !important; }
[data-testid="stProgress"] > div > div > div { background: var(--pm-primary) !important; }
[data-testid="stProgress"] p { color: var(--pm-muted) !important; }
[data-testid="stTooltipContent"] { background: var(--pm-ink) !important; color: var(--pm-canvas) !important; }

/* ------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {
    background: var(--pm-surface-soft);
    border-right: 1px solid var(--pm-hairline-soft);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    display: flex; flex-direction: column;
    min-height: calc(100vh - 4rem);
    padding: var(--pm-space-lg) var(--pm-space-base) var(--pm-space-lg) var(--pm-space-base);
    gap: 0;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: var(--pm-space-sm); }
.pm-brand { display: flex; align-items: center; gap: 10px; }
.pm-brand-mark {
    width: 36px; height: 36px; border-radius: var(--pm-radius-full);
    background: var(--pm-cta); color: var(--pm-on-primary);
    display: flex; align-items: center; justify-content: center; flex: none;
}
.pm-brand-name { font-size: 17px; font-weight: 600; color: var(--pm-ink); line-height: 1.2; letter-spacing: -0.2px; }
.pm-brand-tag { font-size: 12px; color: var(--pm-muted); line-height: 1.3; }
.pm-nav-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.7px; text-transform: uppercase;
    color: var(--pm-muted); margin: var(--pm-space-lg) 0 0 4px;
}
/* Pill nav: one rule set, active = filled accent pill, inactive = quiet surface. */
div[class*="st-key-pm_nav"] .stButton > button,
div[class*="st-key-pm_nav"] [data-testid^="stBaseButton"] {
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    border-radius: var(--pm-radius-full) !important;
    padding: 0 var(--pm-space-base) !important;
    min-height: 42px;
    font-size: 15px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--pm-muted) !important;
    box-shadow: none !important;
}
div[class*="st-key-pm_nav"] .stButton > button:hover,
div[class*="st-key-pm_nav"] [data-testid^="stBaseButton"]:hover {
    background: var(--pm-surface-strong) !important;
    color: var(--pm-ink) !important;
    border-color: transparent !important;
}
/* The `.stButton > button` form of each selector is load-bearing: it carries two more
   specificity points than the base rule above, which would otherwise win and leave the
   active pill looking inactive. */
div[class*="st-key-pm_nav"] .stButton > button[kind="primary"],
div[class*="st-key-pm_nav"] .stButton > button[data-testid="stBaseButton-primary"],
div[class*="st-key-pm_nav"] [data-testid="stBaseButton-primary"] {
    background: var(--pm-cta) !important;
    border-color: var(--pm-cta) !important;
    color: var(--pm-on-primary) !important;
    font-weight: 600 !important;
}
div[class*="st-key-pm_nav"] .stButton > button[kind="primary"]:hover,
div[class*="st-key-pm_nav"] .stButton > button[data-testid="stBaseButton-primary"]:hover,
div[class*="st-key-pm_nav"] [data-testid="stBaseButton-primary"]:hover {
    background: var(--pm-cta-hover) !important;
    border-color: var(--pm-cta-hover) !important;
    color: var(--pm-on-primary) !important;
}
div[class*="st-key-pm_nav"] .stButton > button:focus-visible,
div[class*="st-key-pm_nav"] [data-testid^="stBaseButton"]:focus-visible {
    outline: 2px solid var(--pm-ink); outline-offset: 2px;
}
div[class*="st-key-pm_nav"] button p {
    font-size: 15px !important; font-weight: inherit !important; color: inherit !important;
}
div[class*="st-key-pm_nav"] button [data-testid="stIconMaterial"] { font-size: 20px; margin-right: 10px; }
/* Appearance picker: outline pills, selected one filled with the accent. */
div[class*="st-key-pm_theme"] .stButton > button,
div[class*="st-key-pm_theme"] [data-testid^="stBaseButton"] {
    border-radius: var(--pm-radius-full) !important;
    background: var(--pm-canvas) !important;
    border: 1px solid var(--pm-hairline) !important;
    color: var(--pm-body) !important;
    font-weight: 500 !important;
    min-height: 44px;
}
div[class*="st-key-pm_theme"] .stButton > button:hover,
div[class*="st-key-pm_theme"] [data-testid^="stBaseButton"]:hover {
    border-color: var(--pm-ink) !important;
    color: var(--pm-ink) !important;
}
div[class*="st-key-pm_theme"] .stButton > button[kind="primary"],
div[class*="st-key-pm_theme"] .stButton > button[data-testid="stBaseButton-primary"],
div[class*="st-key-pm_theme"] [data-testid="stBaseButton-primary"] {
    background: var(--pm-cta) !important;
    border-color: var(--pm-cta) !important;
    color: var(--pm-on-primary) !important;
    font-weight: 600 !important;
}
div[class*="st-key-pm_theme"] .stButton > button[kind="primary"]:hover,
div[class*="st-key-pm_theme"] .stButton > button[data-testid="stBaseButton-primary"]:hover,
div[class*="st-key-pm_theme"] [data-testid="stBaseButton-primary"]:hover {
    background: var(--pm-cta-hover) !important;
    border-color: var(--pm-cta-hover) !important;
    color: var(--pm-on-primary) !important;
}
div[class*="st-key-pm_theme"] button [data-testid="stIconMaterial"] { font-size: 18px; margin-right: 8px; }
.st-key-pm_sidebar_bottom { margin-top: auto; padding-top: var(--pm-space-lg); }
.pm-conn {
    display: flex; align-items: center; gap: 8px;
    padding: var(--pm-space-sm) var(--pm-space-md);
    border: 1px solid var(--pm-hairline); border-radius: var(--pm-radius-sm);
    background: var(--pm-surface-card);
    font-size: 13px; color: var(--pm-body);
    margin-bottom: var(--pm-space-sm);
}
.pm-dot { width: 8px; height: 8px; border-radius: var(--pm-radius-full); flex: none; }
.pm-dot--ok { background: #16a34a; box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.16); }
.pm-dot--idle { background: var(--pm-muted-soft); box-shadow: 0 0 0 3px rgba(128, 128, 128, 0.16); }
.pm-conn-meta { margin-left: auto; color: var(--pm-muted); font-size: 12px; }
.pm-status-inline {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--pm-muted); margin-bottom: var(--pm-space-xs);
}
.pm-status-meta { color: var(--pm-muted); font-size: 12px; }
.pm-profile { display: flex; align-items: center; gap: 10px; margin-bottom: var(--pm-space-base); }
.pm-avatar {
    width: 56px; height: 56px; border-radius: var(--pm-radius-full);
    background: var(--pm-ink); color: var(--pm-canvas);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 600; flex: none; letter-spacing: 0.5px;
}
.pm-avatar--lg { width: 72px; height: 72px; font-size: 26px; }
.pm-profile-name { font-size: 17px; font-weight: 600; color: var(--pm-ink); line-height: 1.25; }
.pm-profile-meta { font-size: 13px; color: var(--pm-muted); }

/* ------------------------------------------------------- state blocks */
.pm-empty, .pm-error {
    display: flex; gap: var(--pm-space-md); align-items: flex-start;
    border-radius: var(--pm-radius-md);
    padding: var(--pm-space-base) var(--pm-space-base);
    margin: var(--pm-space-sm) 0;
}
.pm-empty { background: var(--pm-surface-soft); border: 1px dashed var(--pm-hairline); color: var(--pm-muted); }
.pm-error { background: var(--pm-error-soft); border: 1px solid color-mix(in srgb, var(--pm-error) 32%, transparent); }
.pm-warning { background: var(--pm-warning-soft); border: 1px solid color-mix(in srgb, var(--pm-warning) 32%, transparent); }
.pm-state-icon { flex: none; color: var(--pm-muted); margin-top: 1px; }
.pm-error .pm-state-icon { color: var(--pm-error); }
.pm-warning .pm-state-icon { color: var(--pm-warning); }
.pm-state-title { color: var(--pm-ink); font-weight: 600; font-size: 15px; margin-bottom: 2px; }
.pm-state-body { color: var(--pm-body); font-size: 14px; line-height: 1.5; }
.pm-state-ref { color: var(--pm-muted); font-size: 12px; margin-top: 6px; }
.pm-skeleton {
    border-radius: var(--pm-radius-md); background: var(--pm-surface-soft);
    border: 1px solid var(--pm-hairline-soft);
    padding: var(--pm-space-lg);
    color: var(--pm-muted); font-size: 14px;
}
.pm-skeleton-bar {
    height: 10px; border-radius: var(--pm-radius-full);
    background: linear-gradient(90deg, var(--pm-surface-strong), var(--pm-hairline-soft), var(--pm-surface-strong));
    background-size: 200% 100%;
    animation: pm-shimmer 1.4s ease-in-out infinite;
    margin-bottom: 10px;
}
@keyframes pm-shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

/* --------------------------------------------------------- app content */
.pm-kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: var(--pm-space-md) var(--pm-space-lg); margin: var(--pm-space-sm) 0 var(--pm-space-base) 0; }
.pm-kv-key { font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: var(--pm-muted); }
.pm-kv-val { font-size: 14px; color: var(--pm-ink); font-weight: 500; word-break: break-word; }
.pm-change-title { font-size: 19px; font-weight: 600; color: var(--pm-ink); letter-spacing: -0.2px; margin-bottom: 2px; }
.pm-answer { font-size: 16px; line-height: 1.6; color: var(--pm-body); }
.pm-question-echo { font-size: 17px; font-weight: 600; color: var(--pm-ink); letter-spacing: -0.2px; }
.pm-muted-text { color: var(--pm-muted); font-size: 13px; }

/* --------------------------------------------------------- responsive */
@media (max-width: 900px) {
    .block-container, [data-testid="stAppViewBlockContainer"] { padding: 1.5rem 1rem 3rem 1rem; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}
"""


def _sync_streamlit_base(mode: str) -> None:
    """Point Streamlit's own base theme at the active mode.

    Our tokens drive every surface we style, but a handful of built-ins (the dataframe
    grid, native scrollbars, base-web portals) read Streamlit's config instead. Setting
    the option keeps those in step. Failures are non-fatal by design: the CSS above is
    authoritative, so the app stays correct even when the option cannot be changed.
    """
    base = resolved_mode(mode)
    try:
        current = st._config.get_option("theme.base")  # noqa: SLF001 - no public setter
        if str(current or "").strip().lower() != base:
            st._config.set_option("theme.base", base)  # noqa: SLF001
    except Exception:  # noqa: BLE001 - theme sync must never break a page
        pass


def apply_theme(mode: str | None = None) -> str:
    """Inject the tokens and the component CSS for one theme.

    Called once per rerun by :mod:`app`, so the stylesheet always matches the mode the
    user has chosen. Returns the mode that was applied.
    """
    mode = normalise_mode(mode)
    st.markdown(f"<style>{_variables(mode)}{_STYLES}</style>", unsafe_allow_html=True)
    _sync_streamlit_base(mode)
    return mode


def icon(name: str, size: int = 20) -> str:
    """One inline line-icon as HTML (falls back to a neutral dot)."""
    return svg(ICON_PATHS.get(name, ICON_PATHS["dot"]), size)


__all__ = [
    "CHART_COLORS",
    "COLORS",
    "DARK",
    "FONT_STACK",
    "ICON_PATHS",
    "ICONS",
    "LIGHT",
    "MAX_CONTENT_WIDTH",
    "MODES",
    "MODE_ICONS",
    "MODE_LABELS",
    "MOTION",
    "ROUNDED",
    "SENTIMENT_COLORS",
    "SHADOW",
    "SPACING",
    "THEMES",
    "apply_theme",
    "chart_colors",
    "colors",
    "icon",
    "normalise_mode",
    "resolved_mode",
    "sentiment_colors",
    "svg",
]
