"""PulseMind — memory-driven AI Product Intelligence Agent.

Run with:

    streamlit run app.py

The app boots one PulseMindAgent per Streamlit process. That agent owns the Groq client
and the Hindsight memory connection (embedded by default), so the memory bank is started
once and reused across every page rerun.

All visual decisions live in ``ui/theme.py`` (tokens + CSS), all shared UI in
``ui/components.py``, and the two user preferences in ``ui/state.py``. This module only
composes them: sidebar chrome, page routing and one top-level error net.
"""

from __future__ import annotations

import logging

import streamlit as st

from core.agent import PulseMindAgent
from core.config import get_settings
from ui import (
    ask_pulsemind,
    dashboard,
    feedback_explorer,
    insights,
    memory_explorer,
    product_changes,
)
from ui import components as ui
from ui import settings as settings_page
from ui import state as user_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pulsemind.app")

PAGES = {
    "Executive Dashboard": dashboard.render,
    "Feedback Explorer": feedback_explorer.render,
    "Product Changes": product_changes.render,
    "Insights": insights.render,
    "Memory Explorer": memory_explorer.render,
    "Ask PulseMind": ask_pulsemind.render,
}

# Settings lives at the bottom of the sidebar, outside the numbered product pages.
SETTINGS_PAGES = {
    "Settings": settings_page.render,
}

ALL_PAGES = {**PAGES, **SETTINGS_PAGES}


@st.cache_resource(show_spinner=False)
def get_agent() -> PulseMindAgent:
    """Create the single agent for this process and try to bring memory up.

    Failures are recorded on the agent and surfaced as plain-language notices by the
    pages that need them — never as a traceback, and never with endpoints, model names,
    bank ids or ports.
    """
    agent = PulseMindAgent(get_settings())
    if agent.llm.configured:
        try:
            agent.memory.start()
            agent.memory_error = None
        except Exception as exc:  # noqa: BLE001 - keep the app usable without memory
            agent.memory_error = type(exc).__name__
            logger.exception("Memory service could not be started")
    else:
        agent.memory_error = "not_configured"
    return agent


def _sidebar(agent: PulseMindAgent) -> None:
    """Brand, navigation and Settings — no status dumps, no configuration details."""
    with st.sidebar:
        ui.brand()
        ui.nav_label("Navigate")
        ui.sidebar_nav(list(PAGES.keys()))
        with st.container(key="pm_sidebar_bottom"):
            ui.connection_status(agent)
            ui.sidebar_bottom(list(SETTINGS_PAGES.keys()))


def _load_data(agent: PulseMindAgent) -> str | None:
    """Load the datasets, returning a log reference when they cannot be read."""
    try:
        agent.reload_data()
        agent.data_error = None
    except Exception as exc:  # noqa: BLE001 - the page renders a friendly failure
        reference = ui.log_exception("Load feedback data", exc)
        agent.data_error = reference
        return reference
    return None


def main() -> None:
    st.set_page_config(
        page_title="PulseMind",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui.apply_theme(user_state.theme_mode())

    if ui.PAGE_KEY not in st.session_state:
        st.session_state[ui.PAGE_KEY] = ui.DEFAULT_PAGE

    with st.spinner("Starting PulseMind…"):
        agent = get_agent()

    _sidebar(agent)

    page = ui.current_page()
    if page not in ALL_PAGES:
        page = ui.DEFAULT_PAGE

    if page in SETTINGS_PAGES:
        ALL_PAGES[page](agent)
        return

    reference = _load_data(agent)
    if reference:
        ui.page_header("PulseMind", "The feedback dataset could not be loaded.")
        ui.error_state(
            section="Load feedback data",
            title="We couldn't load the feedback data",
            body="The demo dataset is missing or unreadable. Regenerate it with "
            "`python data/prepare_dataset.py demo`, then try again.",
            reference=reference,
            retry_key="retry_dataset",
        )
        return

    try:
        ALL_PAGES[page](agent)
    except Exception as exc:  # noqa: BLE001 - the last net before Streamlit's own handler
        ui.page_header(page, "Something went wrong while rendering this page.")
        ui.error_state(exc, section=page, retry_key=f"retry_page_{page.lower().replace(' ', '_')}")


if __name__ == "__main__":
    main()
