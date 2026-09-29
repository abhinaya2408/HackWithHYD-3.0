"""PulseMind — memory-driven AI Product Intelligence Agent.

Run with:

    streamlit run app.py

The app boots one PulseMindAgent per Streamlit process. That agent owns the Groq
client and the Hindsight memory connection (embedded by default), so the memory bank
is started once and reused across every page rerun.

Visual design follows the Airbnb DESIGN.md system (see ``DESIGN.md`` and
``ui/theme.py``): coral accent, white canvas, rounded cards, generous spacing and
modest-weight typography.
"""

from __future__ import annotations

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

PAGES = {
    "Executive Dashboard": dashboard.render,
    "Feedback Explorer": feedback_explorer.render,
    "Product Changes": product_changes.render,
    "Insights": insights.render,
    "Memory Explorer": memory_explorer.render,
    "Ask PulseMind": ask_pulsemind.render,
}


@st.cache_resource(show_spinner=False)
def get_agent() -> PulseMindAgent:
    """Create the single agent for this process and try to bring up memory."""
    agent = PulseMindAgent(get_settings())
    try:
        agent.reload_data()
        agent.data_error = None
    except Exception as exc:  # noqa: BLE001 - the UI explains how to generate the data
        agent.data_error = str(exc)

    if agent.llm.configured:
        try:
            agent.memory.start()
            agent.memory_error = None
        except Exception as exc:  # noqa: BLE001 - keep the app usable without memory
            agent.memory_error = str(exc)
    else:
        agent.memory_error = "GROQ_API_KEY is not set, so Hindsight cannot start."
    return agent


def _sidebar_brand() -> None:
    st.markdown(
        '<div class="pm-brand">'
        '<div class="pm-brand-mark">🧠</div>'
        "<div>"
        '<div class="pm-brand-name">PulseMind</div>'
        '<div class="pm-brand-tag">Memory-driven product intelligence</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="PulseMind",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui.apply_theme()

    settings = get_settings()
    problems = settings.missing_requirements()

    with st.sidebar:
        _sidebar_brand()
        st.markdown('<div class="pm-nav-label">Navigate</div>', unsafe_allow_html=True)
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

        st.divider()

        with st.spinner("Starting PulseMind (Hindsight + Groq)…"):
            agent = get_agent()

        st.markdown('<div class="pm-nav-label">System status</div>', unsafe_allow_html=True)
        if problems:
            st.error("Configuration incomplete")
            for problem in problems:
                st.caption(f"• {problem}")
        else:
            memory = agent.memory.status()
            if agent.memory.is_started:
                ui.badge_row([ui.badge(f"Hindsight {memory['mode']} · running", "success")])
            else:
                ui.badge_row([ui.badge("Hindsight not running", "warning")])
            st.caption(f"endpoint: {memory['endpoint'] or '—'}")
            st.caption(f"bank: `{memory['bank_id']}`")
            st.caption(f"memory LLM ({memory['llm_provider']}): `{memory['llm_model']}`")
            st.caption(f"reasoning LLM: `{memory.get('reasoning_model', agent.llm.model)}`")
            st.caption(
                "Separate models on purpose — Groq rate limits are per model, so memory "
                "extraction gets its own tokens-per-minute budget."
            )
            if getattr(agent, "memory_error", None):
                st.warning("Memory not available")
                st.caption(getattr(agent, "memory_error"))

        if getattr(agent, "data_error", None):
            st.error("Dataset missing")
            st.caption(getattr(agent, "data_error"))

        st.divider()
        st.markdown('<div class="pm-nav-label">This session</div>', unsafe_allow_html=True)
        st.caption(
            f"Groq calls: **{agent.llm.usage.calls}** "
            f"({agent.llm.usage.total_tokens:,} tokens)"
        )
        if agent.llm.usage.failures:
            st.caption(f"Groq failures: {agent.llm.usage.failures}")
        if agent.memory.is_started:
            try:
                overview = agent.memory_overview()
                st.caption(f"Memory units in bank: **{overview['total']}**")
            except Exception:  # noqa: BLE001 - sidebar must never break the app
                pass
        st.caption(
            "Free tier: Groq + embedded Hindsight. "
            "No paid APIs, no cloud infrastructure, no paid vector database."
        )

    if getattr(agent, "data_error", None):
        ui.page_header("PulseMind", "The feedback dataset could not be loaded.")
        st.error(
            "The feedback dataset could not be loaded.\n\n"
            "Generate the demo dataset with:\n\n"
            "```bash\npython data/prepare_dataset.py demo\n```\n\n"
            "or import the real Kaggle dataset with:\n\n"
            "```bash\npython data/prepare_dataset.py kaggle --raw data/raw\n```"
        )
        return

    PAGES[page](agent)


if __name__ == "__main__":
    main()
