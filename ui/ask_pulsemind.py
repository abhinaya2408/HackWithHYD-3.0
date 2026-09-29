"""Ask PulseMind: questions answered from deterministic metrics plus long-term memory.

Each answer shows the memories that were recalled, the memory synthesis when it was used,
and the deterministic metrics the model was given — so a reviewer can check the reasoning
rather than trust it.
"""

from __future__ import annotations

import logging

import streamlit as st

from core.agent import Answer, PulseMindAgent
from ui import components as ui

logger = logging.getLogger("pulsemind.ui.ask")

# The static fallback list: these chips always render, even if generating smarter
# suggestions fails, so the page never shows an error where a control should be.
DEFAULT_SUGGESTED = [
    "What are customers complaining about now?",
    "Have we seen UPI issues before?",
    "What changed after Checkout V2?",
    "Did customer feedback improve after the release?",
    "What issue is emerging?",
    "What should we investigate next?",
    "What have we learned?",
]


def _suggestions(agent: PulseMindAgent) -> list[str]:
    """Suggested questions, biased toward what is actually moving in the data."""
    try:
        signals = agent.emerging_signals()
        dynamic = [
            f"What is driving {ui.theme_label(signal.theme).lower()} complaints?"
            for signal in signals[:2]
        ]
    except Exception:  # noqa: BLE001 - suggestions are a nicety, never a failure
        logger.debug("Falling back to the static suggestion list", exc_info=True)
        return list(DEFAULT_SUGGESTED)
    merged = dynamic + [question for question in DEFAULT_SUGGESTED if question not in dynamic]
    return merged[:8]


def _render_answer(answer: Answer) -> None:
    grounded = bool(answer.memories_used or answer.reflect_text)
    if grounded:
        ui.badge_row(
            [
                ui.badge("Grounded in memory", "success", icon_name="check"),
                ui.badge(f"{len(answer.memories_used)} memories recalled", "soft", icon_name="memory"),
            ]
        )
    else:
        ui.badge_row([ui.badge("No memory matched this question", "warning", icon_name="alert")])

    st.markdown(f'<div class="pm-answer">{answer.text}</div>', unsafe_allow_html=True)
    if answer.error:
        ui.log_failure("Answer", str(answer.error))
        ui.warning_state(
            "The model didn't finish this answer",
            "The answer above was assembled deterministically from the data. Please ask again "
            "in a moment for a reasoned answer.",
        )

    columns = st.columns(3)
    columns[0].caption(f"memories recalled: {len(answer.memories_used)}")
    columns[1].caption(f"memory facts used: {len(answer.basis)}")
    columns[2].caption(f"periods in data: {len(answer.metrics.get('periods', []))}")

    if answer.memories_used:
        with st.expander(f"Memories used to answer this ({len(answer.memories_used)})"):
            ui.evidence_list(answer.memories_used, empty_message="No memories recalled.")
    else:
        with st.expander("Memories used to answer this (0)"):
            st.info(
                "No memories were recalled for this question. Retain some feedback or a "
                "decision first — that is the difference between guessing and knowing."
            )

    if answer.reflect_text:
        with st.expander("Memory synthesis"):
            st.markdown(answer.reflect_text)
            if answer.basis:
                st.markdown("**Based on:**")
                for fact in answer.basis:
                    st.markdown(f"- {fact}")


def _ask(agent: PulseMindAgent, question: str, history: list[Answer]) -> None:
    """Run a question through the agent and append the answer to the transcript."""
    is_learning = question.lower().startswith("what have we learned")
    label = (
        "Reflecting over everything retained in memory…"
        if is_learning
        else "Recalling memory and reasoning…"
    )
    try:
        with st.spinner(label):
            answer = agent.learnings() if is_learning else agent.ask(question)
    except Exception as exc:  # noqa: BLE001 - a failed answer must not raise a traceback
        reference = ui.log_exception("Ask PulseMind", exc)
        ui.error_state(
            section="Ask PulseMind",
            title="We couldn't answer that question",
            body="The question was not lost. Please try again in a moment.",
            reference=reference,
            retry_key=f"retry_ask_{len(history)}",
        )
        return
    history.append(answer)


def _body(agent: PulseMindAgent) -> None:
    if not agent.llm.configured:
        ui.warning_state(
            "PulseMind needs a model key",
            "Add your key to the .env file, then reload this page to ask questions.",
        )
        return

    history: list[Answer] = st.session_state.setdefault("qa_history", [])

    # The indicator is driven by the shared, real check in `ui.memory_state` so this
    # page and the sidebar can never disagree about whether memory is reachable.
    state = ui.memory_state(agent)
    ui.status_indicator(state)
    if state[0] != "pm-dot--ok":
        ui.warning_state(
            "Long-term memory is not reachable",
            "Answers will use the feedback data only, without recalled history.",
        )

    with ui.card():
        ui.section(
            "Ask a question",
            "Questions are answered from the feedback data and the memories retained for it — "
            "not from general knowledge.",
        )
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Ask PulseMind",
                placeholder="e.g. Did customer feedback improve after the Checkout V2 release?",
            )
            submitted = st.form_submit_button("Ask PulseMind", type="primary")
        if submitted:
            if question.strip():
                _ask(agent, question.strip(), history)
            else:
                ui.warning_state(
                    "Type a question first",
                    "For example: what changed after Checkout V2?",
                )

        st.caption("Suggested questions")
        try:
            questions = _suggestions(agent)
        except Exception:  # noqa: BLE001 - last resort; chips must still render
            logger.debug("Suggestion list failed entirely", exc_info=True)
            questions = list(DEFAULT_SUGGESTED)
        if not questions:
            questions = list(DEFAULT_SUGGESTED)
        columns = st.columns(4)
        for index, suggested in enumerate(questions):
            if columns[index % 4].button(suggested, key=f"suggested_{index}"):
                _ask(agent, suggested, history)

    if history:
        if st.button("Clear conversation"):
            st.session_state["qa_history"] = []
            st.rerun()

    for item in reversed(history):
        with ui.card():
            st.markdown(
                f'<div class="pm-question-echo">{item.question}</div>',
                unsafe_allow_html=True,
            )
            _render_answer(item)


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Ask PulseMind",
        "Answers combine deterministic metrics with what long-term memory holds: earlier "
        "feedback, recorded product changes, measured outcomes and past decisions.",
        eyebrow="Ask",
    )
    with ui.guard("Ask PulseMind", retry_key="retry_ask"):
        _body(agent)
