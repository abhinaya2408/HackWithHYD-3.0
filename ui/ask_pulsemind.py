"""Ask PulseMind: questions answered from deterministic metrics plus Hindsight memory.

Each answer shows the memories that were recalled, the Hindsight `reflect` synthesis
when it was used, and the deterministic metrics the model was given — so a reviewer
can check the reasoning rather than trust it.
"""

from __future__ import annotations

import streamlit as st

from core.agent import Answer, PulseMindAgent
from ui import components as ui

SUGGESTED = [
    "What are customers complaining about now?",
    "Have we seen UPI issues before?",
    "What changed after Checkout V2?",
    "Did customer feedback improve after the release?",
    "What issue is emerging?",
    "What should we investigate next?",
    "What have we learned?",
]


def _render_answer(answer: Answer) -> None:
    grounded = bool(answer.memories_used or answer.reflect_text)
    if grounded:
        ui.badge_row(
            [
                ui.badge("✓ Grounded in Hindsight memory", "success"),
                ui.badge(f"{len(answer.memories_used)} memories recalled", "primary"),
            ]
        )
    else:
        ui.badge_row([ui.badge("○ No memory matched this question", "warning")])

    st.markdown(f'<div class="pm-answer">{answer.text}</div>', unsafe_allow_html=True)
    if answer.error:
        st.warning(f"Groq error: {answer.error}")

    columns = st.columns(4)
    columns[0].caption(f"memories recalled: {len(answer.memories_used)}")
    columns[1].caption(f"reflect facts: {len(answer.basis)}")
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
        with st.expander("Hindsight reflect output"):
            st.markdown(answer.reflect_text)
            if answer.basis:
                st.markdown("**Based on:**")
                for fact in answer.basis:
                    st.markdown(f"- {fact}")


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Ask PulseMind",
        "Answers combine deterministic metrics with what Hindsight remembers: earlier "
        "feedback, recorded product changes, measured outcomes and past decisions.",
        eyebrow="PulseMind",
    )

    if not agent.llm.configured:
        ui.setup_banner(["GROQ_API_KEY is not set"])
        return
    if not agent.memory.is_started:
        st.warning("Hindsight memory is not running — answers will not use long-term memory.")
    else:
        ui.badge_row([ui.badge("✓ Hindsight memory connected", "success")])

    history: list[Answer] = st.session_state.setdefault("qa_history", [])

    with ui.card():
        ui.section(
            "Ask a question",
            "Questions are answered from the feedback data and the memories retained in "
            "Hindsight — not from general knowledge.",
        )
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Ask PulseMind",
                placeholder="e.g. Did customer feedback improve after the Checkout V3 release?",
            )
            submitted = st.form_submit_button("Ask PulseMind", type="primary")
        if submitted and question.strip():
            _ask(agent, question.strip(), history)

        st.caption("Suggested questions")
        columns = st.columns(4)
        for index, suggested in enumerate(SUGGESTED):
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


def _ask(agent: PulseMindAgent, question: str, history: list[Answer]) -> None:
    """Run a question through the agent and append the answer to the transcript."""
    is_learning = question.lower().startswith("what have we learned")
    label = "Reflecting over everything retained in Hindsight…" if is_learning else "Recalling memory and reasoning…"
    with st.spinner(label):
        try:
            answer = agent.learnings() if is_learning else agent.ask(question)
        except Exception as exc:  # noqa: BLE001 - show the real failure to the user
            answer = Answer(question=question, text=f"The agent could not answer: {exc}", error=str(exc))
    history.append(answer)
