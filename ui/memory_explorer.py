"""Memory Explorer: the real contents of the long-term memory bank.

The counters come from the memory store itself, the recall results come from a live
retrieval, and the reflect output comes from a live synthesis — all against the same bank
PulseMind writes to during normal operation. Nothing here is generated for display.
"""

from __future__ import annotations

import streamlit as st

from core.agent import PulseMindAgent
from core.config import MEMORY_KIND_LABELS, MEMORY_KINDS
from ui import components as ui

_KIND_ICONS = {
    "feedback": ":material/forum:",
    "product_change": ":material/rocket_launch:",
    "outcome": ":material/trending_up:",
    "emerging_issue": ":material/warning:",
    "product_manager_decision": ":material/check_circle:",
}


def _connect_card(agent: PulseMindAgent) -> None:
    with ui.card():
        ui.section("Long-term memory", "Connect it to keep what PulseMind learns across sessions.")
        st.markdown(
            "Long-term memory has not started yet. It starts automatically with the app once "
            "the environment is configured, and the rest of PulseMind works without it."
        )
        if st.button("Try to connect now", type="primary"):
            try:
                with st.spinner("Connecting long-term memory…"):
                    agent.memory.start()
            except Exception as exc:  # noqa: BLE001 - friendly failure, logged in full
                ui.error_state(
                    exc,
                    section="Start memory",
                    title="We couldn't connect long-term memory",
                    body="PulseMind continues to work without it. Please try again in a moment.",
                )
            else:
                st.success("Long-term memory is connected.")
                st.rerun()


def _overview_metrics(agent: PulseMindAgent) -> dict:
    overview = agent.memory_overview()
    st.caption(f"**{overview['total']:,} memories** are retained in total.")
    columns = st.columns(len(MEMORY_KIND_LABELS))
    for column, (kind, label) in zip(columns, MEMORY_KIND_LABELS.items()):
        column.metric(
            label,
            overview["counts"].get(kind, 0),
            icon=_KIND_ICONS.get(kind, ":material/dot:"),
        )
    return overview


def _demo_helpers(agent: PulseMindAgent, overview: dict) -> None:
    with st.expander("Demo helpers", expanded=overview["total"] == 0, icon=":material/build:"):
        emerging_now = [s for s in agent.emerging_signals() if s.direction == "emerging"]
        planned_writes = (
            len(agent.periods()) + len(agent.changes) + len(agent.changes) + min(1, len(emerging_now))
        )
        st.markdown(
            "**Bootstrap the demo scenario** writes the whole longitudinal story into memory: "
            "one feedback memory per period, each recorded product change, the measured "
            "outcome of each change, and the top emerging issue. "
            f"That is **{planned_writes} writes** in total, paced so they complete reliably."
        )
        columns = st.columns(3)

        if columns[0].button("Bootstrap demo memory", type="primary"):
            total_steps = planned_writes or 1
            progress = st.progress(0.0, text="Retaining feedback periods…")
            written = failed = 0
            step = 0

            for period in agent.periods():
                report = agent.ingest_feedback(period)
                written += report.written
                failed += report.failed
                step += 1
                progress.progress(step / total_steps, text=f"Retained feedback for {period}")

            for _, change in agent.changes.iterrows():
                receipt = agent.retain_product_change_memory(change)
                written += int(receipt.ok)
                failed += int(not receipt.ok)
                step += 1
                progress.progress(step / total_steps, text=f"Retained change {change['change_name']}")

            for _, change in agent.changes.iterrows():
                _, receipt = agent.measure_outcome(str(change["change_name"]))
                ok = bool(receipt and receipt.ok)
                written += int(ok)
                failed += int(not ok)
                step += 1
                progress.progress(step / total_steps, text=f"Measured outcome of {change['change_name']}")

            for signal in emerging_now[:1]:
                receipt = agent.retain_emerging_issue(signal.theme)
                written += int(receipt.ok)
                failed += int(not receipt.ok)
                step += 1
                progress.progress(step / total_steps, text=f"Retained emerging issue {signal.theme}")

            progress.progress(1.0, text="Done")
            st.session_state.pop("pm_memory_probe", None)
            if failed:
                ui.warning_state(
                    f"Retained {written} of {written + failed} memories",
                    "Some writes did not complete. Bootstrap again to fill in what is missing.",
                )
            else:
                st.success(f"Retained {written} memories. Ask PulseMind can use them now.")
            st.rerun()

        confirmed = columns[1].checkbox("I understand this erases everything remembered")
        if columns[1].button("Reset memory", disabled=not confirmed):
            with st.spinner("Erasing everything remembered…"):
                deleted = agent.reset_memory()
            st.session_state.pop("pm_memory_probe", None)
            if deleted:
                st.success("Memory reset.")
            else:
                ui.log_failure("Reset memory", str(agent.memory.last_error or "unknown"))
                ui.warning_state(
                    "Memory could not be reset",
                    "Nothing was removed. Please try again, or restart the app.",
                )
            st.rerun()

        columns[2].caption(
            "The bank persists between runs, which is why PulseMind keeps learning across "
            "sessions. Reset only when you want to replay the demo from scratch."
        )


def _retained(agent: PulseMindAgent, overview: dict) -> None:
    ui.section(
        "Retained memories",
        "Filter the bank by type, or search its contents directly.",
    )
    with ui.card():
        columns = st.columns([1, 1, 2, 1])
        kind_label = columns[0].selectbox(
            "Type",
            ["all", *MEMORY_KIND_LABELS.keys()],
            format_func=lambda key: "All memories" if key == "all" else MEMORY_KIND_LABELS[key],
        )
        fact_type = columns[1].selectbox("Category", ["all", "world", "experience", "observation"])
        search = columns[2].text_input("Search memories", placeholder="e.g. checkout, UPI, outcome")
        limit = columns[3].number_input("Show up to", min_value=10, max_value=500, value=100, step=10)

    with ui.loading("Loading memories…"):
        records = agent.list_memory_records(
            kind=None if kind_label == "all" else kind_label,
            fact_type=None if fact_type == "all" else fact_type,
            search_query=search.strip() or None,
            limit=int(limit),
        )
    ui.badge_row([ui.badge(f"{len(records)} memories returned", "muted")])
    ui.evidence_list(
        records,
        empty_message="No memories match. Use **Bootstrap demo memory** above to populate the bank.",
    )


def _recall_playground(agent: PulseMindAgent) -> None:
    with ui.card():
        ui.section(
            "Search memory",
            "Ask the memory store directly for the memories most relevant to a question.",
        )
        with st.form("recall_form"):
            query = st.text_input(
                "Question",
                value="What did customers complain about in Checkout before Checkout V2?",
            )
            columns = st.columns(3)
            filter_kind = columns[0].selectbox(
                "Restrict to type",
                ["all", *MEMORY_KINDS],
                format_func=lambda key: "All memories" if key == "all" else MEMORY_KIND_LABELS[key],
            )
            budget = columns[1].selectbox("Search depth", ["low", "mid", "high"], index=1)
            max_tokens = columns[2].number_input(
                "Detail (tokens)", 512, 16384, 4096, step=512,
                help="How much memory text to pull back before answering.",
            )
            submitted = st.form_submit_button("Search memory", type="primary")

        if submitted:
            try:
                with st.spinner("Searching memory…"):
                    hits = agent.recall(
                        query,
                        kinds=None if filter_kind == "all" else [filter_kind],
                        budget=budget,
                        max_tokens=int(max_tokens),
                    )
            except Exception as exc:  # noqa: BLE001
                ui.error_state(
                    exc,
                    section="Search memory",
                    title="We couldn't search memory right now",
                    body="Please try again in a moment.",
                )
            else:
                if not hits:
                    ui.empty_state(
                        "Nothing matched that question",
                        "Try different words, or retain more feedback first.",
                    )
                else:
                    st.success(f"Found {len(hits)} memories.")
                    ui.evidence_list(hits, empty_message="Nothing matched.")


def _reflect(agent: PulseMindAgent) -> None:
    with ui.card():
        ui.section(
            "Reason across everything",
            "Ask for a synthesis across the whole memory bank rather than a lookup of one fact. "
            "This is what PulseMind uses for open questions such as *what have we learned?*",
        )
        with st.form("reflect_form"):
            reflect_query = st.text_input(
                "Question",
                value="What have we learned about our product changes and their outcomes?",
            )
            reflect_budget = st.selectbox("Search depth", ["low", "mid", "high"], index=0)
            reflect_submitted = st.form_submit_button("Reason across memory", type="primary")

        if reflect_submitted:
            try:
                with st.spinner("Reasoning across the memory bank…"):
                    answer = agent.memory.reflect(
                        reflect_query, budget=reflect_budget, include_facts=True
                    )
            except Exception as exc:  # noqa: BLE001
                ui.error_state(
                    exc,
                    section="Reason across memory",
                    title="We couldn't reason across memory right now",
                    body="Please try again in a moment.",
                )
                return
            if not answer.text:
                ui.empty_state(
                    "Nothing to reason over yet",
                    "Retain some feedback or a product change first.",
                )
            else:
                st.markdown(f'<div class="pm-answer">{answer.text}</div>', unsafe_allow_html=True)
                if answer.based_on:
                    with st.expander(f"Memories this answer was based on ({len(answer.based_on)})"):
                        for fact in answer.based_on:
                            st.markdown(f"- {fact}")


def _body(agent: PulseMindAgent) -> None:
    if not agent.memory.is_started:
        _connect_card(agent)
        return

    overview = _overview_metrics(agent)
    _demo_helpers(agent, overview)
    st.divider()
    _retained(agent, overview)
    st.divider()
    _recall_playground(agent)
    _reflect(agent)


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Memory Explorer",
        "Everything PulseMind has retained and can recall: feedback, product changes, "
        "measured outcomes, emerging issues and product decisions.",
        eyebrow="Memory",
    )
    with ui.guard("Memory Explorer", retry_key="retry_memory"):
        _body(agent)
