"""Memory Explorer: the real contents of the Hindsight memory bank.

Nothing on this page is a simulation. The counters come from `list_memories`, the
recall results come from `recall`, and the reflect output comes from `reflect` — all
against the same bank PulseMind writes to during normal operation.
"""

from __future__ import annotations

import streamlit as st

from core.agent import PulseMindAgent
from core.config import MEMORY_KIND_LABELS, MEMORY_KINDS
from ui import components as ui


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Memory Explorer",
        "The real contents of the Hindsight bank: every memory unit PulseMind has "
        "retained, searchable and recallable. Nothing here is mocked or simulated.",
        eyebrow="PulseMind",
    )
    status = agent.memory.status()

    if not agent.memory.is_started:
        with ui.card():
            ui.badge_row([ui.badge("Hindsight: not running", "warning")])
            st.markdown(
                "Hindsight has not started yet. It starts automatically with the app once "
                "`GROQ_API_KEY` is set in `.env`."
            )
            st.json(status)
            if st.button("Try to start Hindsight now", type="primary"):
                with st.spinner("Starting embedded Hindsight (first run provisions PostgreSQL)…"):
                    try:
                        agent.memory.start()
                    except Exception as exc:  # noqa: BLE001 - show the real failure
                        st.error(f"Could not start Hindsight: {exc}")
                    else:
                        st.success("Hindsight is running.")
                        st.rerun()
        return

    # ------------------------------------------------------------------
    # Bank state
    # ------------------------------------------------------------------
    ui.badge_row(
        [
            ui.badge(f"mode: {status['mode']}", "primary"),
            ui.badge(f"bank: {status['bank_id']}", "pm"),
            ui.badge(f"memory LLM: {status['llm_model']}", "muted"),
        ]
    )
    columns = st.columns(4)
    columns[0].metric("Mode", status["mode"])
    columns[1].metric("Endpoint", status["endpoint"] or "—")
    columns[2].metric("Bank", status["bank_id"])
    columns[3].metric("Hindsight LLM", status["llm_model"])

    overview = agent.memory_overview()
    st.caption(
        f"The bank holds **{overview['total']}** memory units. "
        f"Counts below are read straight from `list_memories`."
    )
    count_columns = st.columns(len(MEMORY_KIND_LABELS))
    for column, (kind, label) in zip(count_columns, MEMORY_KIND_LABELS.items()):
        column.metric(f"{ui.KIND_ICONS.get(kind, '')} {label}", overview["counts"].get(kind, 0))

    st.divider()

    with st.expander("Demo helpers", expanded=overview["total"] == 0):
        emerging_now = [s for s in agent.emerging_signals() if s.direction == "emerging"]
        # One Hindsight write per period, per change, per outcome, plus the single
        # top emerging issue: this is the whole demo scenario's memory in one click.
        planned_writes = (
            len(agent.periods()) + len(agent.changes) + len(agent.changes) + min(1, len(emerging_now))
        )
        st.markdown(
            "**Bootstrap the demo scenario** writes the whole longitudinal story into "
            "Hindsight: one feedback memory per period, each recorded product change, the "
            "measured outcome of each change, and the top emerging issue. "
            f"That is **{planned_writes} RETAIN calls** — each one triggers one Hindsight "
            f"fact-extraction call on Groq, paced {int(agent.settings.retain_pace_seconds)}s "
            "apart so the free tier keeps up."
        )
        columns = st.columns(3)
        if columns[0].button("Bootstrap demo memory", type="primary"):
            total_steps = planned_writes
            progress = st.progress(0.0, text="Retaining feedback periods…")
            written = failed = 0
            step = 0

            for period in agent.periods():
                report = agent.ingest_feedback(period)
                written += report.written
                failed += report.failed
                step += 1
                progress.progress(step / total_steps, text=f"Retained feedback for {period}")

            # Retain recorded changes as memory only — this never rewrites the CSV.
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
            if failed:
                st.warning(f"Wrote {written} memories, {failed} failed (see the errors below).")
            st.success(
                f"Retained {written} memories. Open **Ask PulseMind** and ask "
                '"What have we learned?" to see them recalled.'
            )
            st.rerun()

        confirmed = columns[1].checkbox("I understand this deletes the bank")
        if columns[1].button("Reset memory bank", disabled=not confirmed):
            with st.spinner("Deleting and recreating the bank…"):
                deleted = agent.reset_memory()
            if deleted:
                st.success("Bank reset.")
            else:
                st.error(
                    "The bank could not be deleted"
                    + (f": {agent.memory.last_error}" if agent.memory.last_error else ".")
                    + " Nothing was removed — try again or restart the app."
                )
            st.rerun()

        columns[2].caption(
            "The bank persists between runs, which is why PulseMind keeps learning across "
            "sessions. Reset only when you want to replay the demo from scratch."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Retained memories
    # ------------------------------------------------------------------
    ui.section("Retained memories", "Filter the bank by kind, fact type or a server-side search query.")
    with ui.card():
        columns = st.columns([1, 1, 2, 1])
        kind_label = columns[0].selectbox(
            "Kind",
            ["all", *MEMORY_KIND_LABELS.keys()],
            format_func=lambda key: "all" if key == "all" else MEMORY_KIND_LABELS[key],
        )
        fact_type = columns[1].selectbox("Fact type", ["all", "world", "experience", "observation"])
        search = columns[2].text_input("Server-side search query", placeholder="e.g. checkout, UPI, outcome")
        limit = columns[3].number_input("Limit", min_value=10, max_value=500, value=100, step=10)

    records = agent.list_memory_records(
        kind=None if kind_label == "all" else kind_label,
        fact_type=None if fact_type == "all" else fact_type,
        search_query=search.strip() or None,
        limit=int(limit),
    )
    ui.badge_row([ui.badge(f"{len(records)} memory unit(s) returned", "muted")])
    ui.evidence_list(
        records,
        empty_message="No memories match. Use **Bootstrap demo memory** above to populate the bank.",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Recall playground
    # ------------------------------------------------------------------
    with ui.card():
        ui.section(
            "Recall playground",
            "Call `recall` directly against the bank. This is the same retrieval PulseMind "
            "uses when it answers a question, including the per-memory retrieval scores.",
        )
        with st.form("recall_form"):
            query = st.text_input(
                "Query",
                value="What did customers complain about in Checkout before Checkout V2?",
            )
            columns = st.columns(3)
            filter_kind = columns[0].selectbox(
                "Restrict to kind",
                ["all", *MEMORY_KINDS],
                format_func=lambda key: "all" if key == "all" else MEMORY_KIND_LABELS[key],
            )
            budget = columns[1].selectbox("Budget", ["low", "mid", "high"], index=1)
            max_tokens = columns[2].number_input("max_tokens", 512, 16384, 4096, step=512)
            submitted = st.form_submit_button("Recall", type="primary")

        if submitted:
            with st.spinner("Recalling from Hindsight…"):
                hits = agent.recall(
                    query,
                    kinds=None if filter_kind == "all" else [filter_kind],
                    budget=budget,
                    max_tokens=int(max_tokens),
                )
            if not hits:
                st.warning("Nothing recalled. Has the relevant history been retained yet?")
            else:
                st.success(f"Recalled {len(hits)} memory unit(s).")
                ui.evidence_list(hits, empty_message="Nothing recalled.")

    # ------------------------------------------------------------------
    # Reflect
    # ------------------------------------------------------------------
    with ui.card():
        ui.section(
            "Reflect",
            "`reflect` asks Hindsight to reason across the whole bank rather than look up a "
            "fact. This is what PulseMind uses for open questions such as *what have we learned?*",
        )
        with st.form("reflect_form"):
            reflect_query = st.text_input(
                "Reflect query",
                value="What have we learned about our product changes and their outcomes?",
            )
            reflect_budget = st.selectbox("Reflect budget", ["low", "mid", "high"], index=0)
            reflect_submitted = st.form_submit_button("Reflect", type="primary")

        if reflect_submitted:
            with st.spinner("Reflecting over the memory bank…"):
                answer = agent.memory.reflect(reflect_query, budget=reflect_budget, include_facts=True)
            if not answer.text:
                st.warning("Reflect returned nothing. Retain some memories first.")
            else:
                st.markdown(f'<div class="pm-answer">{answer.text}</div>', unsafe_allow_html=True)
                if answer.based_on:
                    with st.expander(f"Memories this answer was based on ({len(answer.based_on)})"):
                        for fact in answer.based_on:
                            st.markdown(f"- {fact}")
                if answer.usage:
                    st.caption(f"usage: {answer.usage}")
