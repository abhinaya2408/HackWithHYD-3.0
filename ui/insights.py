"""Insights: where memory visibly changes the agent's behaviour.

Three blocks:

1. **Without memory vs with memory** — the same period analysed twice, the second time
   with the recalled history attached, so the difference is observable.
2. **Emerging issue** — trend, sentiment change, recalled historical context, the related
   product change and the measured outcome for it, all in one place.
3. **Recommendation and decision** — what to investigate next, and the Product Manager's
   decision, which is retained as memory so later answers can learn from it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.agent import PulseMindAgent
from ui import components as ui

_FALLBACK_NOTE = (
    "The text above was computed deterministically in Python — it is a fallback, not a "
    "model analysis."
)


def _memory_analysis(agent: PulseMindAgent, period_list: list[str]) -> None:
    with ui.card():
        ui.section(
            "Without memory vs with memory",
            "The difference between the two columns is the block of memories recalled from "
            "long-term memory — earlier feedback, recorded product changes, measured outcomes "
            "and past decisions.",
        )
        columns = st.columns([2, 2, 1])
        period = columns[0].selectbox("Period to analyse", period_list, index=len(period_list) - 1)
        columns[1].markdown(
            "Both runs use Python-computed metrics for every number. Each run costs one model call."
        )
        run = columns[2].button("Run both", type="primary", disabled=not agent.llm.configured)

        if run:
            with st.spinner("Analysing without memory…"):
                without = agent.analyze_period(period, use_memory=False)
            with st.spinner("Analysing with recalled memory…"):
                with_memory = agent.analyze_period(period, use_memory=True)
            st.session_state["analysis_without"] = without
            st.session_state["analysis_with"] = with_memory

        without = st.session_state.get("analysis_without")
        with_memory = st.session_state.get("analysis_with")
        if without and with_memory and without.period == period and with_memory.period == period:
            left, right = st.columns(2, gap="large")
            with left:
                with st.container(border=True):
                    ui.badge_row([ui.badge("Without memory", "muted", icon_name="alert")])
                    st.markdown(f"#### {without.mode}")
                    st.write(without.narrative)
                    st.caption("Memories used: 0")
                    if without.error:
                        ui.log_failure("Analyse period without memory", str(without.error))
                        ui.warning_state("The model didn't answer", _FALLBACK_NOTE)
            with right:
                with st.container(border=True):
                    ui.badge_row([ui.badge("With memory", "soft", icon_name="memory")])
                    st.markdown(f"#### {with_memory.mode}")
                    st.write(with_memory.narrative)
                    st.caption(f"Memories recalled and used: {with_memory.memory_count}")
                    if with_memory.error:
                        ui.log_failure("Analyse period with memory", str(with_memory.error))
                        ui.warning_state("The model didn't answer", _FALLBACK_NOTE)

            if with_memory.memories_used:
                with st.expander(
                    f"Memories the second analysis actually recalled ({with_memory.memory_count})"
                ):
                    ui.evidence_list(
                        with_memory.memories_used,
                        empty_message="No memories were recalled.",
                    )
            with st.expander("How this analysis was produced"):
                st.caption(
                    "This is the context the model was given: the deterministic metrics first, "
                    "then the memories recalled from long-term memory."
                )
                st.code(with_memory.prompt_context, language="text")
        elif not agent.llm.configured:
            ui.warning_state(
                "Analysis needs a model key",
                "Add your key to the .env file, then reload this page.",
            )
        else:
            st.info("Press **Run both** to compare the two analyses for the selected period.")


def _emerging_issue(agent: PulseMindAgent, latest: str) -> None:
    ui.section(
        "Emerging issue",
        "Deterministic trend signals, with the history memory recalls for them.",
    )
    signals = agent.emerging_signals(latest)
    if not signals:
        ui.empty_state(
            "Nothing emerging in this period",
            "A theme needs a share change of at least 8 points, 4 complaints and 12% share "
            "to be flagged.",
        )
        return

    by_theme = {f"{ui.theme_label(signal.theme)} ({signal.direction})": signal for signal in signals}
    choice = st.selectbox("Detected signal", list(by_theme.keys()))
    signal = by_theme[choice]
    ui.metric_row(
        [
            ("Complaints", signal.latest_count, f"in {signal.latest_period}"),
            ("Share now", f"{signal.latest_share:.1f}%", "of all complaints"),
            ("Baseline share", f"{signal.baseline_share:.1f}%", "earlier periods"),
            ("Movement", f"{signal.share_delta_pts:+.1f} pts", "direction: " + signal.direction),
            ("Avg rating", f"{signal.latest_avg_rating:.2f}/5", None),
        ]
    )
    st.write("")

    columns = st.columns([1, 1, 2])
    build = columns[0].button("Build insight", type="primary", disabled=not agent.llm.configured)
    retain = columns[1].button("Retain as memory", disabled=not agent.memory.is_started)
    columns[2].caption(
        "Building an insight costs one model call (it reasons over recalled history). "
        "Retaining costs nothing extra — the signal is deterministic and its historical "
        "context comes straight from recall."
    )

    if build:
        try:
            with st.spinner("Recalling history and reasoning over it…"):
                insight = agent.build_insight(signal.theme, latest)
            st.session_state["insight"] = insight
            st.session_state["insight_theme"] = signal.theme
        except Exception as exc:  # noqa: BLE001 - the insight block degrades, the page does not
            ui.error_state(
                exc,
                section="Build insight",
                title="We couldn't build that insight right now",
                body="Please try again in a moment.",
            )

    if retain:
        with st.spinner("Retaining the emerging issue…"):
            receipt = agent.retain_emerging_issue(signal.theme, latest)
        if receipt.ok:
            st.success(
                "Retained this emerging issue with its evidence and the historical context "
                "recalled for it."
            )
        else:
            ui.log_failure("Retain emerging issue", str(receipt.error))
            ui.warning_state(
                "We couldn't retain that yet",
                "The emerging issue is unchanged. Please try again in a moment.",
            )

    insight = st.session_state.get("insight")
    if insight and st.session_state.get("insight_theme") == signal.theme:
        _render_insight(insight)


def _next_steps(agent: PulseMindAgent, latest: str) -> None:
    with ui.card():
        ui.section(
            "What should we investigate next?",
            "Ranked deterministically in Python: complaint-share movement, share, volume, "
            "rating severity, and whether this area already has a recorded change behind it.",
        )
        opportunities = agent.opportunities(latest)
        if opportunities:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Topic": item["label"],
                            "Area": item["product_area"],
                            "Complaints": item["complaints"],
                            "Share": f"{item['share']}%",
                            "Movement": f"{item['share_delta_pts']:+.1f} pts",
                            "Direction": item["direction"],
                            "Avg rating": item["avg_rating"],
                            "Priority score": item["score"],
                            "Change already recorded": "yes" if item["already_addressed_by_change"] else "no",
                        }
                        for item in opportunities
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            ui.empty_state("No complaints available to rank")

        if st.button("Generate recommendation", type="primary", disabled=not agent.llm.configured):
            try:
                with st.spinner("Ranking candidates and reasoning over recalled memory…"):
                    recommendation = agent.recommend_next(latest)
                st.session_state["recommendation"] = recommendation
            except Exception as exc:  # noqa: BLE001
                ui.error_state(
                    exc,
                    section="Generate recommendation",
                    title="We couldn't generate a recommendation right now",
                    body="Please try again in a moment.",
                )

    recommendation = st.session_state.get("recommendation")
    if recommendation:
        with ui.card():
            ui.badge_row(
                [
                    ui.badge(f"confidence: {recommendation.confidence}", "soft"),
                    ui.badge(
                        ui.theme_label(recommendation.theme) if recommendation.theme else "topic —",
                        "muted",
                    ),
                    ui.badge(recommendation.product_area or "area —", "muted"),
                    ui.badge(f"{len(recommendation.memories_used)} memories used", "muted"),
                ]
            )
            st.markdown(f"### {recommendation.title}")
            st.markdown(recommendation.action)
            st.markdown(f"**Why now:** {recommendation.why}")
            if recommendation.error:
                ui.log_failure("Recommendation", str(recommendation.error))
                ui.warning_state("The model didn't answer", "The recommendation above is a "
                                 "deterministic fallback, not a model analysis.")
            if recommendation.what_to_check:
                st.markdown("**What to check:**")
                for item in recommendation.what_to_check:
                    st.markdown(f"- {item}")
            if recommendation.supporting_evidence:
                st.markdown("**Evidence recalled from memory:**")
                for item in recommendation.supporting_evidence:
                    st.markdown(f"- {item}")

        with ui.card():
            with st.form("decision_form"):
                st.markdown("#### Record the Product Manager decision")
                decision = st.radio("Decision", ["accepted", "rejected", "deferred"], horizontal=True)
                reason = st.text_area("Reason", placeholder="Why did you decide this?", height=80)
                learning = st.text_area(
                    "Resulting learning",
                    placeholder="What should PulseMind carry forward from this decision?",
                    height=80,
                )
                saved = st.form_submit_button(
                    "Retain decision",
                    type="primary",
                    disabled=not agent.memory.is_started,
                )
            if saved:
                with st.spinner("Retaining the decision as long-term memory…"):
                    receipt = agent.record_decision(
                        recommendation,
                        decision=decision,
                        reason=reason,
                        learning=learning,
                    )
                if receipt.ok:
                    st.success(
                        "Decision retained. Ask PulseMind *“What have we learned?”* to see it used."
                    )
                    st.session_state.pop("recommendation", None)
                else:
                    ui.log_failure("Retain decision", str(receipt.error))
                    ui.warning_state(
                        "We couldn't retain that decision yet",
                        "Nothing was lost — please try again in a moment.",
                    )


def _body(agent: PulseMindAgent) -> None:
    period_list = agent.periods()
    if not period_list:
        ui.empty_state(
            "No feedback yet",
            "Load the demo dataset, then reload this page to see memory change the analysis.",
        )
        return

    latest = period_list[-1]
    _memory_analysis(agent, period_list)
    st.divider()
    _emerging_issue(agent, latest)
    st.divider()
    _next_steps(agent, latest)

    if not agent.memory.is_started:
        ui.warning_state(
            "Long-term memory is not available",
            "Insights will not use recalled history until it is connected. Open Memory "
            "Explorer to connect it.",
        )


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Insights",
        "Where long-term memory changes the answer. The same numbers are given to the model "
        "twice — once without memory, once with the history recalled for it.",
        eyebrow="Analysis",
    )
    with ui.guard("Insights", retry_key="retry_insights"):
        _body(agent)


def _render_insight(insight) -> None:
    """Render one Insight object."""
    with ui.card():
        ui.badge_row(
            [
                ui.badge(insight.product_area, "pm"),
                ui.badge(f"period {insight.period}", "muted"),
                ui.badge(f"direction: {insight.direction}", "muted"),
                ui.badge(f"{insight.memory_count} memories used", "soft", icon_name="memory"),
            ]
        )
        st.markdown(f"### {insight.label}")
        if insight.narrative:
            st.markdown(insight.narrative)
        if insight.trend_text:
            st.markdown(f"**Trend:** {insight.trend_text}")
        if insight.sentiment_change:
            st.markdown(f"**Sentiment change:** {insight.sentiment_change}")

        columns = st.columns(2, gap="large")
        with columns[0]:
            st.markdown("**Historical context recalled from memory**")
            if insight.historical_context:
                for item in insight.historical_context:
                    st.markdown(f"- {item}")
            else:
                st.info("No relevant history in memory for this topic.")
                st.caption("Retain a feedback period or a product change to build that history.")
        with columns[1]:
            st.markdown("**Related product change**")
            if insight.related_change:
                change = insight.related_change
                ui.badge_row([ui.source_badge(change)])
                st.markdown(
                    f"**{change['change_name']}** ({change['date']}, {change['product_area']})\n\n"
                    f"Targeted: {change['problem_targeted']}\n\n"
                    f"Expected: {change['expected_outcome']}"
                )
                if str(change.get("source", "") or "") == "github":
                    github_bits = [
                        bit
                        for bit in (
                            f"repo `{change.get('github_repo', '')}`" if change.get("github_repo") else "",
                            f"PR #{change.get('github_number', '')}" if change.get("github_number") else "",
                            f"version `{change.get('github_version', '')}`" if change.get("github_version") else "",
                        )
                        if bit
                    ]
                    st.caption("Detected automatically from GitHub (" + " · ".join(github_bits) + ")")
                    if change.get("github_url"):
                        st.link_button(
                            "Open on GitHub", str(change.get("github_url")), icon=":material/open_in_new:"
                        )
            else:
                st.info("No recorded product change for this product area yet.")

        if insight.outcome:
            st.markdown("**Measured outcome of that change**")
            ui.metric_row(
                [
                    ("Complaints before", insight.outcome["before"]["complaints"],
                     f"{insight.outcome['before']['share_of_complaints']}% of complaints"),
                    ("Complaints after", insight.outcome["after"]["complaints"],
                     f"{insight.outcome['after']['share_of_complaints']}% of complaints"),
                    ("Share change", f"{insight.outcome['share_delta_pts']:+.1f} pts", None),
                    (
                        "Rating move",
                        f"{insight.outcome['before']['avg_rating']:.2f} → "
                        f"{insight.outcome['after']['avg_rating']:.2f}",
                        "1 is worst, 5 is best — falling means worse",
                    ),
                    ("Verdict", insight.outcome["verdict"], "deterministic comparison"),
                ]
            )
            ui.before_after_chart(insight.outcome)

        counts = insight.metrics.get("counts_by_period") or {}
        if counts:
            st.markdown("**Topic volume across every period**")
            st.dataframe(
                pd.DataFrame({"Period": list(counts.keys()), "Mentions": list(counts.values())}),
                width="stretch",
                hide_index=True,
            )

        if insight.evidence_memories:
            with st.expander(f"Memories used as evidence ({insight.memory_count})"):
                ui.evidence_list(insight.evidence_memories, empty_message="No memories recalled.")
