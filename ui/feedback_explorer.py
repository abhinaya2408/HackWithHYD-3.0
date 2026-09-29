"""Feedback Explorer: the raw evidence behind every PulseMind claim."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.agent import PulseMindAgent
from core.config import PRODUCT_AREAS, SENTIMENTS, THEMES
from core.groq_client import GroqUnavailable
from services import feedback_analyzer as analyzer
from ui import components as ui


def _filters(frame: pd.DataFrame, period_list: list[str]):
    with ui.card():
        ui.section("Filters & search", "Narrow the dataset down to the records you care about.")
        columns = st.columns(4)
        selected_periods = columns[0].multiselect("Period", period_list, default=period_list)
        selected_areas = columns[1].multiselect("Product area", sorted(frame["product_area"].unique()))
        selected_themes = columns[2].multiselect(
            "Topic", sorted(frame["theme"].unique()), format_func=ui.theme_label
        )
        selected_sentiment = columns[3].multiselect("Sentiment", list(SENTIMENTS))

        columns = st.columns([3, 1, 1])
        search = columns[0].text_input(
            "Search feedback text", placeholder="e.g. UPI, checkout, refund"
        )
        rating_range = columns[1].slider("Rating", 1, 5, (1, 5))
        sort_desc = columns[2].toggle("Newest first", value=True)

    view = frame.copy()
    if selected_periods:
        view = view[view["period"].isin(selected_periods)]
    if selected_areas:
        view = view[view["product_area"].isin(selected_areas)]
    if selected_themes:
        view = view[view["theme"].isin(selected_themes)]
    if selected_sentiment:
        view = view[view["sentiment"].isin(selected_sentiment)]
    if search.strip():
        view = view[view["feedback_text"].str.contains(search.strip(), case=False, na=False)]
    view = view[(view["rating"] >= rating_range[0]) & (view["rating"] <= rating_range[1])]
    return view.sort_values("date", ascending=not sort_desc)


def _analysis_card(agent: PulseMindAgent, frame: pd.DataFrame) -> None:
    with ui.card():
        ui.section("Semantic analysis", "Let the model label the feedback that has no topic yet.")
        unlabeled = analyzer.unlabeled_rows(frame)
        st.caption(
            f"{len(frame) - len(unlabeled)} of {len(frame)} items already carry a topic. "
            f"{len(unlabeled)} still need semantic classification."
        )
        if st.button("Extract topics", disabled=not agent.llm.configured):
            if unlabeled.empty:
                st.success("Nothing to classify — every item already has a topic.")
            else:
                try:
                    with st.spinner("Classifying the unlabelled feedback…"):
                        labelled, notes = analyzer.label_themes_with_llm(frame, agent.llm)
                except GroqUnavailable as exc:
                    ui.error_state(
                        exc,
                        section="Classify feedback",
                        title="We couldn't classify the feedback right now",
                        body="The model call did not complete. Please try again in a moment.",
                    )
                else:
                    new_topics = int((labelled["theme"] != frame["theme"]).sum())
                    st.success(f"The model assigned {new_topics} topic(s).")
                    for note in notes:
                        st.caption(note)
                    st.dataframe(
                        labelled.loc[
                            labelled["theme"] != frame["theme"],
                            ["feedback_id", "product_area", "theme", "sentiment", "feedback_text"],
                        ].head(20),
                        width="stretch",
                        hide_index=True,
                    )
                    st.info(
                        "These labels live in this session only — the dataset stays the source "
                        "of truth for the committed demo. Use the button on the right to write "
                        "the period into long-term memory."
                    )


def _retention_card(agent: PulseMindAgent, period_list: list[str]) -> None:
    with ui.card():
        ui.section("Retain a period as memory", "One document per period becomes long-term memory.")
        st.caption(
            "Retaining builds **one** memory document per period — volume, sentiment, every "
            "topic with its share, a per-area breakdown and representative verbatims — and "
            "stores it in long-term memory. One document per period is also what keeps the "
            "whole demo inside a free tier's limits."
        )
        target_period = st.selectbox("Period to retain", period_list, index=len(period_list) - 1)
        documents = agent.feedback_memory_documents(target_period)
        ui.badge_row(
            [
                ui.badge(f"{len(documents)} memory document(s)", "soft", icon_name="memory"),
                ui.badge(f"{len(documents)} write(s)", "muted"),
            ]
        )
        with st.expander("Preview the document that will be retained"):
            for document in documents:
                st.markdown(
                    f"**{document.metadata['period']}** — areas: "
                    f"{document.metadata['product_areas']} — tags `{document.tags}`"
                )
                st.code(document.content, language="text")
        if st.button(
            "Retain selected period",
            type="primary",
            disabled=not agent.memory.is_started,
        ):
            with st.spinner(f"Retaining {len(documents)} memories…"):
                report = agent.ingest_feedback(target_period)
            if report.failed:
                for error in report.errors():
                    ui.log_failure("Retain feedback period", str(error))
                ui.warning_state(
                    f"Retained {report.written} of {report.written + report.failed} memories",
                    "Some memories could not be written. Please try again for the remaining "
                    "items.",
                )
            else:
                st.success(
                    f"Retained {report.written} memories for {target_period}. Ask PulseMind can "
                    "use them from now on."
                )


def _body(agent: PulseMindAgent) -> None:
    frame = agent.frame
    if frame.empty:
        ui.empty_state(
            "No feedback yet",
            "Load the demo dataset, then reload this page to explore the records behind every "
            "PulseMind claim.",
        )
        return

    period_list = agent.periods()
    view = _filters(frame, period_list)

    st.write("")
    ui.metric_row(
        [
            ("Matching items", len(view), None),
            ("Complaints", int((view["sentiment"] == "negative").sum()), None),
            ("Average rating", f"{view['rating'].mean():.2f}/5" if len(view) else "—", None),
            ("Topics covered", int(view["theme"].nunique()), None),
        ]
    )

    with ui.card():
        ui.section("Feedback records", "Filtered records, newest first when selected.")
        ui.feedback_table(view)

    with st.expander("Read individual feedback", icon=":material/forum:"):
        if len(view) == 0:
            ui.empty_state(
                "No feedback matches these filters",
                "Widen the filters or clear the search box to see feedback again.",
            )
        for _, row in view.head(25).iterrows():
            ui.badge_row(
                [
                    ui.badge(str(row["date"])[:10], "muted"),
                    ui.badge(str(row["product_area"]), "pm"),
                    ui.badge(ui.theme_label(row["theme"]), "muted"),
                    ui.badge(
                        str(row["sentiment"]),
                        "error" if row["sentiment"] == "negative" else "muted",
                    ),
                    ui.badge(
                        f"rating {row['rating']}/5",
                        "error" if row["rating"] <= 2 else "muted",
                    ),
                ]
            )
            st.markdown(f"`{row['feedback_id']}` · {row.get('channel', '')}")
            st.markdown(f"**{row['feedback_text']}**")
            st.divider()

    ui.section(
        "Analyse & retain",
        "Classify unlabelled feedback, then write a period into long-term memory.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        _analysis_card(agent, frame)
    with right:
        _retention_card(agent, period_list)

    with st.expander("Topic distribution across the whole dataset"):
        counts = frame["theme"].value_counts()
        st.dataframe(
            pd.DataFrame(
                {
                    "Topic": [ui.theme_label(theme) for theme in counts.index],
                    "Items": counts.values,
                    "Share of all feedback": [
                        f"{value / len(frame) * 100:.1f}%" for value in counts.values
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Product areas in the dataset: {', '.join(PRODUCT_AREAS)}. "
            f"Topics: {', '.join(THEMES)}."
        )


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Feedback Explorer",
        "Every insight PulseMind produces traces back to these records. Filter them, read "
        "the verbatims, and retain a period as long-term memory.",
        eyebrow="Evidence",
    )
    with ui.guard("Feedback Explorer", retry_key="retry_feedback"):
        _body(agent)
