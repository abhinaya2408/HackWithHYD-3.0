"""Executive Dashboard: what is happening right now, and what did our changes do?

Everything on this page is computed deterministically in Python so it renders
instantly and the Product Manager can trust the numbers. Narrative reasoning (which
costs a model call) lives on the Insights and Ask PulseMind pages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.agent import PulseMindAgent
from core.config import MEMORY_KIND_LABELS
from services import feedback_analyzer as analyzer
from services import trend_analyzer as trends
from ui import components as ui


def _memory_summary(agent: PulseMindAgent) -> None:
    """What PulseMind remembers — a product signal, never system telemetry."""
    with ui.card():
        ui.section(
            "What PulseMind remembers",
            "Long-term memory is what makes PulseMind more than a dashboard: it holds "
            "earlier feedback, product changes, measured outcomes and product decisions.",
        )
        if not agent.memory.is_started:
            ui.empty_state(
                "Long-term memory is not available",
                "Open Memory Explorer to connect it. Every metric on this page still comes "
                "from the feedback dataset.",
            )
            return
        overview = agent.memory_overview()
        counts = overview.get("counts", {})
        badges = [
            ui.badge(
                f"{counts.get(kind, 0)} {MEMORY_KIND_LABELS.get(kind, kind).lower()}",
                "soft" if counts.get(kind) else "muted",
            )
            for kind in MEMORY_KIND_LABELS
        ]
        ui.badge_row(badges)
        st.caption(f"**{overview.get('total', 0):,} memories** retained in total.")


def _body(agent: PulseMindAgent) -> None:
    frame = agent.frame
    period_list = agent.periods()
    if not period_list:
        ui.empty_state(
            "No feedback yet",
            "Load the demo dataset, then reload this page — the dashboard builds itself from it.",
        )
        return

    latest = period_list[-1]
    ui.badge_row(
        [
            ui.badge(f"{len(frame)} feedback items", "pm", icon_name="feedback"),
            ui.badge(f"{len(period_list)} periods", "pm"),
            ui.badge(f"Latest period: {latest}", "primary"),
        ]
    )
    st.caption(
        f"The dataset spans {period_list[0]} → {latest}. Every number below is computed in "
        "Python from that dataset and the memories retained for it."
    )

    # ------------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------------
    summary = trends.period_summary(frame, latest)
    previous = trends.period_summary(frame, period_list[-2]) if len(period_list) > 1 else None
    signals = trends.emerging_signals(frame, latest)
    emerging = [signal for signal in signals if signal.direction == "emerging"]
    improving = [signal for signal in signals if signal.direction == "improving"]

    ui.metric_row(
        [
            (
                "Feedback items",
                summary.total,
                f"{summary.total - previous.total:+d} vs {previous.period}" if previous else None,
            ),
            ("Complaints", summary.negative, f"{summary.negative / summary.total * 100:.0f}% of items"),
            ("Average rating", f"{summary.avg_rating:.2f}/5", None),
            ("Topics tracked", int(frame["theme"].nunique()), None),
            ("Emerging issues", len(emerging), "flagged by the trend detector"),
        ]
    )

    st.write("")
    _memory_summary(agent)

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------
    ui.section("Feedback trends", "Complaint mix and sentiment over every period.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with ui.card():
            ui.sentiment_chart(trends.sentiment_trend(frame))
            ui.theme_share_chart(trends.complaint_share_matrix(frame))
    with right:
        with ui.card():
            ui.section("Major themes now", f"Complaint mix in {latest}.")
            mix = analyzer.theme_mix(trends.filter_period(frame, latest))
            if mix:
                st.dataframe(
                    [
                        {
                            "Topic": item["label"],
                            "Complaints": item["count"],
                            "Share": f"{item['share']}%",
                        }
                        for item in mix
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                ui.empty_state("No complaints in this period")

        with ui.card():
            ui.section("Moving themes", "Issues getting worse or better, detected deterministically.")
            if emerging:
                for signal in emerging:
                    st.error(
                        f"**{ui.theme_label(signal.theme)}** — {signal.latest_count} complaints, "
                        f"{signal.latest_share:.1f}% of all complaints "
                        f"({signal.share_delta_pts:+.1f} points vs earlier periods). "
                        f"Average rating {signal.latest_avg_rating:.2f}/5."
                    )
            else:
                st.caption("No topic crossed the emergence thresholds.")
            for signal in improving:
                st.success(
                    f"**{ui.theme_label(signal.theme)}** — down "
                    f"{abs(signal.share_delta_pts):.1f} points to {signal.latest_share:.1f}% "
                    f"of complaints."
                )

    # ------------------------------------------------------------------
    # Product impact — did our change improve the problem?
    # ------------------------------------------------------------------
    ui.section(
        "Product impact",
        "Did the change we shipped improve the problem it targeted? Measured over 28-day "
        "windows on either side of the release date.",
    )
    changes = agent.changes
    if changes.empty:
        ui.empty_state(
            "No product changes recorded yet",
            "Record one on the Product Changes page — PulseMind never invents releases "
            "or release dates.",
        )
    else:
        change = changes.sort_values("date").iloc[-1]
        area = str(change["product_area"])
        theme = agent.theme_for_area(area)
        comparison = trends.before_after(
            frame,
            change_date=change["date"],
            change_name=str(change["change_name"]),
            product_area=area,
            theme=theme,
            window_days=28,
        )
        with ui.card():
            ui.badge_row(
                [
                    ui.source_badge(change),
                    ui.badge(pd.Timestamp(change["date"]).date().isoformat(), "muted"),
                    ui.badge(area, "muted"),
                ]
            )
            st.markdown(f"### {change['change_name']}")
            st.markdown(
                f"Targeted *{change['problem_targeted']}* expecting "
                f"*{change['expected_outcome']}*."
            )
            ui.metric_row(
                [
                    (
                        f"Complaints before ({comparison.window_days}d)",
                        comparison.before["complaints"],
                        f"{comparison.before['share_of_complaints']}% of complaints",
                    ),
                    (
                        f"Complaints after ({comparison.window_days}d)",
                        comparison.after["complaints"],
                        f"{comparison.after['share_of_complaints']}% of complaints",
                    ),
                    ("Share change", f"{comparison.share_delta_pts:+.1f} pts", "negative = fewer complaints"),
                    (
                        "Rating move",
                        f"{comparison.before['avg_rating']:.2f} → {comparison.after['avg_rating']:.2f}",
                        "1 is worst, 5 is best — falling means worse",
                    ),
                    ("Verdict", comparison.verdict, "deterministic 28-day comparison"),
                ]
            )
            ui.before_after_chart(comparison.as_dict())
            st.caption(
                "This verdict is computed in Python from the two measurement windows. "
                "Open **Insights** to reason over it together with the memories retained "
                "from before the change."
            )

    with st.expander("Latest complaints (newest first)", icon=":material/forum:"):
        recent = trends.recent_complaints(frame, latest, limit=10)
        if not recent:
            ui.empty_state("No complaints in this period")
        for row in recent:
            ui.badge_row(
                [
                    ui.badge(str(row["date"]), "muted"),
                    ui.badge(str(row["product_area"]), "pm"),
                    ui.badge(ui.theme_label(row["theme"]), "muted"),
                    ui.badge(
                        f"rating {row['rating']}/5",
                        "error" if row["rating"] <= 2 else "muted",
                    ),
                ]
            )
            st.markdown(f"{row['text']}")
            st.divider()


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Executive Dashboard",
        "The current state of customer feedback, the themes driving it, and whether our "
        "shipped changes actually moved the numbers.",
        eyebrow="Overview",
    )
    with ui.guard("Executive Dashboard", retry_key="retry_dashboard"):
        _body(agent)
