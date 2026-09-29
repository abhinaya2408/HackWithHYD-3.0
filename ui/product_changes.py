"""Product Changes: the Product Manager records what shipped, PulseMind remembers it.

The AI never invents a release or a release date. Changes arrive one of two ways:

* manually, from the form on this page; or
* automatically, when engineering merges a PR or publishes a release in the configured
  GitHub repository and the webhook receiver retains it in Hindsight.

Either way each change both lands in `data/product_changes.csv` and is RETAINED in
Hindsight, so later analysis can connect feedback to the change and measure whether it
worked. This page makes the source, the GitHub metadata and the retention status of
every change visually explicit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from core.agent import PulseMindAgent
from core.config import PRODUCT_AREAS
from ui import components as ui


def _display_name(path: str | Path) -> str:
    """Normalize a persisted-CSV path to its file name for a status message.

    `PulseMindAgent.record_product_change` returns the target path as a plain `str`
    (its documented contract, asserted in tests), so normalize here at the UI boundary
    instead of assuming a `Path` and crashing on `{path.name}`.
    """
    return Path(path).name


def _github_metadata(change) -> list[tuple[str, str]]:
    """The GitHub fields that make provenance obvious, in a stable order."""
    pairs: list[tuple[str, str]] = [("Source", "GitHub — detected automatically")]
    repo = str(change.get("github_repo", "") or "")
    number = str(change.get("github_number", "") or "")
    version = str(change.get("github_version", "") or "")
    kind = str(change.get("github_kind", "") or "")
    url = str(change.get("github_url", "") or "")
    if repo:
        pairs.append(("Repository", repo))
    if kind == "release":
        pairs.append(("Release version", version or "—"))
        if number:
            pairs.append(("Release id", number))
    else:
        pairs.append(("Pull request", f"#{number}" if number else "—"))
    pairs.append(("GitHub link", url or "not provided"))
    return pairs


def _render_change(agent: PulseMindAgent, change, retention: dict) -> None:
    change_date = pd.Timestamp(change["date"])
    area = str(change["product_area"])
    theme = agent.theme_for_area(area)
    from_github = str(change.get("source", "") or "") == "github"
    change_id = str(change.get("change_id", "") or "")
    status_key = change_id or str(change.get("github_url", "") or "") or str(change["change_name"])
    status = retention.get(status_key, {"retained": False, "units": 0})

    with ui.card():
        ui.badge_row(
            [
                ui.source_badge(change),
                ui.retention_badge(status["retained"], status["units"]),
                ui.badge(area, "muted"),
                ui.badge(f"{change_date.date().isoformat()}", "muted"),
            ]
        )
        st.markdown(
            f'<div class="pm-change-title">{change["change_name"]}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Change id: `{change['change_id']}`")

        if from_github:
            ui.key_values(_github_metadata(change))
            url = str(change.get("github_url", "") or "")
            if url:
                st.link_button("Open on GitHub ↗", url)
            st.caption(
                "PulseMind received a signed GitHub webhook for this event and retained "
                "the change in Hindsight as a product-change memory."
            )
        else:
            ui.key_values(
                [
                    ("Source", "Recorded by Product Manager on this page"),
                    ("Release date", change_date.date().isoformat()),
                    ("Product area", area),
                ]
            )

        st.markdown(f"**Problem targeted:** {change['problem_targeted']}")
        st.markdown(f"**Expected outcome:** {change['expected_outcome']}")

        columns = st.columns([1, 3])
        if columns[0].button("Measure outcome", key=f"measure_{change['change_id']}"):
            with st.spinner("Comparing the windows before and after the change…"):
                comparison, receipt = agent.measure_outcome(str(change["change_name"]))
            if comparison is None:
                st.error("Could not match this change to the feedback data.")
            else:
                st.session_state[f"outcome_{change['change_id']}"] = comparison.as_dict()
                st.session_state[f"outcome_receipt_{change['change_id']}"] = (
                    receipt.ok if receipt else False
                )

        stored = st.session_state.get(f"outcome_{change['change_id']}")
        if stored:
            ui.metric_row(
                [
                    ("Complaints before", stored["before"]["complaints"],
                     f"{stored['before']['share_of_complaints']}% of complaints"),
                    ("Complaints after", stored["after"]["complaints"],
                     f"{stored['after']['share_of_complaints']}% of complaints"),
                    ("Share change", f"{stored['share_delta_pts']:+.1f} pts", "primary measure"),
                    (
                        "Rating move",
                        f"{stored['before']['avg_rating']:.2f} → {stored['after']['avg_rating']:.2f}",
                        "1 is worst, 5 is best — falling means worse",
                    ),
                    ("Verdict", stored["verdict"], None),
                ]
            )
            ui.before_after_chart(stored)
            if st.session_state.get(f"outcome_receipt_{change['change_id']}"):
                st.success(
                    "Retained the measured outcome in Hindsight: future analysis will "
                    "recall what happened after this change."
                )
            st.caption(
                f"Topic used for the comparison: {ui.theme_label(theme) if theme else 'all topics in this area'}."
            )


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Product Changes",
        "Recorded changes are the bridge between feedback and outcomes. Each change is "
        "stored in data/product_changes.csv and retained in Hindsight with the problem it "
        "targeted and the outcome that was expected.",
        eyebrow="PulseMind",
    )

    changes = agent.changes
    github_count = (
        int((changes["source"].astype(str) == "github").sum()) if not changes.empty else 0
    )
    ui.badge_row(
        [
            ui.badge(f"{len(changes)} recorded changes", "primary"),
            ui.badge(f"{github_count} from GitHub", "github"),
            ui.badge(f"{len(changes) - github_count} from Product Manager", "pm"),
        ]
    )
    if not agent.memory.is_started:
        st.warning(
            "Hindsight is not running, so changes are saved to the CSV only and cannot be "
            "retained as memory. Open **Memory Explorer** to start it."
        )
    st.write("")

    # ------------------------------------------------------------------
    # Add a change
    # ------------------------------------------------------------------
    with ui.card():
        ui.section(
            "Record a product change",
            "Use this when engineering ships something PulseMind should connect to feedback.",
        )
        with st.form("add_change", clear_on_submit=False):
            columns = st.columns(2)
            name = columns[0].text_input("Change name", placeholder="Checkout V2")
            area = columns[1].selectbox("Product area", PRODUCT_AREAS)
            columns = st.columns(2)
            released = columns[0].date_input("Release date", value=date(2026, 2, 15))
            columns[1].markdown(
                "PulseMind compares the change date against real feedback dates, "
                "so pick the date the change actually shipped."
            )
            problem = st.text_area(
                "Problem targeted",
                placeholder="Checkout friction: multi-step flow, saved address reset, confusing coupon",
                height=80,
            )
            expected = st.text_area(
                "Expected outcome",
                placeholder="Reduce checkout complaints and cart abandonment",
                height=80,
            )
            submitted = st.form_submit_button("Save and retain in Hindsight", type="primary")

    if submitted:
        if not name.strip():
            st.error("A change name is required.")
        elif not problem.strip():
            st.error(
                "The problem targeted is required — without it PulseMind cannot connect "
                "the change to the feedback it was meant to fix."
            )
        else:
            change = {
                "change_id": "",
                "date": released.isoformat(),
                "change_name": name.strip(),
                "product_area": area,
                "problem_targeted": problem.strip(),
                "expected_outcome": expected.strip() or "not specified",
            }
            with st.spinner("Saving and retaining in Hindsight…"):
                path, receipt = agent.record_product_change(change)
            if receipt.ok:
                st.success(
                    f"Recorded **{change['change_name']}** in `{_display_name(path)}` and retained "
                    "it in Hindsight as a product-change memory."
                )
            else:
                st.warning(
                    f"Saved to `{_display_name(path)}`, but the Hindsight retain call failed: "
                    f"{receipt.error}"
                )
            st.rerun()

    st.divider()

    # ------------------------------------------------------------------
    # Existing changes
    # ------------------------------------------------------------------
    ui.section(
        f"Recorded changes ({len(changes)})",
        "Newest first. GitHub changes show repository, PR/release reference, version and link.",
    )
    if changes.empty:
        ui.empty_state(
            "No product changes recorded yet",
            "Record one above, or let a merged PR / published release arrive through the "
            "GitHub webhook receiver.",
        )
        return

    retention = agent.change_retention()
    for _, change in changes.sort_values("date", ascending=False).iterrows():
        _render_change(agent, change, retention)

    st.divider()
    with st.expander("Raw product_changes.csv"):
        st.dataframe(
            changes.assign(date=changes["date"].dt.strftime("%Y-%m-%d")),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Product changes also arrive automatically: when engineering merges a PR or "
        "publishes a release in the configured GitHub repository, the webhook receiver "
        "(`python -m core.github_webhook`) verifies the event and retains it here as a "
        "product-change memory."
    )
