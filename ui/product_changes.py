"""Product Changes: the Product Manager records what shipped, PulseMind remembers it.

The AI never invents a release or a release date. Changes arrive one of two ways:

* manually, from the form on this page; or
* automatically, when engineering merges a PR or publishes a release in the connected
  GitHub repository and the webhook receiver retains it in long-term memory.

Either way each change both lands in the product-changes dataset and is retained as a
memory, so later analysis can connect feedback to the change and measure whether it
worked. This page makes the source, the GitHub metadata and the retention status of
every change visible.
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
                st.link_button("Open on GitHub", url, icon=":material/open_in_new:")
            st.caption(
                "This change arrived from a signed GitHub event and was retained in "
                "long-term memory as a product change."
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
            comparison, receipt = agent.measure_outcome(str(change["change_name"]))
            if comparison is None:
                ui.warning_state(
                    "We couldn't match this change to the feedback data",
                    "Check the product area and release date, then measure again.",
                )
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
                    "Retained the measured outcome in long-term memory: future analysis will "
                    "recall what happened after this change."
                )
            st.caption(
                f"Topic used for the comparison: "
                f"{ui.theme_label(theme) if theme else 'all topics in this area'}."
            )


def _record_form(agent: PulseMindAgent) -> None:
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
            submitted = st.form_submit_button("Save and retain", type="primary")

    if not submitted:
        return

    if not name.strip():
        ui.warning_state(
            "Give the change a name",
            "A short name like “Checkout V2” is how PulseMind refers to this change later.",
        )
        return
    if not problem.strip():
        ui.warning_state(
            "Describe the problem this change targeted",
            "Without it PulseMind cannot connect the change to the feedback it was meant to fix.",
        )
        return

    change = {
        "change_id": "",
        "date": released.isoformat(),
        "change_name": name.strip(),
        "product_area": area,
        "problem_targeted": problem.strip(),
        "expected_outcome": expected.strip() or "not specified",
    }
    try:
        with st.spinner("Saving the change…"):
            path, receipt = agent.record_product_change(change)
    except Exception as exc:  # noqa: BLE001 - a failed save must not leak a traceback
        ui.error_state(
            exc,
            section="Record product change",
            title="We couldn't save that change",
            body="Nothing was written. Please try again in a moment.",
        )
        return

    if receipt.ok:
        st.success(
            f"Recorded **{change['change_name']}** in `{_display_name(path)}` and retained it "
            "in long-term memory."
        )
    else:
        ui.log_failure("Retain product change", str(receipt.error))
        ui.warning_state(
            f"Saved {change['change_name']}, but not yet retained as memory",
            "The change is in the dataset. Use Retain on the change below to try the memory "
            "write again.",
        )
    st.rerun()


def _body(agent: PulseMindAgent) -> None:
    changes = agent.changes
    github_count = (
        int((changes["source"].astype(str) == "github").sum()) if not changes.empty else 0
    )
    ui.badge_row(
        [
            ui.badge(f"{len(changes)} recorded changes", "primary", icon_name="change"),
            ui.badge(f"{github_count} from GitHub", "github"),
            ui.badge(f"{len(changes) - github_count} from Product Manager", "pm"),
        ]
    )
    if not agent.memory.is_started:
        ui.warning_state(
            "Long-term memory is not available",
            "Changes are still saved to the dataset, but they cannot be retained as memory "
            "yet. Open Memory Explorer to connect it.",
        )
    st.write("")

    _record_form(agent)

    ui.section(
        f"Recorded changes ({len(changes)})",
        "Newest first. GitHub changes show repository, PR/release reference, version and link.",
    )
    if changes.empty:
        ui.empty_state(
            "No product changes yet",
            "Record one above, or let a merged pull request or published release arrive "
            "automatically from GitHub.",
        )
        return

    retention = agent.change_retention()
    for _, change in changes.sort_values("date", ascending=False).iterrows():
        _render_change(agent, change, retention)

    with st.expander("Dataset rows", icon=":material/table:"):
        st.dataframe(
            changes.assign(date=changes["date"].dt.strftime("%Y-%m-%d")),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Product changes also arrive automatically: when engineering merges a pull request "
        "or publishes a release in the connected repository, the change is verified and "
        "retained here."
    )


def render(agent: PulseMindAgent) -> None:
    ui.page_header(
        "Product Changes",
        "Recorded changes are the bridge between feedback and outcomes. Each change is stored "
        "in the dataset and retained as memory with the problem it targeted and the outcome "
        "that was expected.",
        eyebrow="Timeline",
    )
    with ui.guard("Product Changes", retry_key="retry_changes"):
        _body(agent)
