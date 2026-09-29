"""Settings: the profile shown in PulseMind, and the appearance of the whole app.

Both choices are written to ``st.session_state`` *and* to a small JSON file on this
machine, so they survive a browser reload and an app restart. The appearance control
writes the same tokens :mod:`ui.theme` injects, which is why switching theme re-skins the
sidebar, inputs, dropdown menus, cards and chart labels without a page reload.
"""

from __future__ import annotations

import streamlit as st

from core.agent import PulseMindAgent
from ui import components as ui
from ui import state as user_state
from ui.theme import MODE_ICONS, MODE_LABELS, MODES

_PROFILE_KEYS = {"name": "pm_profile_name", "email": "pm_profile_email", "role": "pm_profile_role"}


def _valid_email(value: str) -> bool:
    value = value.strip()
    if not value:
        return True  # optional field
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")


def _theme_picker() -> None:
    """Light / Dark / System as three pill buttons."""
    active = user_state.theme_mode()
    with st.container(key="pm_theme_picker"):
        columns = st.columns(len(MODES))
        for column, mode in zip(columns, MODES):
            clicked = column.button(
                MODE_LABELS[mode],
                key=f"pm_theme_{mode}",
                icon=MODE_ICONS[mode],
                type="primary" if mode == active else "secondary",
                width="stretch",
            )
            if clicked and mode != active:
                saved = user_state.set_theme(mode)
                if not saved:
                    st.session_state["pm_theme_note"] = "session"
                st.rerun()


def _profile_card() -> None:
    profile = user_state.profile()
    with ui.card():
        ui.section(
            "Profile",
            "Your name and role identify the decisions PulseMind retains for you.",
        )
        ui.profile_line(profile["name"], profile["email"], profile["role"])

        with st.form("pm_profile_form"):
            columns = st.columns(2)
            name = columns[0].text_input(
                "Full name", value=profile["name"], key=_PROFILE_KEYS["name"], placeholder="Ada Lovelace"
            )
            email = columns[1].text_input(
                "Email",
                value=profile["email"],
                key=_PROFILE_KEYS["email"],
                placeholder="ada@company.com",
            )
            role = st.text_input(
                "Role",
                value=profile["role"],
                key=_PROFILE_KEYS["role"],
                placeholder="Product Manager",
            )
            submitted = st.form_submit_button("Save profile", type="primary")

        if submitted:
            if not _valid_email(email):
                ui.warning_state(
                    "That email address doesn't look right",
                    "Check the address (for example ada@company.com) and save again.",
                )
            elif user_state.save_profile(name, email, role):
                st.success("Profile saved. It will be here the next time you open PulseMind.")
            else:
                ui.warning_state(
                    "Saved for this session only",
                    "Your details are active now, but this machine didn't allow them to be "
                    "written to disk.",
                )


def _appearance_card() -> None:
    with ui.card():
        ui.section(
            "Appearance",
            "Light and dark are built from the same design tokens, so every page, input and "
            "chart follows your choice.",
        )
        _theme_picker()
        note = st.session_state.pop("pm_theme_note", None)
        if note == "session":
            ui.warning_state(
                "Applied for this session only",
                "The theme is active now, but this machine didn't allow it to be written to disk.",
            )
        st.caption(
            "You can switch back at any time — PulseMind remembers your choice on this machine."
        )


def render(agent: PulseMindAgent | None = None) -> None:  # noqa: ARG001 - uniform page signature
    ui.page_header(
        "Settings",
        "Your profile and how PulseMind looks on this machine.",
        eyebrow="Preferences",
    )
    _profile_card()
    _appearance_card()


__all__ = ["render"]
