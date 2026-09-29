"""Small, dependency-free persistence for the two things the user can change.

The profile and the appearance choice live in ``st.session_state`` for the life of the
session **and** in ``.pulsemind/user.json`` so they survive a full page reload or an app
restart. The file is optional: if it cannot be written (read-only checkout, permissions),
the app keeps working with the in-session values and simply says the choice was not
saved to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from core.config import PROJECT_ROOT
from ui.theme import MODES, normalise_mode

logger = logging.getLogger("pulsemind.ui.state")

STATE_DIR = PROJECT_ROOT / ".pulsemind"
STATE_PATH = STATE_DIR / "user.json"

PROFILE_DEFAULTS: dict[str, str] = {"name": "", "email": "", "role": ""}

_SESSION_PROFILE = "pm_profile"
_SESSION_THEME = "pm_theme"
_SESSION_LOADED = "pm_state_loaded"


def _read_file() -> dict:
    try:
        raw = STATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Could not read saved preferences: %s", exc)
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("Saved preferences are not valid JSON, ignoring them: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_file(data: dict) -> bool:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning("Could not save preferences to %s: %s", STATE_PATH, exc)
        return False


def _restore() -> None:
    """Load the saved preferences into the session exactly once per session."""
    if st.session_state.get(_SESSION_LOADED):
        return
    data = _read_file()
    profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
    st.session_state[_SESSION_PROFILE] = {
        key: str(profile.get(key, "") or "") for key in PROFILE_DEFAULTS
    }
    st.session_state[_SESSION_THEME] = normalise_mode(data.get("theme"))
    st.session_state[_SESSION_LOADED] = True


def profile() -> dict[str, str]:
    """The current profile (never ``None``; empty strings mean "not set yet")."""
    _restore()
    stored = st.session_state.get(_SESSION_PROFILE) or {}
    return {key: str(stored.get(key, "") or "") for key in PROFILE_DEFAULTS}


def theme_mode() -> str:
    """``"light"``, ``"dark"`` or ``"system"``."""
    _restore()
    return normalise_mode(st.session_state.get(_SESSION_THEME))


def set_theme(mode: str) -> bool:
    """Persist the appearance choice; returns whether it reached disk."""
    mode = normalise_mode(mode)
    _restore()
    st.session_state[_SESSION_THEME] = mode
    saved = _write_file({"theme": mode, "profile": profile()})
    if not saved:
        logger.info("Appearance choice kept for this session only (disk write failed)")
    return saved


def save_profile(name: str, email: str, role: str) -> bool:
    """Persist the profile; returns whether it reached disk."""
    _restore()
    st.session_state[_SESSION_PROFILE] = {
        "name": name.strip(),
        "email": email.strip(),
        "role": role.strip(),
    }
    saved = _write_file({"theme": theme_mode(), "profile": profile()})
    if not saved:
        logger.info("Profile kept for this session only (disk write failed)")
    return saved


def initials(name: str = "", email: str = "") -> str:
    """Two-letter avatar fallback: initials from the name, else the email local part."""
    parts = [part for part in str(name or "").replace(".", " ").split() if part]
    if parts:
        return (parts[0][:1] + (parts[1][:1] if len(parts) > 1 else "")).upper()
    local = str(email or "").split("@")[0]
    local = "".join(ch for ch in local if ch.isalnum())
    return (local[:2] or "PM").upper()


def storage_path() -> Path:
    """Where preferences are saved (used for the honest Settings note)."""
    return STATE_PATH


__all__ = [
    "MODES",
    "PROFILE_DEFAULTS",
    "STATE_PATH",
    "initials",
    "profile",
    "save_profile",
    "set_theme",
    "storage_path",
    "theme_mode",
]
