"""Design-token and memory-status regressions.

`ui/theme.py` is the single source of truth for colour, so these tests guard the two
promises the UI makes:

* text that carries meaning clears WCAG AA (4.5:1 for normal text) in **both** themes —
  which is why filled controls use the deeper `cta` Rausch and not `primary`; and
* the memory indicator only claims a connection that was actually verified, rather than
  inferring "Connected" from a client object existing.
"""

from __future__ import annotations

import pytest

from core.config import get_settings
from core.hindsight_client import HindsightMemory
from ui import components as ui
from ui.theme import THEMES, _STYLES

AA_NORMAL_TEXT = 4.5


def _relative_luminance(colour: str) -> float:
    value = colour.lstrip("#")
    channels = [int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [
        channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_filled_control_labels_meet_aa(mode: str) -> None:
    """White label on the CTA fill, at rest and on hover."""
    tokens = THEMES[mode]
    for token in ("cta", "cta-hover"):
        ratio = contrast_ratio(tokens["on-primary"], tokens[token])
        assert ratio >= AA_NORMAL_TEXT, f"{mode} {token}: {ratio:.2f}:1"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_running_text_meets_aa(mode: str) -> None:
    tokens = THEMES[mode]
    for token in ("ink", "body", "muted"):
        ratio = contrast_ratio(tokens[token], tokens["canvas"])
        assert ratio >= AA_NORMAL_TEXT, f"{mode} {token}: {ratio:.2f}:1"
    link_ratio = contrast_ratio(tokens["link"], tokens["canvas"])
    assert link_ratio >= AA_NORMAL_TEXT, f"{mode} link: {link_ratio:.2f}:1"


def test_white_text_is_never_placed_on_the_shallow_accent() -> None:
    """`--pm-primary` is an accent; any white-on-accent text must use `--pm-cta`.

    Catches a regression where a new rule reintroduces white text on #ff385c (3.52:1).
    """
    for block in _STYLES.split("}"):
        if "background: var(--pm-primary)" in block or "background:var(--pm-primary)" in block:
            assert "var(--pm-on-primary)" not in block, block.strip()


class _UnreachableClient:
    def get_version(self) -> dict:
        raise RuntimeError("connection refused")


class _ReachableClient:
    def get_version(self) -> dict:
        return {"version": "test"}


def test_check_connection_distinguishes_reachable_from_unreachable() -> None:
    """The probe must not report a connection that was never made."""
    memory = HindsightMemory(get_settings())
    assert memory.check_connection() is False  # no client at all

    memory._client = _UnreachableClient()  # noqa: SLF001 - exercising the probe directly
    assert memory.check_connection() is False
    assert memory.last_error  # the real cause is kept for the logs

    memory._client = _ReachableClient()  # noqa: SLF001
    assert memory.check_connection() is True


def _fake_agent(*, started: bool, probe=None, total: int | None = None):
    memory = type("Memory", (), {"is_started": started})()
    if probe is not None:
        memory.check_connection = probe

    class Agent:
        pass

    agent = Agent()
    agent.memory = memory
    if total is not None:
        agent.memory_overview = lambda: {"total": total}
    return agent


def _probe(monkeypatch, agent):
    """Evaluate one state with an empty probe cache, as a fresh session would."""
    monkeypatch.setattr(ui.st, "session_state", {})
    return ui.memory_state(agent)


def test_memory_state_wording_matches_what_was_checked(monkeypatch) -> None:
    # Not started: no claim about reachability at all.
    assert _probe(monkeypatch, _fake_agent(started=False))[:2] == ("pm-dot--idle", "Not connected")

    # Started but the probe failed: honest, not "Connected".
    failing = _fake_agent(started=True, probe=lambda: False)
    assert _probe(monkeypatch, failing)[:2] == ("pm-dot--idle", "Connection problem")

    # Started without a probe available: admit we cannot tell.
    assert _probe(monkeypatch, _fake_agent(started=True))[:2] == (
        "pm-dot--idle",
        "Status unavailable",
    )

    # A verified round-trip is what earns "Connected".
    healthy = _fake_agent(started=True, probe=lambda: True, total=23)
    assert _probe(monkeypatch, healthy) == ("pm-dot--ok", "Connected", "23 memories")


def test_memory_state_probe_is_cached_within_its_ttl(monkeypatch) -> None:
    """The probe costs a round-trip, so it must not run on every single rerun."""
    monkeypatch.setattr(ui.st, "session_state", {})
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return True

    agent = _fake_agent(started=True, probe=probe, total=23)
    assert ui.memory_state(agent)[1] == "Connected"
    assert ui.memory_state(agent)[1] == "Connected"
    assert len(calls) == 1
