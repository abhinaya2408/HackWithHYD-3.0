"""PulseMind must refuse to start on a corrupted .env, with a fixable message.

A .env written by a non-UTF-8 tool (or damaged on disk) can contain NUL bytes; the
raw python-dotenv failure for that is ``ValueError: embedded null character`` with no
hint at which file or what to do. `core.config._load_env_file` validates the file
first so the error names the file and the remedy. These tests call the loader with
throwaway paths and never read or print any env *value*.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import _load_env_file


def test_valid_utf8_env_loads(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(b"PULSEMIND_TEST_KEY=1\n")
    _load_env_file(env)  # must not raise


def test_missing_env_is_not_an_error(tmp_path: Path) -> None:
    _load_env_file(tmp_path / "absent.env")  # the app still runs on OS env vars alone


def test_nul_byte_env_raises_actionable_error(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(b"GROQ_API_KEY=gsk\x00broken\n")
    with pytest.raises(RuntimeError) as excinfo:
        _load_env_file(env)
    message = str(excinfo.value)
    assert "NUL" in message
    assert ".env" in message
    assert "Rewrite" in message
    assert "gsk" not in message  # no secret fragment is ever echoed


def test_invalid_utf8_env_raises_actionable_error(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_bytes(b"GROQ_API_KEY=\xff\xfe\x00bad\n")
    with pytest.raises(RuntimeError) as excinfo:
        _load_env_file(env)
    message = str(excinfo.value)
    assert "UTF-8" in message
    assert ".env" in message
    assert "\xff" not in message  # the offending bytes are not echoed


def test_bom_is_tolerated(tmp_path: Path) -> None:
    """A UTF-8 BOM must not break loading (some Windows editors add one)."""
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfPULSEMIND_TEST_KEY=1\n")
    _load_env_file(env)  # must not raise
