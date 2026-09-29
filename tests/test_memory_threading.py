"""The Hindsight client must be driven from exactly one thread.

Regression test for a real failure: the generated client keeps one aiohttp
``ClientSession`` bound to the asyncio loop of whichever thread first used it.
Reusing that session from another thread raises ``Timeout context manager should
be used inside a task``, so a webhook delivery handled by ``ThreadingHTTPServer``
(a fresh thread per request) made RETAIN fail with HTTP 502 while the Streamlit
app, on the main thread, worked.

`HindsightMemory` now funnels every client call through one owner thread. These
tests pin that invariant without needing a live server or any LLM tokens, and the
loop-bound double below reproduces the original error if the invariant is broken.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from core.config import Settings, get_settings
from core.github_webhook import GithubEventStore, make_handler
from core.hindsight_client import HindsightMemory


class LoopBoundClient:
    """Stands in for the real client: its session belongs to one thread."""

    def __init__(self) -> None:
        self.session_thread: str | None = None
        self.threads: list[str] = []

    def _request(self, name: str) -> Any:
        current = threading.current_thread().name
        if self.session_thread is None:
            self.session_thread = current  # the session is created on first use
        elif current != self.session_thread:
            # Exactly what aiohttp/anyio raise when a session crosses loops.
            raise RuntimeError("Timeout context manager should be used inside a task")
        self.threads.append(current)
        if name == "retain":
            return SimpleNamespace(success=True, items_count=1, usage=None)
        if name == "close":
            return None
        raise AssertionError(f"unexpected call {name}")

    def retain(self, **kwargs: Any) -> Any:
        return self._request("retain")

    def close(self) -> None:
        self._request("close")


@pytest.fixture()
def settings() -> Settings:
    return dataclasses.replace(get_settings(), groq_api_key="test-key")


@pytest.fixture()
def memory(settings: Settings) -> HindsightMemory:
    started = HindsightMemory(settings)
    started._client = LoopBoundClient()  # noqa: SLF001 - exercising the real dispatch
    return started


def _retain_from_another_thread(memory: HindsightMemory) -> Any:
    """Call retain on a plain worker thread, the way the webhook handler does."""
    box: dict[str, Any] = {}

    def run() -> None:
        box["receipt"] = memory.retain("Checkout steps reduced", kind="product_change")

    thread = threading.Thread(target=run, name="webhook-handler-double")
    thread.start()
    thread.join()
    return box["receipt"]


def test_retain_works_from_a_worker_thread(memory: HindsightMemory) -> None:
    """The exact production failure: a retain issued from a request thread."""
    receipt = _retain_from_another_thread(memory)

    assert receipt.ok is True
    assert receipt.error is None
    assert receipt.items_count == 1


def test_every_call_runs_on_one_owner_thread(memory: HindsightMemory) -> None:
    main_thread = threading.current_thread().name
    client = memory._client  # noqa: SLF001

    assert memory.retain("first", kind="feedback").ok
    _retain_from_another_thread(memory)
    assert memory.retain("third", kind="feedback").ok

    assert client.session_thread is not None
    assert set(client.threads) == {client.session_thread}
    # The client is never driven from the caller's thread, only from its owner.
    assert client.session_thread != main_thread
    assert client.session_thread.startswith("hindsight-memory")


def test_close_happens_on_the_owner_thread_and_releases_it(memory: HindsightMemory) -> None:
    client = memory._client  # noqa: SLF001
    assert memory.retain("first", kind="feedback").ok

    memory.close()

    assert client.threads[-1] == client.session_thread
    assert memory._owner is None  # noqa: SLF001
    assert memory.is_started is False
    # After close, retain degrades instead of touching a dead client.
    assert memory.retain("after close", kind="feedback").ok is False


def test_webhook_delivery_retains_over_http(settings: Settings, memory: HindsightMemory) -> None:
    """End-to-end over a real ThreadingHTTPServer: 200, not 502."""
    settings = dataclasses.replace(settings, github_webhook_secret="s3cret")
    store = GithubEventStore(memory, settings, append_to_csv=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, settings))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import hashlib
        import hmac
        import urllib.request

        body = json.dumps(
            {
                "action": "closed",
                "repository": {"full_name": "pulsemind/demo-shop"},
                "pull_request": {
                    "number": 7,
                    "title": "Reduce checkout steps",
                    "merged": True,
                    "merged_at": "2026-03-20T09:00:00Z",
                    "html_url": "https://github.com/pulsemind/demo-shop/pull/7",
                    "base": {"ref": "main"},
                },
            }
        ).encode()
        signature = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/webhooks/github",
            data=body,
            headers={
                "X-Hub-Signature-256": signature,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode())
        assert payload["ok"] is True
        assert payload["document_id"] == "github-pr-pulsemind-demo-shop-7"
    finally:
        server.shutdown()
        server.server_close()
