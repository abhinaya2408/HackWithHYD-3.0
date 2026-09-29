"""Tests for the GitHub product-change detection layer.

Covers the spec's test matrix offline (no network, no real Groq/Hindsight):
valid PR-merged and release-published events, invalid webhook signature,
unsupported events, missing GitHub configuration, and GitHub API failure —
plus the graceful-fallback guarantees (GitHub failure never breaks PulseMind).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import pytest

from core.github_client import GithubClient, GithubError
from core.github_events import (
    EventNotSupported,
    normalize_event,
    product_area_from_event,
    verify_signature,
)
from core.github_webhook import GithubEventStore, make_handler
from core.hindsight_client import RetainReceipt


SECRET = "test-webhook-secret"


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


PR_PAYLOAD = {
    "action": "closed",
    "sender": {"login": "dev-one"},
    "repository": {"full_name": "acme/shop"},
    "pull_request": {
        "number": 42,
        "title": "Checkout V2: fewer steps",
        "body": "Reduced checkout steps and improved payment flow.",
        "merged": True,
        "merged_at": "2026-02-15T10:00:00Z",
        "html_url": "https://github.com/acme/shop/pull/42",
        "merged_by": {"login": "dev-two"},
        "base": {"ref": "main"},
        "merge_commit_sha": "abc123",
    },
}

RELEASE_PAYLOAD = {
    "action": "published",
    "sender": {"login": "dev-one"},
    "repository": {"full_name": "acme/shop"},
    "release": {
        "id": 7,
        "name": "Checkout V2",
        "tag_name": "v2.0.0",
        "body": "Reduced checkout steps and improved payment flow.",
        "published_at": "2026-02-15T12:00:00Z",
        "html_url": "https://github.com/acme/shop/releases/tag/v2.0.0",
        "draft": False,
        "prerelease": False,
    },
}


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def test_valid_signature_passes():
    body = json.dumps(PR_PAYLOAD).encode()
    assert verify_signature(SECRET, sign(body), body) is True


def test_invalid_signature_fails():
    body = json.dumps(PR_PAYLOAD).encode()
    wrong = sign(body, "other-secret")
    assert verify_signature(SECRET, wrong, body) is False


def test_missing_signature_or_secret_fails():
    body = b"x"
    assert verify_signature(SECRET, "", body) is False
    assert verify_signature("", sign(body), body) is False
    assert verify_signature(SECRET, "md5=deadbeef", body) is False


# ---------------------------------------------------------------------------
# Event normalization: supported, unsupported, malformed
# ---------------------------------------------------------------------------


def test_merged_pr_event_normalized():
    event = normalize_event("pull_request", PR_PAYLOAD)
    assert event.event_type == "pull_request"
    assert event.repo == "acme/shop"
    assert event.number == "42"
    assert event.title == "Checkout V2: fewer steps"
    assert event.author == "dev-two"  # merged_by wins over sender
    assert event.occurred_at == "2026-02-15T10:00:00Z"
    assert event.url.endswith("/pull/42")
    assert event.base_branch == "main"
    assert event.is_release is False


def test_release_event_normalized():
    event = normalize_event("release", RELEASE_PAYLOAD)
    assert event.is_release is True
    assert event.version == "v2.0.0"
    assert event.title == "Checkout V2"
    assert event.occurred_at == "2026-02-15T12:00:00Z"
    assert event.url.endswith("/tag/v2.0.0")


def test_pr_opened_event_is_ignored():
    payload = dict(PR_PAYLOAD, action="opened")
    with pytest.raises(EventNotSupported):
        normalize_event("pull_request", payload)


def test_pr_closed_without_merge_is_ignored():
    payload = json.loads(json.dumps(PR_PAYLOAD))
    payload["pull_request"]["merged"] = False
    with pytest.raises(EventNotSupported):
        normalize_event("pull_request", payload)


def test_release_edited_event_is_ignored():
    payload = dict(RELEASE_PAYLOAD, action="edited")
    with pytest.raises(EventNotSupported):
        normalize_event("release", payload)


def test_unsupported_event_types_are_ignored():
    for event_name in ("push", "issues", "ping", "workflow_run", "star"):
        with pytest.raises(EventNotSupported):
            normalize_event(event_name, {})


def test_ping_has_helpful_message():
    with pytest.raises(EventNotSupported, match="ping"):
        normalize_event("ping", {})


def test_product_area_inference_is_deterministic():
    event = normalize_event("release", RELEASE_PAYLOAD)
    assert product_area_from_event(event) == "Checkout"
    payment_event = normalize_event(
        "pull_request",
        {
            **PR_PAYLOAD,
            "pull_request": {**PR_PAYLOAD["pull_request"], "title": "Fix UPI retry loop"},
        },
    )
    assert product_area_from_event(payment_event) == "Payments"


# ---------------------------------------------------------------------------
# GitHub → Product Change document (and → real retain call shape)
# ---------------------------------------------------------------------------


class _StubMemory:
    """Captures the retain call instead of hitting Hindsight."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls: list[dict] = []

    def retain(self, **kwargs):
        self.calls.append(kwargs)
        return RetainReceipt(ok=self.ok, items_count=1 if self.ok else 0, document_id=kwargs.get("document_id"))


class _StubSettings:
    github_webhook_secret = SECRET
    data_dir = None


def test_store_builds_product_change_document_from_github_fields():
    memory = _StubMemory()
    store = GithubEventStore(memory, _StubSettings(), append_to_csv=False)
    event = normalize_event("release", RELEASE_PAYLOAD)

    result = store.retain(event)

    assert result.retained is True
    document = memory.calls[0]
    assert document["kind"] == "product_change"
    lowered = document["content"].lower()
    assert "detected automatically from github" in lowered
    assert "Repository: acme/shop" in document["content"]
    assert "Version / tag: v2.0.0" in document["content"]
    assert "GitHub URL: https://github.com/acme/shop/releases/tag/v2.0.0" in document["content"]
    assert "source:github" in document["tags"]
    assert "github_release" in document["tags"]
    assert document["metadata"]["source"] == "github"
    assert document["metadata"]["version_tag"] == "v2.0.0"
    assert document["when"] == "2026-02-15"
    assert document["document_id"].startswith("github-release-acme-shop-")


def test_store_retain_failure_is_reported_not_raised():
    memory = _StubMemory(ok=False)
    memory.retain = lambda **kwargs: RetainReceipt(ok=False, error="boom: quota")
    store = GithubEventStore(memory, _StubSettings(), append_to_csv=False)
    event = normalize_event("pull_request", PR_PAYLOAD)

    result = store.retain(event)

    assert result.retained is False
    assert result.status == 502
    assert "boom" in (result.retain_error or "")


# ---------------------------------------------------------------------------
# Full HTTP webhook: valid events, invalid signature, unsupported events
# ---------------------------------------------------------------------------


@pytest.fixture()
def webhook_server():
    memory = _StubMemory()
    settings = _StubSettings()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(GithubEventStore(memory, settings, append_to_csv=False), settings))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", memory
    server.shutdown()
    server.server_close()


def _post(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, dict]:
    request = urllib.request.Request(url + "/webhooks/github", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def test_webhook_pr_merged_end_to_end(webhook_server):
    url, memory = webhook_server
    body = json.dumps(PR_PAYLOAD).encode()
    status, payload = _post(
        url,
        body,
        {
            "X-Hub-Signature-256": sign(body),
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "d-1",
            "Content-Type": "application/json",
        },
    )
    assert status == 200
    assert payload["ok"] is True
    assert payload["document_id"].startswith("github-pr-acme-shop-42")
    assert len(memory.calls) == 1
    assert memory.calls[0]["metadata"]["source"] == "github"


def test_webhook_release_published_end_to_end(webhook_server):
    url, memory = webhook_server
    body = json.dumps(RELEASE_PAYLOAD).encode()
    status, payload = _post(
        url,
        body,
        {
            "X-Hub-Signature-256": sign(body),
            "X-GitHub-Event": "release",
            "X-GitHub-Delivery": "d-2",
            "Content-Type": "application/json",
        },
    )
    assert status == 200
    assert payload["ok"] is True
    assert memory.calls[0]["metadata"]["version_tag"] == "v2.0.0"


def test_webhook_rejects_invalid_signature(webhook_server):
    url, memory = webhook_server
    body = json.dumps(PR_PAYLOAD).encode()
    status, payload = _post(
        url,
        body,
        {
            "X-Hub-Signature-256": sign(body, "wrong-secret"),
            "X-GitHub-Event": "release",
            "Content-Type": "application/json",
        },
    )
    assert status == 403
    assert payload["ok"] is False
    assert memory.calls == []  # nothing retained


def test_webhook_ignores_unsupported_events(webhook_server):
    url, memory = webhook_server
    body = json.dumps({"zen": "Design for failure."}).encode()
    status, payload = _post(
        url,
        body,
        {
            "X-Hub-Signature-256": sign(body),
            "X-GitHub-Event": "push",
            "Content-Type": "application/json",
        },
    )
    assert status == 200
    assert payload.get("ignored") is True
    assert "push" in payload.get("reason", "")
    assert memory.calls == []


def test_webhook_ignores_pr_closed_without_merge(webhook_server):
    url, memory = webhook_server
    payload = json.loads(json.dumps(PR_PAYLOAD))
    payload["pull_request"]["merged"] = False
    body = json.dumps(payload).encode()
    status, payload_body = _post(
        url,
        body,
        {
            "X-Hub-Signature-256": sign(body),
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert status == 200
    assert payload_body.get("ignored") is True
    assert memory.calls == []


# ---------------------------------------------------------------------------
# Read-only client: missing configuration + API failure → graceful fallback
# ---------------------------------------------------------------------------


class _UnconfiguredSettings:
    github_repo_owner = ""
    github_repo_name = ""
    github_token = ""


class _ConfiguredSettings:
    github_repo_owner = "acme"
    github_repo_name = "shop"
    github_token = ""


def test_client_without_configuration_returns_none_and_sets_message():
    client = GithubClient(_UnconfiguredSettings())
    assert client.get_pull_request(1) is None
    assert client.latest_merged_pull_requests() == []
    assert client.latest_releases() == []
    assert "not configured" in (client.last_error or "").lower()


def test_client_api_failure_becomes_graceful_none():
    client = GithubClient(_ConfiguredSettings())

    with patch.object(client, "_get", side_effect=GithubError("GitHub API unreachable: connection refused")):
        assert client.get_pull_request(42) is None
        assert client.get_release_by_tag("v2.0.0") is None
        assert client.latest_merged_pull_requests() == []
        assert client.latest_releases() == []
    assert "unreachable" in (client.last_error or "")


def test_store_csv_append_is_deduplicated_on_redelivery(monkeypatch, tmp_path):
    """A re-delivered GitHub event must not create a duplicate CSV row."""
    import pandas as pd

    from services import feedback_analyzer as analyzer

    memory = _StubMemory()
    store = GithubEventStore(memory, _StubSettings(), append_to_csv=True)
    event = normalize_event("release", RELEASE_PAYLOAD)

    def frame(**overrides) -> "pd.DataFrame":
        row = {
            "change_id": "CHG-001", "date": "2026-02-15", "change_name": "Checkout V2",
            "product_area": "Checkout", "problem_targeted": "x", "expected_outcome": "y",
            "source": "github", "github_repo": "acme/shop", "github_number": "",
            "github_kind": "release", "github_version": "v2.0.0",
            "github_url": "https://github.com/acme/shop/releases/tag/v2.0.0",
        }
        row.update(overrides)
        return pd.DataFrame([row])

    appended: list[dict] = []
    monkeypatch.setattr(
        analyzer, "append_product_change", lambda change, *a, **k: appended.append(change)
    )

    # 1. The same GitHub URL is already recorded -> no duplicate row.
    monkeypatch.setattr(analyzer, "load_product_changes", lambda *a, **k: frame())
    store.retain(event)
    assert appended == []

    # 2. The same change is already recorded without GitHub provenance, and its date
    #    is stored as text: a legacy row. The name+date check still catches it, and it
    #    must not depend on `.dt`, which only works on a real datetime column.
    legacy = frame().drop(columns=["github_url"])
    monkeypatch.setattr(analyzer, "load_product_changes", lambda *a, **k: legacy)
    store.retain(event)
    assert appended == []

    # 3. A different change with no GitHub URL -> recorded.
    other = frame(change_id="CHG-000", change_name="Checkout V1", date="2025-12-01")
    monkeypatch.setattr(
        analyzer, "load_product_changes", lambda *a, **k: other.drop(columns=["github_url"])
    )
    store.retain(event)
    assert len(appended) == 1
    assert appended[0]["source"] == "github"


def test_client_caches_responses_within_ttl():
    client = GithubClient(_ConfiguredSettings())
    calls = {"n": 0}

    def _fake_urlopen(request, timeout):
        calls["n"] += 1

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def read(self):
                return json.dumps([{"merged_at": "2026-02-15T10:00:00Z"}]).encode()

        return _Response()

    with patch("core.github_client.urllib.request.urlopen", side_effect=_fake_urlopen):
        first = client.latest_merged_pull_requests()
        second = client.latest_merged_pull_requests()
    assert first == second
    assert calls["n"] == 1  # second call served from cache


def test_demo_trigger_produces_a_delivery_the_receiver_accepts():
    """`scripts/send_test_webhook.py` must speak the same protocol as GitHub.

    It signs and posts a delivery by hand, so this runs it against the real handler:
    if the payload shape or the signature ever drifts, the demo trigger would break
    without any other test noticing.
    """
    import threading

    from scripts.send_test_webhook import build_payload, send

    memory = _StubMemory()
    settings = _StubSettings()
    store = GithubEventStore(memory, settings, append_to_csv=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, settings))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/webhooks/github"
    try:
        for kind, expected_event in (("pr", "pull_request"), ("release", "release")):
            event_name, payload = build_payload(kind=kind, number=99, title="Checkout V9")
            assert event_name == expected_event
            status, body = send(url, event_name, payload, SECRET)
            assert status == 200, body
            assert body["ok"] is True
            assert body["event"] == expected_event
        assert len(memory.calls) == 2  # one product_change memory per delivery

        # A wrong secret is rejected rather than retained.
        event_name, payload = build_payload()
        status, body = send(url, event_name, payload, "not-the-secret")
        assert status == 403
        assert len(memory.calls) == 2
    finally:
        server.shutdown()
        server.server_close()
