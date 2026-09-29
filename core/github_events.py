"""GitHub webhook events → PulseMind product-change memories.

The engineering team changes the product in GitHub; PulseMind detects it automatically
and retains a Product Change memory in Hindsight. Two event types count as a
customer-facing product change:

- ``pull_request`` with ``action == "closed"`` and ``merged == true``  (engineering change)
- ``release`` with ``action == "published"``                           (released version)

A plain commit or push is deliberately NOT a product change: a commit landing on a
branch is not a customer-facing release. The PulseMind spec is explicit about that.

Normalisation is a pure function over the decoded GitHub payload so it can be tested
without any network, server, or Hindsight involvement.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_EVENTS: dict[str, str] = {
    "pull_request": "merged pull request",
    "release": "published release",
}


@dataclass
class NormalizedGithubEvent:
    """The fields PulseMind needs from a GitHub webhook event."""

    event_type: str  # "pull_request" | "release"
    repo: str  # "owner/name"
    number: str  # PR number or release id
    title: str
    description: str
    author: str
    occurred_at: str  # ISO-8601 (merged_at for PRs, published_at for releases)
    url: str
    version: str  # tag name for releases, "" for PRs
    base_branch: str  # PRs only
    is_release: bool
    action: str  # GitHub action that produced this event ("closed", "published", …)
    delivery_id: str = ""
    raw_event: str = ""  # X-GitHub-Event header value
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        kind = "GitHub Release" if self.is_release else "merged GitHub pull request"
        suffix = f" (version {self.version})" if self.is_release and self.version else ""
        return f"{kind} #{self.number} in {self.repo}{suffix}"


class EventNotSupported(Exception):
    """Raised for events PulseMind deliberately does not treat as a product change."""


def verify_signature(secret: str, signature_header: str, body: bytes) -> bool:
    """Constant-time check of GitHub's ``X-Hub-Signature-256`` header."""
    if not secret or not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature_header.strip(), f"sha256={expected}")


def normalize_event(event_name: str, payload: dict[str, Any]) -> NormalizedGithubEvent:
    """Normalise a webhook payload, or raise :class:`EventNotSupported`.

    Only merged PRs and published releases become product changes; everything else
    (pushes, opened PRs, drafts, edits, …) raises so the webhook returns 204 without
    writing memory.
    """
    event_name = (event_name or "").strip()

    if event_name == "ping":
        raise EventNotSupported("ping event received (webhook configured successfully)")

    if event_name not in SUPPORTED_EVENTS:
        raise EventNotSupported(f"event '{event_name or '?'}' is not a product-change signal")

    repo = str((payload.get("repository") or {}).get("full_name") or "")
    sender = str((payload.get("sender") or {}).get("login") or "")
    delivery = str(payload.get("__delivery_id") or "")

    if event_name == "release":
        action = str(payload.get("action") or "")
        if action != "published":
            raise EventNotSupported(f"release action '{action or '?'}' is ignored (only 'published' counts)")
        release = payload.get("release") or {}
        title = str(release.get("name") or release.get("tag_name") or "untitled release")
        return NormalizedGithubEvent(
            event_type="release",
            repo=repo,
            number=str(release.get("id") or ""),
            title=title,
            description=str(release.get("body") or "").strip(),
            author=sender,
            occurred_at=str(release.get("published_at") or ""),
            url=str(release.get("html_url") or ""),
            version=str(release.get("tag_name") or ""),
            base_branch="",
            is_release=True,
            action=action,
            delivery_id=delivery,
            raw_event=event_name,
            extra={
                "draft": str(release.get("draft", "")),
                "prerelease": str(release.get("prerelease", "")),
            },
        )

    # pull_request
    action = str(payload.get("action") or "")
    if action != "closed":
        raise EventNotSupported(f"pull_request action '{action or '?'}' is ignored (only 'closed' merges count)")
    pr = payload.get("pull_request") or {}
    if not bool(pr.get("merged")):
        raise EventNotSupported("pull request was closed without being merged (not a product change)")
    return NormalizedGithubEvent(
        event_type="pull_request",
        repo=repo,
        number=str(pr.get("number") or ""),
        title=str(pr.get("title") or "untitled pull request"),
        description=str(pr.get("body") or "").strip(),
        author=str(pr.get("merged_by", {}).get("login") or sender),
        occurred_at=str(pr.get("merged_at") or ""),
        url=str(pr.get("html_url") or ""),
        version="",
        base_branch=str((pr.get("base") or {}).get("ref") or ""),
        is_release=False,
        action=action,
        delivery_id=delivery,
        raw_event=event_name,
        extra={"merge_commit_sha": str(pr.get("merge_commit_sha") or "")},
    )


def product_area_from_event(event: NormalizedGithubEvent) -> str:
    """Deterministic area guess from the repo name (never an LLM call)."""
    repo = (event.repo or "").lower()
    haystack = f"{repo} {event.title.lower()} {event.version.lower()}"
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("Checkout", ("checkout", "cart", "coupon")),
        ("Payments", ("payment", "upi", "billing", "wallet")),
        ("Delivery", ("delivery", "shipping", "logistics", "tracking")),
        ("Search", ("search", "discovery", "ranking")),
        ("Returns", ("return", "refund")),
        ("App Performance", ("performance", "latency", "infra", "mobile")),
    ]
    for area, needles in rules:
        if any(needle in haystack for needle in needles):
            return area
    return "Support"


def slugify(text: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "-"
        for char in (text or "").strip()
    ).strip("-")[:60] or "github-change"
