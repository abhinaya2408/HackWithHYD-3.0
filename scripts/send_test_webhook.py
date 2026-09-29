"""Send one GitHub-style, correctly signed webhook delivery to a local receiver.

GitHub itself is the real source of these events; this helper exists so the pipeline
can be demonstrated (and manually checked) on a laptop with no public URL. It mirrors
what GitHub sends: an ``X-GitHub-Event`` header, an ``X-Hub-Signature-256`` HMAC of the
raw body, and a payload shaped like a real ``pull_request`` / ``release`` event.

Read-only in spirit: it only POSTs to your own receiver and never calls GitHub.

    # merged PR (default) -> becomes a product_change memory
    python scripts/send_test_webhook.py --number 42 --title "Checkout V3"

    # published release
    python scripts/send_test_webhook.py --kind release --version v3.0.0 --title "Checkout V3"

The secret is read from settings (``GITHUB_WEBHOOK_SECRET``) and never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# `python scripts/send_test_webhook.py` puts the script's own directory on sys.path,
# not the repository root, so make `core` importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings  # noqa: E402


def sign(secret: str, body: bytes) -> str:
    """Exactly what GitHub puts in ``X-Hub-Signature-256``: sha256= + HMAC-SHA256."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def build_payload(
    *,
    kind: str = "pr",
    repo: str = "pulsemind/demo-shop",
    number: int = 42,
    title: str = "Checkout V3: fewer steps to pay",
    description: str = "Reduces the checkout flow to two steps and retries failed UPI payments.",
    version: str = "v3.0.0",
    when: str | None = None,
    base_branch: str = "main",
) -> tuple[str, dict[str, Any]]:
    """Build a GitHub-shaped event. Returns ``(event_name, payload)``."""
    occurred = when or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if kind == "release":
        payload = {
            "action": "published",
            "sender": {"login": "pulse-demo"},
            "repository": {"full_name": repo},
            "release": {
                "id": number,
                "name": title,
                "tag_name": version,
                "body": description,
                "published_at": occurred,
                "html_url": f"https://github.com/{repo}/releases/tag/{version}",
                "draft": False,
                "prerelease": False,
            },
        }
        return "release", payload
    payload = {
        "action": "closed",
        "sender": {"login": "pulse-demo"},
        "repository": {"full_name": repo},
        "pull_request": {
            "number": number,
            "title": title,
            "body": description,
            "merged": True,
            "merged_at": occurred,
            "html_url": f"https://github.com/{repo}/pull/{number}",
            "merged_by": {"login": "pulse-merger"},
            "base": {"ref": base_branch},
        },
    }
    return "pull_request", payload


def send(
    url: str,
    event_name: str,
    payload: dict[str, Any],
    secret: str,
    *,
    timeout: float = 300.0,
) -> tuple[int, dict[str, Any]]:
    """POST one signed delivery. ``timeout`` is generous: a real RETAIN runs an LLM."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "X-Hub-Signature-256": sign(secret, body),
            "X-GitHub-Event": event_name,
            "X-GitHub-Delivery": hashlib.sha1(body).hexdigest()[:32],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _receiver_url() -> str:
    """Same env vars (and defaults) the receiver in ``core.github_webhook`` listens on."""
    host = os.environ.get("GITHUB_WEBHOOK_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.environ.get("GITHUB_WEBHOOK_PORT", "8765").strip() or "8765"
    return f"http://{host}:{port}/webhooks/github"


def _default_repo(settings) -> str:
    """Use the configured repository when there is one, else a clearly-labelled demo."""
    owner = (settings.github_repo_owner or "").strip()
    name = (settings.github_repo_name or "").strip()
    return f"{owner}/{name}" if owner and name else "pulsemind/demo-shop"


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=("pr", "release"), default="pr")
    parser.add_argument("--repo", default=_default_repo(settings))
    parser.add_argument("--number", type=int, default=42, help="PR number, or release id")
    parser.add_argument("--title", default="Checkout V3: fewer steps to pay")
    parser.add_argument("--description", default=None)
    parser.add_argument("--version", default="v3.0.0", help="release tag (--kind release)")
    parser.add_argument("--when", default=None, help="ISO timestamp, defaults to now")
    parser.add_argument("--url", default=_receiver_url())
    args = parser.parse_args(argv)

    if not settings.github_webhook_secret:
        print(
            "GITHUB_WEBHOOK_SECRET is not set, so the receiver would reject the delivery.\n"
            "Set it in .env (the same value you put in the GitHub webhook settings).",
            file=sys.stderr,
        )
        return 2

    event_name, payload = build_payload(
        kind=args.kind,
        repo=args.repo,
        number=args.number,
        title=args.title,
        version=args.version,
        when=args.when,
    )
    if args.description:
        if args.kind == "release":
            payload["release"]["body"] = args.description
        else:
            payload["pull_request"]["body"] = args.description

    print(f"POST {args.url}  event={event_name}  repo={args.repo}")
    try:
        status, body = send(args.url, event_name, payload, settings.github_webhook_secret)
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        print(f"could not reach the receiver at {args.url}: {reason}", file=sys.stderr)
        if "10061" in reason or "refused" in reason.lower():
            print(
                "The webhook receiver is not running (WinError 10061 = nothing is "
                "listening on that port). Start it in ANOTHER terminal first:\n"
                "    .venv\\Scripts\\python.exe -m core.github_webhook\n"
                "then wait for the line 'PulseMind GitHub webhook listening on ...' "
                "and run this command again.",
                file=sys.stderr,
            )
        else:
            print("start the receiver first:  python -m core.github_webhook", file=sys.stderr)
        return 3
    print(f"HTTP {status}: {json.dumps(body)[:400]}")
    return 0 if status == 200 else 1


if __name__ == "__main__":  # pragma: no cover
    import logging

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
