"""GitHub webhook receiver: POST /webhooks/github → real Hindsight RETAIN.

Runs as a small standalone process next to the Streamlit app. It deliberately uses
only the standard library (no new framework) and attaches to the *same* embedded
Hindsight daemon as the app — ``HindsightMemory.start()`` reuses an already-running
daemon for this profile, so events land in the same ``pulsemind`` bank the UI reads.

Pipeline per event:

    POST /webhooks/github
      → verify X-Hub-Signature-256 (constant-time)
      → normalize (merged PR / published release only; everything else → 204)
      → build a Product Change document (source = GitHub)
      → REAL Hindsight RETAIN into the pulsemind bank
      → 200 with what was written, or 4xx with the reason

Responses never echo secrets, and failures never raise out of the handler.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from core.config import TAG_PRODUCT_CHANGE, get_settings
from core.github_events import (
    EventNotSupported,
    NormalizedGithubEvent,
    normalize_event,
    product_area_from_event,
    slugify,
    verify_signature,
)

logger = logging.getLogger("pulsemind.github_webhook")

MAX_BODY_BYTES = 512 * 1024


@dataclass
class WebhookResult:
    """What the webhook did with one delivery — also used by tests."""

    status: int
    event_type: str = ""
    action: str = ""
    message: str = ""
    document_id: str | None = None
    retained: bool = False
    retain_error: str | None = None


class GithubEventStore:
    """Turns a normalized GitHub event into a real Hindsight RETAIN (source = GitHub)."""

    def __init__(self, memory, settings, *, append_to_csv: bool = True) -> None:
        self.memory = memory
        self.settings = settings
        self.append_to_csv = append_to_csv

    def build_document(self, event: NormalizedGithubEvent) -> dict[str, Any]:
        """Product-change memory document fields, entirely from the GitHub payload."""
        area = product_area_from_event(event)
        kind_label = "GitHub Release" if event.is_release else "merged GitHub pull request"
        version_line = f"Version / tag: {event.version}" if event.is_release else ""
        base_line = f"Base branch: {event.base_branch}" if event.base_branch else ""
        number_line = f"PR number: #{event.number}" if not event.is_release else ""
        occurred = event.occurred_at or "unknown"
        content = "\n".join(
            [
                "Product change detected automatically from GitHub.",
                f"Change name: {event.title}",
                f"Source: {kind_label}",
                f"Repository: {event.repo}",
                number_line,
                version_line,
                f"Event type: {event.raw_event}",
                f"Action: {event.action}",
                f"Author: {event.author or 'unknown'}",
                f"Occurred at: {occurred}",
                base_line,
                f"Product area (inferred from repository/title): {area}",
                "",
                "Description:",
                (event.description or "(no description provided)")[:1200],
                "",
                f"GitHub URL: {event.url}",
                "",
                "This product change was recorded automatically from a GitHub event; it was "
                "not invented by the AI and not entered manually by the Product Manager.",
            ]
        )
        return {
            "content": "\n".join(line for line in content.splitlines() if line != ""),
            "kind": TAG_PRODUCT_CHANGE,
            "when": occurred[:10] if occurred and occurred != "unknown" else None,
            "context": f"Product change from GitHub ({kind_label}) in {event.repo}",
            "tags": [
                TAG_PRODUCT_CHANGE,
                "source:github",
                f"area:{area.lower().replace(' ', '-')}",
                f"repo:{slugify(event.repo)}",
                "github_release" if event.is_release else "github_pr",
            ],
            "metadata": {
                k: v
                for k, v in {
                    "source": "github",
                    "event_type": event.raw_event,
                    "action": event.action,
                    "repo": event.repo,
                    "pr_number": "" if event.is_release else event.number,
                    "release_id": event.number if event.is_release else "",
                    "version_tag": event.version,
                    "author": event.author,
                    "occurred_at": occurred,
                    "base_branch": event.base_branch,
                    "url": event.url,
                    "product_area": area,
                    "change_name": event.title,
                }.items()
                if v != ""
            },
            "document_id": f"github-{'release' if event.is_release else 'pr'}-{slugify(event.repo)}-{event.number}",
        }

    def retain(self, event: NormalizedGithubEvent) -> WebhookResult:
        document = self.build_document(event)
        if self.append_to_csv:
            try:
                # Reuse the existing CSV persistence so the change also appears in
                # Product Changes / before-after analysis. Never rewrites history,
                # and re-delivered webhooks do not create duplicate rows.
                from services import feedback_analyzer

                github_fields = {
                    "source": "github",
                    "github_repo": event.repo,
                    "github_number": "" if event.is_release else event.number,
                    "github_kind": "release" if event.is_release else "pr",
                    "github_version": event.version,
                    "github_url": event.url,
                }
                existing = feedback_analyzer.load_product_changes(self.settings)
                # `.get` (not `[...]`): legacy CSVs may pre-date the GitHub columns.
                stored_urls = existing.get("github_url") if not existing.empty else None
                same_url = (
                    stored_urls is not None and bool(event.url) and (stored_urls == event.url).any()
                )
                same_change = False
                if (
                    not existing.empty
                    and "change_name" in existing.columns
                    and "date" in existing.columns
                    and event.occurred_at
                ):
                    # Compare dates as text: `date` is a datetime column when the
                    # DataFrame came from `load_product_changes`, but rows written
                    # before this feature (or frames built by callers) may hold it as
                    # a string, where `.dt` raises instead of comparing.
                    same_change = bool(
                        (existing["change_name"] == event.title).any()
                        and (
                            existing["date"].astype(str).str.slice(0, 10)
                            == event.occurred_at[:10]
                        ).any()
                    )
                already = bool(same_url or same_change)
                if not already:
                    feedback_analyzer.append_product_change(
                        {
                            "date": (event.occurred_at or "")[:10],
                            "change_name": event.title,
                            "product_area": product_area_from_event(event),
                            "problem_targeted": (event.description or "")[:200]
                            or "see GitHub description",
                            "expected_outcome": "recorded automatically from a GitHub event",
                            **github_fields,
                        },
                        self.settings,
                    )
            except Exception as exc:  # noqa: BLE001 - CSV is a convenience, memory is the source of truth
                logger.warning("could not append GitHub change to CSV: %s", exc)
        receipt = self.memory.retain(**document)
        return WebhookResult(
            status=200 if receipt.ok else 502,
            event_type=event.raw_event,
            action=event.action,
            message=(
                f"retained product change '{event.title}' ({event.summary})"
                if receipt.ok
                else "Hindsight retain failed"
            ),
            document_id=document["document_id"],
            retained=bool(receipt.ok),
            retain_error=str(receipt.error or "")[:400] if not receipt.ok else None,
        )


def make_handler(store: GithubEventStore, settings) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to a store + settings (no globals)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "PulseMindWebhook/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.info("%s - %s", self.address_string(), format % args)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"ok": True, "service": "pulsemind-github-webhook"})
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/webhooks/github":
                self._json(404, {"ok": False, "error": "unknown path"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(413, {"ok": False, "error": "body too large or missing"})
                return
            body = self.rfile.read(length)
            signature = self.headers.get("X-Hub-Signature-256", "")
            event_name = self.headers.get("X-GitHub-Event", "")
            delivery_id = self.headers.get("X-GitHub-Delivery", "")

            if not verify_signature(settings.github_webhook_secret, signature, body):
                self._json(403, {"ok": False, "error": "invalid webhook signature"})
                return

            try:
                payload = json.loads(body.decode("utf-8", "replace"))
            except json.JSONDecodeError as exc:
                self._json(400, {"ok": False, "error": f"invalid JSON: {exc}"})
                return
            if isinstance(payload, dict):
                payload["__delivery_id"] = delivery_id

            try:
                event = normalize_event(event_name, payload)
            except EventNotSupported as exc:
                # Not an error: pushes, edits, pings etc. are simply not product changes.
                # 200 with a body (not 204) so callers/tests can read why it was skipped.
                self._json(200, {"ok": True, "ignored": True, "reason": str(exc)})
                return

            result = store.retain(event)
            status = result.status
            self._json(
                status,
                {
                    "ok": result.retained,
                    "event": event.raw_event,
                    "action": event.action,
                    "document_id": result.document_id,
                    "message": result.message,
                    **({"error": result.retain_error} if result.retain_error else {}),
                },
            )

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8765):
    """Create (not start) the webhook HTTP server with real settings/memory."""
    settings = get_settings()
    if not settings.github_webhook_secret:
        raise RuntimeError(
            "GITHUB_WEBHOOK_SECRET is not set; refusing to accept unverified webhooks"
        )

    # On an 8 GB laptop the local embedding/reranker models can take longer than the
    # server's 300 s default to load while other apps are open (Hindsight then aborts
    # startup with "Model/connection initialization did not complete within 300s" and
    # its own message recommends raising HINDSIGHT_API_MODEL_INIT_TIMEOUT). Give the
    # receiver a bigger budget unless the operator already set one.
    os.environ.setdefault("HINDSIGHT_API_MODEL_INIT_TIMEOUT", "900")

    # Attach to the same embedded Hindsight daemon the Streamlit app uses. When the
    # daemon is already running this is cheap; when it is not, this process starts it.
    from core.hindsight_client import HindsightMemory

    try:
        startup_timeout = float(
            os.environ.get("GITHUB_WEBHOOK_STARTUP_TIMEOUT", "900").strip() or "900"
        )
    except ValueError:
        startup_timeout = 900.0
    memory = HindsightMemory(settings).start(timeout=startup_timeout)
    store = GithubEventStore(memory, settings)
    return ThreadingHTTPServer((host, port), make_handler(store, settings)), memory


def serve_forever(host: str = "127.0.0.1", port: int = 8765) -> None:  # pragma: no cover
    try:
        server, memory = create_server(host, port)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or "address already in use" in str(exc).lower():
            raise SystemExit(
                f"Port {port} is already in use — the PulseMind webhook receiver is probably "
                "already running in another terminal. If you want a fresh one, set "
                "GITHUB_WEBHOOK_PORT in .env to a free port (and use the same port in the "
                "webhook URL / test trigger)."
            ) from exc
        raise
    logger.info("PulseMind GitHub webhook listening on http://%s:%s/webhooks/github", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        memory.close()


if __name__ == "__main__":  # pragma: no cover
    import os

    _host = os.environ.get("GITHUB_WEBHOOK_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        _port = int(os.environ.get("GITHUB_WEBHOOK_PORT", "8765").strip() or 8765)
    except ValueError:
        _port = 8765
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    serve_forever(_host, _port)


__all__ = [
    "GithubEventStore",
    "WebhookResult",
    "create_server",
    "make_handler",
    "serve_forever",
]
