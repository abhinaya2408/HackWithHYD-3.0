"""Read-only GitHub REST lookups with caching and graceful degradation.

PulseMind never mutates anything on GitHub: no issues, no comments, no merges, no
labels — a plain ``Authorization: Bearer <token>`` header is the entire auth surface,
and every method is a GET. All failures are converted into a :class:`GithubError`
carrying a user-facing message so callers can show

    "No related GitHub engineering event found."

instead of an exception. Successful JSON responses are cached in-process for
``GithubClient.CACHE_TTL`` seconds, which keeps the demo well inside unauthenticated
rate limits when the same repo/release is looked up repeatedly.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class GithubError(Exception):
    """A GitHub lookup failed; ``user_message`` is safe to render in the UI."""


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class GithubClient:
    """Minimal read-only GitHub REST v3 client (stdlib only, no extra dependency)."""

    CACHE_TTL = 300.0
    TIMEOUT = 15.0

    def __init__(self, settings) -> None:
        self.settings = settings
        self._cache: dict[str, _CacheEntry] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.settings.github_repo_owner.strip() and self.settings.github_repo_name.strip())

    @property
    def repo_full_name(self) -> str:
        return f"{self.settings.github_repo_owner.strip()}/{self.settings.github_repo_name.strip()}"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "PulseMind"}
        token = self.settings.github_token.strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get(self, path: str, *, force_refresh: bool = False) -> Any:
        url = f"https://api.github.com{path}"
        entry = self._cache.get(url)
        now = time.time()
        if entry and not force_refresh and entry.expires_at > now:
            return entry.value
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            self.last_error = f"GitHub API HTTP {exc.code} for {path}"
            detail = exc.read().decode("utf-8", "replace")[:300] if exc.fp else ""
            raise GithubError(self.last_error + (f": {detail}" if detail else "")) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self.last_error = f"GitHub API unreachable: {exc}"
            raise GithubError(self.last_error) from exc
        self._cache[url] = _CacheEntry(value=data, expires_at=now + self.CACHE_TTL)
        return data

    # ------------------------------------------------------------------
    def get_pull_request(self, number: int | str) -> dict[str, Any] | None:
        """Merged PR detail, or None when the lookup cannot be completed."""
        if not self.configured:
            self.last_error = "GitHub is not configured (GITHUB_REPO_OWNER / GITHUB_REPO_NAME missing)"
            return None
        try:
            return dict(self._get(f"/repos/{self.repo_full_name}/pulls/{number}") or {})
        except GithubError as exc:
            self.last_error = str(exc) or self.last_error
            return None

    def get_release_by_tag(self, tag: str) -> dict[str, Any] | None:
        """Release detail by tag, or None when the lookup cannot be completed."""
        if not self.configured or not tag.strip():
            self.last_error = self.last_error or "GitHub is not configured or tag missing"
            return None
        try:
            data = dict(self._get(f"/repos/{self.repo_full_name}/releases/tags/{tag}") or {})
            return data or None
        except GithubError as exc:
            self.last_error = str(exc) or self.last_error
            return None

    def latest_merged_pull_requests(self, limit: int = 5) -> list[dict[str, Any]]:
        """Recently merged PRs (read-only). Empty list on any failure."""
        if not self.configured:
            self.last_error = "GitHub is not configured (GITHUB_REPO_OWNER / GITHUB_REPO_NAME missing)"
            return []
        try:
            data = self._get(
                f"/repos/{self.repo_full_name}/pulls?state=closed&sort=updated&direction=desc&per_page=25"
            )
        except GithubError as exc:
            self.last_error = str(exc) or self.last_error
            return []
        merged = [item for item in data if isinstance(item, dict) and item.get("merged_at")]
        return merged[: max(1, limit)]

    def latest_releases(self, limit: int = 5) -> list[dict[str, Any]]:
        """Recently published releases (read-only). Empty list on any failure."""
        if not self.configured:
            self.last_error = "GitHub is not configured (GITHUB_REPO_OWNER / GITHUB_REPO_NAME missing)"
            return []
        try:
            data = self._get(f"/repos/{self.repo_full_name}/releases?per_page=25")
        except GithubError as exc:
            self.last_error = str(exc) or self.last_error
            return []
        return [item for item in data if isinstance(item, dict)][: max(1, limit)]
