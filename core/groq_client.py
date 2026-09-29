"""Groq LLM access for PulseMind.

Groq is used for the things only an LLM can do: understanding raw feedback,
extracting semantic themes, reasoning over recalled history and writing
explanations and recommendations. Every *number* PulseMind reports is computed in
Python (see services/trend_analyzer.py) and handed to Groq as fixed context, so the
model is never asked to do arithmetic.

Groq's free tier is rate limited, so this wrapper retries on 429 with the delay the
API asks for and counts its own calls for display in the UI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class GroqUnavailable(RuntimeError):
    """Raised when Groq cannot be reached or is not configured."""


@dataclass
class LLMUsage:
    """Small running tally so the UI can show free-tier pressure honestly."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    failures: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, prompt: str, completion: str, usage: Any, label: str) -> None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total
        self.history.append(
            {
                "label": label,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            }
        )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON object/array out of an LLM reply, fences and prose included."""
    if not text:
        raise ValueError("empty LLM response")
    candidate = text.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from LLM response: {text[:200]}")


class GroqLLM:
    """Thin, retrying wrapper around the Groq chat completions API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.groq_model
        self.usage = LLMUsage()
        self._client: Any = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.settings.has_groq_key

    def _get_client(self) -> Any:
        if not self.configured:
            raise GroqUnavailable(
                "GROQ_API_KEY is not set. Add it to .env — free key: "
                "https://console.groq.com/keys"
            )
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.settings.groq_api_key)
        return self._client

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        json_mode: bool = False,
        label: str = "completion",
        max_retries: int = 3,
    ) -> str:
        """Run one chat completion, retrying transient/rate-limit failures."""
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            # PulseMind's configured models (openai/gpt-oss-*) are reasoning models:
            # they spend completion tokens on internal reasoning before emitting the
            # JSON document, so a tight `max_tokens` truncates it and Groq fails with
            # "max completion tokens reached before generating a valid document".
            # `low` keeps a short reasoning budget while leaving room for the document.
            kwargs["reasoning_effort"] = "low"
            kwargs["max_completion_tokens"] = max_tokens

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                self.usage.record(prompt, content, getattr(response, "usage", None), label)
                self.last_error = None
                return content
            except Exception as exc:  # noqa: BLE001 - classify by message, retry once
                last_exc = exc
                self.usage.failures += 1
                message = str(exc)
                wait = self._retry_delay(exc, attempt)
                is_rate_limit = "429" in message or "rate limit" in message.lower()
                if attempt < max_retries - 1 and (is_rate_limit or "timeout" in message.lower()):
                    logger.warning("Groq retry in %.1fs: %s", wait, message[:160])
                    time.sleep(wait)
                    continue
                break

        self.last_error = str(last_exc)
        raise GroqUnavailable(f"Groq call failed: {last_exc}") from last_exc

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None) or {}
            retry_after = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
        try:
            if retry_after is not None:
                return max(1.0, min(30.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
        return min(20.0, 2.0**attempt)

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        label: str = "json",
    ) -> Any:
        """Run a completion that must return JSON, and parse it."""
        raw = self.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            label=label,
        )
        return extract_json(raw)

    def complete_json_safe(
        self,
        prompt: str,
        *,
        system: str | None = None,
        fallback: Any = None,
        temperature: float = 0.1,
        max_tokens: int = 1600,
        label: str = "json",
    ) -> Any:
        """Like `complete_json` but returns `fallback` instead of raising."""
        try:
            return self.complete_json(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                label=label,
            )
        except (GroqUnavailable, ValueError) as exc:
            logger.warning("%s failed: %s", label, exc)
            return fallback

    def health(self) -> tuple[bool, str]:
        """Cheap reachability check used by the sidebar."""
        if not self.configured:
            return False, "GROQ_API_KEY not set"
        try:
            self.complete("Reply with the single word: ok", max_tokens=8, label="health")
            return True, f"Groq reachable ({self.model})"
        except GroqUnavailable as exc:
            return False, str(exc)
