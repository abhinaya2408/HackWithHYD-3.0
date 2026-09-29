"""Central configuration for PulseMind.

Everything that can vary between machines (keys, paths, model names, Hindsight
deployment mode) is resolved here from environment variables so that no secret is
ever hardcoded. The controlled domain vocabulary lives here too, because the
feedback analyzer, the trend analyzer, the agent and the UI all have to agree on
the same set of product areas, themes and sentiment labels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time, without overriding real environment variables.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(env_path: Path | None = None) -> None:
    """Load .env, failing with a plain-language reason when the file itself is broken.

    A .env written by a non-UTF-8 tool (or damaged by a crashed editor/disk) can contain
    NUL bytes, which python-dotenv rejects as ``ValueError: embedded null character``
    with no hint at which file or why. Read the raw bytes first so the failure names the
    file and the exact defect instead. Values are never logged or printed.
    """
    env_path = env_path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        raw = env_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f".env at {env_path} could not be read: {exc}") from exc
    if b"\x00" in raw:
        raise RuntimeError(
            f".env at {env_path} contains NUL (null) bytes and is corrupted. "
            "Rewrite it as plain UTF-8 text — e.g. recreate it from .env.example and "
            "re-enter the values — then start PulseMind again."
        )
    try:
        raw.decode("utf-8-sig")  # a BOM is tolerated and stripped
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f".env at {env_path} is not valid UTF-8 text ({exc}). "
            "Rewrite it as plain UTF-8 — e.g. recreate it from .env.example and "
            "re-enter the values — then start PulseMind again."
        ) from exc
    load_dotenv(env_path, override=False)


_load_env_file()


# --------------------------------------------------------------------------
# Controlled domain vocabulary
# --------------------------------------------------------------------------

# Product areas PulseMind reasons about. Deliberately small and stable: a
# controlled vocabulary is what makes period-over-period trend math comparable.
PRODUCT_AREAS: tuple[str, ...] = (
    "Checkout",
    "Payments",
    "Delivery",
    "Search",
    "Returns",
    "App Performance",
    "Support",
)

# Themes are the semantically meaningful complaint/feedback categories.
THEMES: tuple[str, ...] = (
    "checkout_friction",
    "payment_reliability",
    "delivery_delay",
    "search_relevance",
    "return_process",
    "app_crash",
    "support_response",
)

SENTIMENTS: tuple[str, ...] = ("negative", "neutral", "positive")

# Human-readable labels for the UI.
THEME_LABELS: dict[str, str] = {
    "checkout_friction": "Checkout friction",
    "payment_reliability": "Payment / UPI reliability",
    "delivery_delay": "Delivery delay",
    "search_relevance": "Search relevance",
    "return_process": "Return process",
    "app_crash": "App crashes / performance",
    "support_response": "Support response",
}

# Hindsight tags PulseMind writes with every retained memory. Tags are the real
# Hindsight filtering mechanism, so they are the backbone of Memory Explorer.
TAG_FEEDBACK = "feedback"
TAG_PRODUCT_CHANGE = "product_change"
TAG_OUTCOME = "outcome"
TAG_EMERGING_ISSUE = "emerging_issue"
TAG_DECISION = "product_manager_decision"

MEMORY_KINDS: tuple[str, ...] = (
    TAG_FEEDBACK,
    TAG_PRODUCT_CHANGE,
    TAG_OUTCOME,
    TAG_EMERGING_ISSUE,
    TAG_DECISION,
)

MEMORY_KIND_LABELS: dict[str, str] = {
    TAG_FEEDBACK: "Customer feedback",
    TAG_PRODUCT_CHANGE: "Product changes",
    TAG_OUTCOME: "Outcomes",
    TAG_EMERGING_ISSUE: "Emerging issues",
    TAG_DECISION: "PM decisions",
}


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    groq_api_key: str
    # PulseMind's own reasoning model (themes, insights, recommendations).
    groq_model: str
    # Hindsight's extraction/consolidation model. Deliberately a *different* Groq
    # model from `groq_model`: Groq enforces rate limits per model, so this gives the
    # memory workload its own tokens-per-minute budget instead of competing with
    # PulseMind's reasoning for the same 8k TPM allowance.
    hindsight_llm_model: str
    bank_id: str
    hindsight_mode: str
    hindsight_base_url: str
    hindsight_api_key: str
    hindsight_llm_provider: str
    data_dir: Path
    # Minimum seconds between two operations that make Hindsight call the LLM provider.
    # Each extraction costs ~4.3k tokens against Hindsight's 8k tokens/minute budget
    # (its extraction prompt alone is ~1.7k tokens of fixed overhead), so two per
    # minute is the ceiling; 30s keeps us under it and avoids wasted retried RETAINs.
    # Set to 0 to disable pacing on a paid tier.
    retain_pace_seconds: float = 30.0
    # ---- GitHub (optional, read-only) -----------------------------------
    # GitHub is an optional source of product changes: merged PRs and published
    # releases are detected automatically and retained in Hindsight. Webhook calls
    # are verified with GITHUB_WEBHOOK_SECRET; GITHUB_TOKEN is a read-only token for
    # API lookups only. Missing/failed GitHub degrades to "No related GitHub
    # engineering event found." — it never blocks memory, analysis or answers.
    github_webhook_secret: str = ""
    github_token: str = ""
    github_repo_owner: str = ""
    github_repo_name: str = ""
    project_root: Path = field(default=PROJECT_ROOT)

    # ---- derived paths -------------------------------------------------
    @property
    def feedback_csv(self) -> Path:
        return self.data_dir / "feedback.csv"

    @property
    def product_changes_csv(self) -> Path:
        return self.data_dir / "product_changes.csv"

    @property
    def is_embedded(self) -> bool:
        return self.hindsight_mode.strip().lower() != "remote"

    # ---- validation ----------------------------------------------------
    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key.strip())

    def missing_requirements(self) -> list[str]:
        """Human-readable list of what still has to be configured."""
        problems: list[str] = []
        if not self.has_groq_key:
            problems.append(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and put it in .env"
            )
        if not self.is_embedded and not self.hindsight_base_url.strip():
            problems.append("HINDSIGHT_MODE=remote requires HINDSIGHT_BASE_URL")
        return problems


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except ValueError:
        return default


def get_settings() -> Settings:
    """Build settings from the environment. Cheap enough to call per rerun."""
    data_dir = Path(_env("PULSEMIND_DATA_DIR", "data") or "data")
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL", "openai/gpt-oss-120b"),
        hindsight_llm_model=_env("HINDSIGHT_LLM_MODEL", "openai/gpt-oss-20b"),
        bank_id=_env("HINDSIGHT_BANK_ID", "pulsemind"),
        hindsight_mode=_env("HINDSIGHT_MODE", "embedded"),
        hindsight_base_url=_env("HINDSIGHT_BASE_URL", "http://localhost:8888"),
        hindsight_api_key=_env("HINDSIGHT_API_KEY"),
        hindsight_llm_provider=_env("HINDSIGHT_LLM_PROVIDER", "groq"),
        data_dir=data_dir,
        retain_pace_seconds=_env_float("PULSEMIND_RETAIN_PACE_SECONDS", 30.0),
        github_webhook_secret=_env("GITHUB_WEBHOOK_SECRET"),
        github_token=_env("GITHUB_TOKEN"),
        github_repo_owner=_env("GITHUB_REPO_OWNER"),
        github_repo_name=_env("GITHUB_REPO_NAME"),
    )
