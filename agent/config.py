"""All tunables live here and come from environment variables.

Model IDs are config, not code: free-tier model lists change often, and a new model
should be re-scored by the eval harness before it is switched on.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    # --- Models ---------------------------------------------------------------------------
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    primary_model: str = "gemini-3.5-flash-lite"
    # "minimal" | "low" | "medium" | "high" | "" (let the model decide). `minimal` is rejected
    # by Gemini 3.8 Flash and Gemini 4, so change this together with primary_model.
    primary_thinking_level: str = "minimal"
    # Gemini 3.5+ ignores `temperature` and newer models reject it, so it is only sent when set.
    primary_temperature: float | None = None

    second_model: str = "openai/gpt-oss-120b"
    # gpt-oss is text-only. Scans need a second model that can see the page. Set to "" to make
    # the second model check a transcript written by the first one (cheaper, less independent).
    second_vision_model: str = "qwen/qwen3.8-27b"
    second_reasoning_effort: str = "low"
    second_vision_reasoning_effort: str = "none"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Used only when the primary is out of quota or down. Served by Groq.
    fallback_model: str = "qwen/qwen3.8-27b"

    # USD per 1M tokens as "model=in/out", comma separated. Overrides agent/llm/pricing.py.
    price_overrides: str = ""

    llm_timeout_s: float = 30.0
    llm_max_retries: int = 1

    # Forces the deterministic, no-network readers even when keys are present.
    offline: bool = False

    # --- Agent behaviour (tuned on the dev split, never by feel) -----------------------------
    confidence_threshold: float = 0.85
    max_repairs: int = 2
    cross_check_mode: Literal["all", "low_confidence"] = "all"
    vision_max_edge: int = 1600
    vision_jpeg_quality: int = 85
    min_chars_per_page: int = 50

    # --- Guardrails ------------------------------------------------------------------------
    max_upload_bytes: int = 5 * 1024 * 1024
    max_pages: int = 5
    runs_per_ip_per_hour: int = 5
    runs_per_day: int = 50
    max_concurrent_runs: int = 2
    # Abuse switch: false leaves only the bundled samples available.
    uploads_enabled: bool = True
    # Header set by a proxy we trust (Render sits behind Cloudflare: "cf-connecting-ip").
    # Empty means use the socket address, which a client cannot spoof.
    trusted_ip_header: str = ""

    # --- Service ---------------------------------------------------------------------------
    checkpoint_db: str = "checkpoints.sqlite"
    run_ttl_s: int = 24 * 3600
    # Where CI publishes the eval report. Empty means serve the copy bundled in the image.
    eval_report_url: str = ""
    # Origin of the static landing page that wakes the service (GitHub Pages).
    landing_origin: str = ""

    # --- Tracing ---------------------------------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # Makes every trace viewable by anyone with its link, which the demo relies on.
    langfuse_public_traces: bool = True
    release: str = Field(default="local", validation_alias="RENDER_GIT_COMMIT")

    @property
    def live(self) -> bool:
        return bool(self.gemini_api_key) and not self.offline

    @property
    def mode(self) -> Literal["live", "offline"]:
        return "live" if self.live else "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
