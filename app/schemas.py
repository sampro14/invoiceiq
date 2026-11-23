from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.state import (
    CheckResult,
    Correction,
    Decision,
    FileKind,
    InvoiceFields,
    LLMCall,
    NodeMetric,
    RepairRecord,
)


class RunCreated(BaseModel):
    run_id: str
    cached: bool = False


class MetricsOut(BaseModel):
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int = Field(
        description="Upload to result, wall clock. Excludes time spent waiting for a person."
    )
    nodes: list[NodeMetric]
    llm_calls: list[LLMCall]


class RunResult(BaseModel):
    run_id: str
    status: Literal["running", "awaiting_review", "completed", "failed"]
    mode: Literal["live", "offline"]
    cached: bool = False
    decision: Decision | None = None
    file_kind: FileKind | None = None
    page_count: int = 0
    fields: InvoiceFields | None = None
    confidence: dict[str, float] = {}
    disagreements: list[str] = []
    second_opinion: str = ""
    checks: list[CheckResult] = []
    review_reasons: list[str] = []
    reviewed_by_human: bool = False
    corrections: list[Correction] = []
    overrides: list[str] = []
    repairs: list[RepairRecord] = []
    path: list[str] = []
    metrics: MetricsOut | None = None
    prompt_version: str = ""
    trace_url: str | None = None
    error: str | None = None


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject"] = "approve"
    corrections: dict[str, Any] = Field(
        default_factory=dict,
        description='Field path to new value, e.g. {"amounts.total": "95507.79"}. '
        '"line_items" replaces the whole list.',
        max_length=400,
    )


class Sample(BaseModel):
    id: str
    title: str
    blurb: str
    kind: Literal["pdf", "image"]
    preview_url: str
    cached: bool


class ConfigOut(BaseModel):
    mode: Literal["live", "offline"]
    models: dict[str, str]
    uploads_enabled: bool
    max_upload_mb: int
    max_pages: int
    confidence_threshold: float
    runs_per_ip_per_hour: int
    runs_left_today: int
    tracing: bool
