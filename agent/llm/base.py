from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Purpose = Literal["extract", "second_opinion", "repair", "baseline", "judge"]


@dataclass(slots=True)
class LLMRequest:
    purpose: Purpose
    system: str
    user: str
    schema: dict[str, Any]
    schema_name: str = "invoice"
    images: list[bytes] = field(default_factory=list)  # JPEG
    # The raw document text, already quoted inside `user`. Only the offline readers use it.
    document_text: str | None = None
    max_output_tokens: int = 4096


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


ErrorKind = Literal["quota", "timeout", "auth", "unavailable", "bad_request", "unsupported"]


class LLMError(Exception):
    def __init__(self, kind: ErrorKind, message: str, provider: str = ""):
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.provider = provider

    @property
    def retryable(self) -> bool:
        return self.kind in ("timeout", "unavailable")


class LLMClient(Protocol):
    provider: str
    model: str
    vision: bool

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


def error_kind(status: int) -> ErrorKind:
    if status == 429:
        return "quota"
    if status in (401, 403):
        return "auth"
    if status >= 500 or status == 408:
        return "unavailable"
    return "bad_request"
