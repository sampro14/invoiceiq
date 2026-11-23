from dataclasses import dataclass, field

from agent.config import Settings
from agent.documents import DocumentStore
from agent.llm.base import LLMClient, LLMRequest, LLMResponse
from agent.llm.pricing import cost_usd, parse_overrides
from agent.llm.router import ModelRouter, build_router
from agent.state import LLMCall
from agent.tracing import NodeSpan, NullTracer, Tracer, build_tracer


@dataclass
class Deps:
    settings: Settings
    router: ModelRouter
    store: DocumentStore = field(default_factory=DocumentStore)
    tracer: Tracer = field(default_factory=NullTracer)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Deps":
        return cls(settings, build_router(settings), DocumentStore(), build_tracer(settings))


@dataclass
class Recorder:
    """Collects what one node execution did: model calls, and a summary for the UI and the trace."""

    node: str
    span: NodeSpan
    deps: Deps
    calls: list[LLMCall] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)

    async def read(self, request: LLMRequest) -> LLMResponse:
        return self._record(request, await self.deps.router.read(request))

    async def call(self, client: LLMClient, request: LLMRequest) -> LLMResponse:
        return self._record(request, await self.deps.router.call(client, request))

    def _record(self, request: LLMRequest, r: LLMResponse) -> LLMResponse:
        overrides = parse_overrides(self.deps.settings.price_overrides)
        call = LLMCall(
            node=self.node,
            provider=r.provider,
            model=r.model,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            cost_usd=cost_usd(r.model, r.tokens_in, r.tokens_out, overrides),
            latency_ms=r.latency_ms,
        )
        self.calls.append(call)
        self.span.generation(
            call, request.purpose, {"images": len(request.images), "chars": len(request.user)}
        )
        return r
