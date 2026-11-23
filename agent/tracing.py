"""One Langfuse trace per run, one span per node, one generation per model call.

Tracing never breaks a run: every SDK call is guarded, and without keys nothing is sent.
Document text and images are never logged, only the extracted fields and metadata.
"""

import logging
from typing import Any, Protocol

from agent.config import Settings
from agent.state import LLMCall

log = logging.getLogger("invoiceiq.tracing")


class NodeSpan(Protocol):
    def generation(self, call: LLMCall, purpose: str, meta: dict[str, Any]) -> None: ...
    def end(self, output: dict[str, Any] | None = None, error: str | None = None) -> None: ...


class Tracer(Protocol):
    def start_run(self, run_id: str, meta: dict[str, Any]) -> None: ...
    def start_node(self, run_id: str, node: str) -> NodeSpan: ...
    def end_run(self, run_id: str, output: dict[str, Any]) -> None: ...
    def score(self, run_id: str, name: str, value: float | str, comment: str = "") -> None: ...
    def trace_url(self, run_id: str) -> str | None: ...
    def shutdown(self) -> None: ...


class _NullSpan:
    def generation(self, call: LLMCall, purpose: str, meta: dict[str, Any]) -> None: ...
    def end(self, output: dict[str, Any] | None = None, error: str | None = None) -> None: ...


class NullTracer:
    def start_run(self, run_id: str, meta: dict[str, Any]) -> None: ...
    def start_node(self, run_id: str, node: str) -> NodeSpan:
        return _NullSpan()

    def end_run(self, run_id: str, output: dict[str, Any]) -> None: ...
    def score(self, run_id: str, name: str, value: float | str, comment: str = "") -> None: ...
    def trace_url(self, run_id: str) -> str | None:
        return None

    def shutdown(self) -> None: ...


class _LangfuseSpan:
    def __init__(self, span: Any):
        self._span = span

    def generation(self, call: LLMCall, purpose: str, meta: dict[str, Any]) -> None:
        try:
            g = self._span.start_observation(
                name=purpose,
                as_type="generation",
                model=call.model,
                input={"note": "document content is not logged", **meta},
                usage_details={"input": call.tokens_in, "output": call.tokens_out},
                cost_details={"total": call.cost_usd},
                metadata={"provider": call.provider, "latency_ms": call.latency_ms},
            )
            g.end()
        except Exception:
            log.warning("langfuse generation failed", exc_info=True)

    def end(self, output: dict[str, Any] | None = None, error: str | None = None) -> None:
        try:
            if error:
                self._span.update(output=output, level="ERROR", status_message=error)
            else:
                self._span.update(output=output)
            self._span.end()
        except Exception:
            log.warning("langfuse span end failed", exc_info=True)


class LangfuseTracer:
    def __init__(self, client: Any, public: bool = True):
        self._client = client
        self._public = public
        self._roots: dict[str, Any] = {}

    def _trace_id(self, run_id: str) -> str:
        # Deterministic, so a resumed review lands on the same trace even after a restart.
        return str(self._client.create_trace_id(seed=run_id))

    def start_run(self, run_id: str, meta: dict[str, Any]) -> None:
        try:
            root = self._client.start_observation(
                name="invoice-run",
                as_type="agent",
                trace_context={"trace_id": self._trace_id(run_id)},
                input={"run_id": run_id, **meta},
                metadata=meta,
            )
            if self._public:
                root.set_trace_as_public()
            self._roots[run_id] = root
        except Exception:
            log.warning("langfuse start_run failed", exc_info=True)

    def start_node(self, run_id: str, node: str) -> NodeSpan:
        root = self._roots.get(run_id)
        if root is None:
            return _NullSpan()
        try:
            return _LangfuseSpan(root.start_observation(name=node, as_type="span"))
        except Exception:
            log.warning("langfuse start_node failed", exc_info=True)
            return _NullSpan()

    def end_run(self, run_id: str, output: dict[str, Any]) -> None:
        root = self._roots.pop(run_id, None)
        if root is None:
            return
        try:
            root.update(output=output)
            root.end()
        except Exception:
            log.warning("langfuse end_run failed", exc_info=True)

    def score(self, run_id: str, name: str, value: float | str, comment: str = "") -> None:
        try:
            self._client.create_score(
                trace_id=self._trace_id(run_id),
                name=name,
                value=value,
                data_type="CATEGORICAL" if isinstance(value, str) else "NUMERIC",
                comment=comment or None,
            )
        except Exception:
            log.warning("langfuse score failed", exc_info=True)

    def trace_url(self, run_id: str) -> str | None:
        """Blocking on first call (it looks up the project), so call it off the event loop."""
        try:
            url = self._client.get_trace_url(trace_id=self._trace_id(run_id))
            return str(url) if url else None
        except Exception:
            log.warning("langfuse trace_url failed", exc_info=True)
            return None

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            log.warning("langfuse shutdown failed", exc_info=True)


def build_tracer(settings: Settings) -> Tracer:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return NullTracer()
    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        release=settings.release,
        environment=settings.mode,
        timeout=10,
    )
    return LangfuseTracer(client, public=settings.langfuse_public_traces)
