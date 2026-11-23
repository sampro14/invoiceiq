"""Seven nodes, two decisions, one repair loop.

ingest ─┬─ extract_text ──┬─ cross_check ── validate ─┬─ (approved)
        └─ extract_image ─┘       ▲                   ├─ repair ─┐
                                  └───────────────────┼──────────┘
                                                      └─ human_review
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_stream_writer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent import routing
from agent import state as state_module
from agent.deps import Deps, Recorder
from agent.nodes.cross_check import cross_check
from agent.nodes.extract import extract
from agent.nodes.human_review import human_review
from agent.nodes.ingest import ingest
from agent.nodes.repair import repair
from agent.nodes.validate import validate
from agent.state import InvoiceState, NodeMetric

NodeFn = Callable[[InvoiceState, Deps, Recorder], Awaitable[dict[str, Any]]]

NODES: dict[str, NodeFn] = {
    "ingest": ingest,
    "extract_text": extract,
    "extract_image": extract,
    "cross_check": cross_check,
    "validate": validate,
    "repair": repair,
    "human_review": human_review,
}


def _instrument(
    name: str, fn: NodeFn, deps: Deps
) -> Callable[[InvoiceState], Awaitable[dict[str, Any]]]:
    async def node(state: InvoiceState) -> dict[str, Any]:
        emit = get_stream_writer()
        attempt = state.path.count(name) + 1
        emit({"type": "node_start", "node": name, "attempt": attempt})
        span = deps.tracer.start_node(state.run_id, name)
        rec = Recorder(name, span, deps)
        started = time.perf_counter()
        try:
            update = await fn(state, deps, rec)
        except GraphInterrupt:
            emit({"type": "node_paused", "node": name, "attempt": attempt})
            span.end({"paused": True})
            raise
        except Exception as e:
            span.end(error=type(e).__name__)
            raise
        latency = int((time.perf_counter() - started) * 1000)
        metrics = state.metrics.model_copy(deep=True)
        metrics.nodes.append(NodeMetric(node=name, attempt=attempt, latency_ms=latency))
        metrics.llm_calls.extend(rec.calls)
        span.end({**rec.summary, "attempt": attempt})
        emit(
            {
                "type": "node_end",
                "node": name,
                "attempt": attempt,
                "latency_ms": latency,
                "summary": rec.summary,
            }
        )
        return {**update, "metrics": metrics, "path": [*state.path, name]}

    return node


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph:
    g: Any = StateGraph(InvoiceState)
    for name, fn in NODES.items():
        g.add_node(name, _instrument(name, fn, deps))
    g.add_edge(START, "ingest")
    g.add_conditional_edges(
        "ingest", routing.after_ingest,
        {"extract_text": "extract_text", "extract_image": "extract_image", "end": END},
    )  # fmt: skip
    for name in ("extract_text", "extract_image"):
        g.add_conditional_edges(
            name, routing.after_extract, {"cross_check": "cross_check", "end": END}
        )
    g.add_edge("cross_check", "validate")
    g.add_conditional_edges(
        "validate", routing.after_validate,
        {"repair": "repair", "human_review": "human_review", "end": END},
    )  # fmt: skip
    g.add_edge("repair", "cross_check")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer)


def checkpoint_serde() -> JsonPlusSerializer:
    """Only this project's own state classes may be rebuilt from a checkpoint file."""
    from pydantic import BaseModel

    classes = [
        (state_module.__name__, name)
        for name, obj in vars(state_module).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == state_module.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=classes)
