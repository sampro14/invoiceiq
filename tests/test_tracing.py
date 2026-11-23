import json

import pytest

from agent.state import LLMCall
from agent.tracing import LangfuseTracer, NullTracer, build_tracer
from tests.conftest import FakeLLM, good_wire, make_settings, page_text


@pytest.fixture
def exported():
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
    )
    yield LangfuseTracer(client), client, exporter


def test_no_keys_means_no_tracing():
    assert isinstance(build_tracer(make_settings()), NullTracer)


async def test_one_trace_with_a_span_per_node_and_no_document_text(harness, exported):
    tracer, client, exporter = exported
    w = good_wire()
    h = harness(FakeLLM(w), FakeLLM(w, model="offline-anchors"))
    h.deps.tracer = tracer
    tracer.start_run("r1", {"source": "test"})
    marker = "UNIQUE-BODY-TEXT-93817"
    await h.run(page_text({**w, "extra": marker}))
    tracer.end_run("r1", {"decision": "approved"})
    client.flush()
    spans = exporter.get_finished_spans()
    names = sorted(s.name for s in spans)
    assert names == sorted(
        [
            "invoice-run",
            "ingest",
            "extract_text",
            "extract",
            "cross_check",
            "second_opinion",
            "validate",
        ]
    )
    assert len({s.context.trace_id for s in spans}) == 1
    assert marker not in json.dumps([dict(s.attributes) for s in spans], default=str)
    generation = next(s for s in spans if s.name == "extract")
    assert "offline-rules" in json.dumps(dict(generation.attributes), default=str)


def test_tracing_failures_never_reach_the_run():
    class Broken:
        def __getattr__(self, _):
            raise RuntimeError("langfuse is down")

    t = LangfuseTracer(Broken())
    t.start_run("r", {})
    t.start_node("r", "ingest").generation(
        LLMCall(node="n", provider="p", model="m", tokens_in=1, tokens_out=1, cost_usd=0, latency_ms=1), "extract", {}
    )  # fmt: skip
    t.end_run("r", {})
    t.score("r", "x", 1)
    assert t.trace_url("r") is None
    t.shutdown()
