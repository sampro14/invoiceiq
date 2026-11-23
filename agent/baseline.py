"""The comparison system: one prompt, one call, no cross-check, no rules, no repair."""

from dataclasses import dataclass, field

from agent import prompts
from agent.deps import Deps, Recorder
from agent.documents import load_document
from agent.llm.schema import SchemaError, parse_wire, wire_to_fields, wire_uncertainty
from agent.nodes.extract import build_request
from agent.state import InvoiceFields, LLMCall, all_paths
from agent.tracing import _NullSpan


@dataclass
class BaselineResult:
    fields: InvoiceFields | None
    confidence: dict[str, float] = field(default_factory=dict)
    calls: list[LLMCall] = field(default_factory=list)


async def run_baseline(data: bytes, deps: Deps) -> BaselineResult:
    document = load_document(data, deps.settings)
    request = build_request(
        "baseline", prompts.load("baseline_system"), prompts.load("extract_user"), document
    )
    rec = Recorder("baseline", _NullSpan(), deps)
    response = await rec.read(request)
    try:
        wire = parse_wire(response.text)
    except SchemaError:
        return BaselineResult(None, calls=rec.calls)
    fields = wire_to_fields(wire)
    doubts = wire_uncertainty(wire)
    confidence = {p: doubts.get(p, 0.95) for p in all_paths(fields) if fields.get(p) is not None}
    return BaselineResult(fields, confidence, rec.calls)
