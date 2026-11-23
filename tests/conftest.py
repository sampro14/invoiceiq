import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.config import Settings
from agent.deps import Deps
from agent.graph import build_graph, checkpoint_serde
from agent.llm.base import LLMError, LLMRequest, LLMResponse
from agent.llm.router import ModelRouter
from agent.state import InvoiceState

SAMPLES = Path(__file__).parent.parent / "app" / "samples"


def make_settings(**over: Any) -> Settings:
    base: dict[str, Any] = {"offline": True, "checkpoint_db": ":memory:", "gemini_api_key": None, "groq_api_key": None,
                            "langfuse_public_key": None, "langfuse_secret_key": None}  # fmt: skip
    return Settings(_env_file=None, **{**base, **over})


def good_wire(**over: Any) -> dict[str, Any]:
    """A consistent invoice, in the flat shape models answer in."""
    base: dict[str, Any] = {
        "document_type": "invoice", "vendor_name": "Deccan Fasteners LLP", "vendor_address": "Plot 5, MIDC, Pune",
        "vendor_tax_id": "27ABCDE1234F1Z5", "invoice_number": "INV-1001", "issue_date": "2026-03-01",
        "due_date": "2026-03-31", "currency": "INR", "po_number": None, "subtotal": "300.00", "discount": None,
        "shipping": None, "tax_amount": "54.00", "tax_rate": "18", "total": "354.00",
        "line_items": [
            {"description": "Hex bolts", "quantity": "10", "unit_price": "20.00", "line_total": "200.00"},
            {"description": "Washers", "quantity": "5", "unit_price": "20.00", "line_total": "100.00"},
        ],
        "uncertain_fields": [], "suspicious_text": [],
    }  # fmt: skip
    return {**base, **over}


class FakeLLM:
    """Answers from a script. Each entry is a dict (JSON), a raw string, or an LLMError to raise."""

    vision = True

    def __init__(self, *script: Any, model: str = "offline-rules", provider: str = "fake"):
        self.model, self.provider = model, provider
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, LLMError):
            raise step
        if isinstance(step, dict):
            step = json.dumps({k: step.get(k) for k in request.schema["required"]})
        return LLMResponse(step, self.provider, self.model, 1000, 200, 5)


def page_text(wire: dict[str, Any]) -> bytes:
    """A PDF whose text layer contains the wire values, so grounding passes."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    lines = [str(v) for k, v in wire.items() if isinstance(v, str)]
    for item in wire["line_items"]:
        lines.append(" ".join(str(v) for v in item.values()))
    page.insert_text((50, 60), "\n".join(lines), fontsize=9)
    return bytes(doc.tobytes())


@pytest.fixture
def settings() -> Settings:
    return make_settings()


class Harness:
    def __init__(self, reader: FakeLLM, second: FakeLLM | None, **settings: Any):
        s = make_settings(**settings)
        router = ModelRouter([reader], second, second, max_retries=0)
        self.deps = Deps(s, router)
        self.graph = build_graph(self.deps, InMemorySaver(serde=checkpoint_serde()))
        self.reader, self.second = reader, second

    async def run(self, data: bytes, run_id: str = "r1") -> dict[str, Any]:
        self.deps.store.put(run_id, data)
        out = await self.graph.ainvoke({"run_id": run_id}, self.config(run_id))
        return dict(
            InvoiceState.model_validate({k: v for k, v in out.items() if k != "__interrupt__"})
        )

    @staticmethod
    def config(run_id: str = "r1") -> Any:
        return {"configurable": {"thread_id": run_id}}


@pytest.fixture
def harness():
    return Harness
