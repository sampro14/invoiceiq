from typing import Any

from langgraph.types import interrupt

from agent.deps import Deps, Recorder
from agent.state import Correction, InvoiceFields, InvoiceState
from agent.validators import run_checks


async def human_review(state: InvoiceState, deps: Deps, rec: Recorder) -> dict[str, Any]:
    """Pause until a person approves, corrects or rejects. Resumes from the SQLite checkpoint.

    LangGraph re-runs this node from the top on resume, so nothing before `interrupt` may have a
    side effect.
    """
    answer: dict[str, Any] = interrupt({"reasons": state.review_reasons})
    before = state.fields or InvoiceFields()
    if answer["action"] == "reject":
        rec.summary = {"action": "reject"}
        return {
            "decision": "failed",
            "reviewed_by_human": True,
            "error": "Rejected by the reviewer",
        }
    values: dict[str, Any] = answer.get("corrections") or {}
    after = before.with_values(values) if values else before
    corrections = [
        Correction(path=p, before=_text(before, p), after=_text(after, p))
        for p in values
        if _text(before, p) != _text(after, p)
    ]
    checks = run_checks(after, state.suspicious_text)
    # The person is the authority: a rule that still fails is recorded as an override, not a block.
    overrides = [c.rule for c in checks if c.status == "fail"]
    confidence = {**state.confidence, **{p: 1.0 for p in values if p != "line_items"}}
    rec.summary = {
        "action": "approve",
        "corrected": [c.path for c in corrections],
        "overrides": overrides,
    }
    return {
        "fields": after,
        "checks": checks,
        "confidence": confidence,
        "decision": "approved",
        "reviewed_by_human": True,
        "corrections": corrections,
        "overrides": overrides,
    }


def _text(fields: InvoiceFields, path: str) -> str | None:
    value = fields.get(path)
    if value is None:
        return None
    return f"{value:f}" if hasattr(value, "as_tuple") else str(value)
