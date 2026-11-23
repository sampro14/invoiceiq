from typing import Any

from agent.confidence import field_confidence
from agent.deps import Deps, Recorder
from agent.routing import decide
from agent.state import CheckResult, InvoiceState
from agent.validators import run_checks


async def validate(state: InvoiceState, deps: Deps, rec: Recorder) -> dict[str, Any]:
    """Rules in code, then one routing decision."""
    if state.fields is None:
        checks = [
            CheckResult(
                rule="schema", status="fail", fields=["*"],
                detail=state.schema_error or "Nothing could be extracted",
            )
        ]  # fmt: skip
    else:
        checks = run_checks(state.fields, state.suspicious_text)
    confidence = field_confidence(state, deps.store.document(state.run_id).text)
    checked = state.model_copy(update={"checks": checks, "confidence": confidence})
    verdict = decide(checked, deps.settings)
    rec.summary = {
        "failed": [c.rule for c in checks if c.status == "fail"],
        "next": verdict.next,
        "reasons": verdict.reasons,
    }
    return {
        "checks": checks,
        "confidence": confidence,
        "decision": {"approve": "approved", "review": "review", "repair": None}[verdict.next],
        "review_reasons": verdict.reasons if verdict.next == "review" else [],
    }
