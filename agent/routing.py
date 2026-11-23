"""The graph's two decisions, as pure functions."""

from dataclasses import dataclass, field
from typing import Literal

from agent.config import Settings
from agent.state import InvoiceState, required_paths


@dataclass
class Verdict:
    next: Literal["approve", "repair", "review"]
    reasons: list[str] = field(default_factory=list)


def decide(state: InvoiceState, settings: Settings) -> Verdict:
    failed = [c for c in state.checks if c.status == "fail"]
    repairs_left = state.repair_attempts < settings.max_repairs and not state.repair_stalled
    if state.fields is None:
        if repairs_left:
            return Verdict("repair")
        return Verdict("review", ["The model's answer never matched the schema"])

    low = [
        p
        for p in required_paths(state.fields)
        if state.confidence.get(p, 0.0) < settings.confidence_threshold
    ]
    if not failed and not low:
        return Verdict("approve")

    # A failure with no field behind it (instruction-like text) cannot be repaired, so spending
    # quota on the other failures first would not change the outcome.
    if failed and repairs_left and all(c.fields for c in failed):
        return Verdict("repair")

    reasons = [f"{c.rule}: {c.detail}" for c in failed]
    if low:
        if state.second_status == "unavailable":
            reasons.append(f"No second opinion: {state.second_detail}")
        if state.disagreements:
            reasons.append("The two models disagree on " + _short(state.disagreements))
        missing_agreement = [p for p in low if not _disagrees(p, state.disagreements)]
        if missing_agreement and state.second_status != "unavailable":
            reasons.append("Low confidence on " + _short(missing_agreement))
    if state.repair_attempts and failed:
        reasons.append(
            "Re-reading the document gave the same values"
            if state.repair_stalled
            else f"Still failing after {state.repair_attempts} repair attempt(s)"
        )
    return Verdict("review", reasons)


def _disagrees(path: str, disagreements: list[str]) -> bool:
    return path in disagreements or (
        path.startswith("line_items.") and "line_items" in disagreements
    )


def _short(paths: list[str], limit: int = 5) -> str:
    shown = ", ".join(paths[:limit])
    return shown + (f" and {len(paths) - limit} more" if len(paths) > limit else "")


def after_ingest(state: InvoiceState) -> Literal["extract_text", "extract_image", "end"]:
    if state.decision == "failed":
        return "end"
    return "extract_text" if state.file_kind == "pdf_text" else "extract_image"


def after_extract(state: InvoiceState) -> Literal["cross_check", "end"]:
    return "end" if state.decision == "failed" else "cross_check"


def after_validate(state: InvoiceState) -> Literal["repair", "human_review", "end"]:
    if state.decision == "approved":
        return "end"
    return "human_review" if state.decision == "review" else "repair"
