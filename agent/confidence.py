"""Turning evidence into one number per field.

A model's own confidence is poorly calibrated, so it is only the starting point. Agreement with a
second model family and presence in the text layer move it. The constants are the ones the dev
split is used to tune; the eval harness reports how well they are calibrated (ECE).
"""

from agent.compare import Grounding
from agent.state import InvoiceState, all_paths, required_paths

BASE = 0.96
DISAGREE_CAP = 0.45
UNGROUNDED_CAP = 0.60
NO_SECOND_OPINION_CAP = 0.80  # below the 0.85 threshold: nothing is approved on one opinion
AGREE_SHRINK = 0.4


def field_confidence(state: InvoiceState, document_text: str | None) -> dict[str, float]:
    fields = state.fields
    if fields is None:
        return {}
    grounding = Grounding(document_text) if document_text else None
    required = set(required_paths(fields))
    lines_differ = "line_items" in state.disagreements
    out: dict[str, float] = {}
    for path in all_paths(fields):
        value = fields.get(path)
        is_line = path.startswith("line_items.")
        disagrees = path in state.disagreements or (is_line and lines_differ)
        if value is None:
            if path in required:
                out[path] = 0.0
            elif disagrees:
                out[path] = DISAGREE_CAP
            continue
        c = state.uncertain.get(path, BASE)
        if is_line and "line_items" in state.uncertain:
            c = min(c, state.uncertain["line_items"])
        c = min(c, 0.99)
        if disagrees:
            c = min(c, DISAGREE_CAP)
        elif state.second_status == "unavailable":
            c = min(c, NO_SECOND_OPINION_CAP)
        elif state.second_status == "done" and _was_checked(state, path):
            c = 1 - (1 - c) * AGREE_SHRINK
        if grounding is not None and grounding.supports(path, value) is False:
            c = min(c, UNGROUNDED_CAP)
        out[path] = round(c, 4)
    return out


def _was_checked(state: InvoiceState, path: str) -> bool:
    if state.second_paths is None:
        return True
    return path in state.second_paths or (
        path.startswith("line_items.") and "line_items" in state.second_paths
    )
