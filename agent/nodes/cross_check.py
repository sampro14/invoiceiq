from typing import Any

from agent import prompts
from agent.compare import find_disagreements
from agent.confidence import BASE, field_confidence
from agent.deps import Deps, Recorder
from agent.documents import Document
from agent.llm.base import LLMError
from agent.llm.schema import SchemaError, parse_wire, paths_to_wire_keys, wire_to_fields
from agent.nodes.extract import build_request
from agent.state import InvoiceFields, InvoiceState, required_paths
from agent.validators import run_checks


async def cross_check(state: InvoiceState, deps: Deps, rec: Recorder) -> dict[str, Any]:
    """Ask a different model family the same question, without showing it the first answer.

    After a repair the stored second reading is compared again; the model is not called twice.
    """
    if state.fields is None:
        rec.summary = {"skipped": "nothing was extracted"}
        return {"second_status": "skipped", "second_detail": "Nothing was extracted to cross-check"}
    update: dict[str, Any] = {}
    second, status, paths = state.second_fields, state.second_status, state.second_paths
    if status in ("pending", "skipped"):
        paths = _paths_in_doubt(state, deps)
        if paths is not None and not paths:
            rec.summary = {"skipped": "no low-confidence fields"}
            return {
                "second_status": "skipped",
                "second_detail": "No low-confidence fields to cross-check",
            }
        second, detail = await _ask(state, deps, rec, paths)
        status = "done" if second is not None else "unavailable"
        update |= {
            "second_fields": second,
            "second_status": status,
            "second_paths": paths,
            "second_detail": detail,
        }
    if second is None:
        rec.summary = {"unavailable": update.get("second_detail", state.second_detail)}
        return update | {"disagreements": []}
    disagreements = find_disagreements(state.fields, second, paths)
    if paths is None and second.document_type != state.fields.document_type:
        # Receipts are held to fewer required fields, so in doubt apply the stricter list.
        update["fields"] = state.fields.model_copy(update={"document_type": "invoice"})
    rec.summary = {
        "disagreements": disagreements,
        "checked": "all fields" if paths is None else len(paths),
    }
    return update | {"disagreements": disagreements}


def _paths_in_doubt(state: InvoiceState, deps: Deps) -> list[str] | None:
    if deps.settings.cross_check_mode == "all":
        return None
    assert state.fields is not None
    confidence = field_confidence(state, deps.store.document(state.run_id).text)
    threshold = deps.settings.confidence_threshold
    doubt = [p for p in required_paths(state.fields) if confidence.get(p, BASE) < threshold]
    for check in run_checks(state.fields, state.suspicious_text):
        if check.status == "fail":
            doubt += [p for p in check.fields if p not in doubt]
    return doubt


async def _ask(
    state: InvoiceState, deps: Deps, rec: Recorder, paths: list[str] | None
) -> tuple[InvoiceFields | None, str]:
    document = deps.store.document(state.run_id)
    has_images = bool(document.images) and document.transcript is None
    client = deps.router.second_for(has_images, state.reader)
    if client is None:
        return None, "No independent second model is configured for this kind of document"
    view = document
    if document.transcript is not None:
        view = Document(document.kind, document.page_count, text=document.transcript)
    keys = paths_to_wire_keys(paths) if paths is not None else None
    request = build_request(
        "second_opinion",
        prompts.load("second_opinion_system"),
        prompts.load("extract_user"),
        view,
        keys=keys,
    )
    try:
        response = await rec.call(client, request)
        wire = parse_wire(response.text, required=keys)
    except LLMError as e:
        return None, f"Second model unavailable ({e.kind})"
    except SchemaError:
        return None, "Second model returned an unusable answer"
    detail = f"{response.provider}:{response.model}"
    if document.transcript is not None:
        detail += " (from a transcript, not the image)"
    return wire_to_fields(wire), detail
