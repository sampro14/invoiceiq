from typing import Any

from agent import prompts
from agent.deps import Deps, Recorder
from agent.documents import Document
from agent.llm.base import LLMError, LLMRequest, Purpose
from agent.llm.schema import (
    SchemaError,
    parse_wire,
    wire_schema,
    wire_to_fields,
    wire_uncertainty,
)
from agent.state import InvoiceState


def build_request(
    purpose: Purpose,
    system: str,
    user_template: str,
    document: Document,
    *,
    keys: list[str] | None = None,
    transcript: bool = False,
    **values: str,
) -> LLMRequest:
    """The document always goes in as quoted data (text) or as images, never as instructions."""
    if document.text is not None:
        where, quoted = prompts.quote(document.text)
        user = user_template.format(where=where, **values) + "\n\n" + quoted
    else:
        user = user_template.format(where=prompts.IMAGES_WHERE, **values)
    return LLMRequest(
        purpose=purpose,
        system=system,
        user=user,
        schema=wire_schema(keys, transcript=transcript),
        images=document.images,
        document_text=document.text,
    )


async def extract(state: InvoiceState, deps: Deps, rec: Recorder) -> dict[str, Any]:
    document = deps.store.document(state.run_id)
    # A second model that cannot see the page needs the first one to write the page down.
    transcript = bool(document.images) and deps.router.second_vision is None
    request = build_request(
        "extract",
        prompts.load("extract_system"),
        prompts.load("extract_user"),
        document,
        transcript=transcript,
    )
    try:
        response = await rec.read(request)
    except LLMError as e:
        rec.summary = {"error": e.kind}
        return {"error": friendly(e), "decision": "failed"}
    reader = f"{response.provider}:{response.model}"
    try:
        wire = parse_wire(response.text)
    except SchemaError as e:
        rec.summary = {"reader": reader, "schema_error": str(e)}
        return {"reader": reader, "schema_error": str(e)}
    if transcript and wire.transcript:
        document.transcript = wire.transcript
    fields = wire_to_fields(wire)
    rec.summary = {"reader": reader, "lines": len(fields.line_items)}
    return {
        "reader": reader,
        "fields": fields,
        "uncertain": wire_uncertainty(wire),
        "suspicious_text": merge_suspicious(state.suspicious_text, wire.suspicious_text),
        "schema_error": None,
    }


def merge_suspicious(known: list[str], reported: list[str]) -> list[str]:
    """The model usually re-reports, in full, text the pattern scan already found a snippet of."""
    out = list(known)
    for text in (" ".join(t.split())[:200] for t in reported if t.strip()):
        core = text.casefold()
        if not any(core in k.casefold() or k.casefold() in core for k in out):
            out.append(text)
    return out[:6]


def friendly(e: LLMError) -> str:
    return {
        "quota": "The free model quota is used up for now. Try a sample, or come back later.",
        "timeout": "The model took too long to answer. Please try again.",
        "unavailable": "The model provider is not responding. Please try again.",
        "auth": "The model provider rejected this server's API key.",
        "bad_request": "The model provider could not process this document.",
        "unsupported": str(e),
    }[e.kind]
