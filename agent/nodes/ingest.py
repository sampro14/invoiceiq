import asyncio
from typing import Any

from agent import prompts
from agent.deps import Deps, Recorder
from agent.documents import DocumentError, load_document
from agent.injection import find_instructions
from agent.state import InvoiceState


async def ingest(state: InvoiceState, deps: Deps, rec: Recorder) -> dict[str, Any]:
    """Work out what the file is, and pick the cheapest reliable way to read it."""
    try:
        document = await asyncio.to_thread(
            load_document, deps.store.raw(state.run_id), deps.settings
        )
    except DocumentError as e:
        rec.summary = {"error": str(e)}
        return {"error": str(e), "decision": "failed"}
    except KeyError:
        return {
            "error": "The document is no longer in memory. Upload it again.",
            "decision": "failed",
        }
    deps.store.set_document(state.run_id, document)
    suspicious = find_instructions(document.text)
    rec.summary = {
        "file_kind": document.kind,
        "pages": document.page_count,
        "instruction_like_text": len(suspicious),
    }
    return {
        "file_kind": document.kind,
        "page_count": document.page_count,
        "prompt_version": prompts.version(),
        "suspicious_text": suspicious,
    }
