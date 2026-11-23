import json
from typing import Any

from agent import prompts
from agent.compare import Grounding, values_agree
from agent.deps import Deps, Recorder
from agent.llm.base import LLMError
from agent.llm.schema import (
    PATH_TO_WIRE,
    SchemaError,
    parse_wire,
    paths_to_wire_keys,
    wire_to_fields,
    wire_uncertainty,
)
from agent.nodes.extract import build_request, merge_suspicious
from agent.state import LINE_LEAVES, InvoiceFields, InvoiceState, RepairRecord


async def repair(state: InvoiceState, deps: Deps, rec: Recorder) -> dict[str, Any]:
    """Re-ask only for the failing fields, with the rule that failed."""
    attempt = state.repair_attempts + 1
    failed = [c for c in state.checks if c.status == "fail" and c.fields]
    paths = list(dict.fromkeys(p for c in failed for p in c.fields))
    document = deps.store.document(state.run_id)
    whole = state.fields is None
    keys = None if whole else paths_to_wire_keys(paths)
    request = build_request(
        "repair",
        prompts.load("extract_system") if whole else prompts.load("repair_system"),
        prompts.load("repair_user"),
        document,
        keys=keys,
        previous="(the previous answer was not usable)" if whole else _previous(state.fields, keys),
        failures="\n".join(f"- {c.rule}: {c.detail}" for c in failed),
        keys_text="" if keys is None else ", ".join(keys),
    )
    update: dict[str, Any] = {"repair_attempts": attempt}
    record = RepairRecord(attempt=attempt, rules=[c.rule for c in failed], fields=paths, changed=[])
    try:
        response = await rec.read(request)
        wire = parse_wire(response.text, required=keys)
    except LLMError as e:
        rec.summary = {"error": e.kind}
        return update | {"repair_stalled": True, "repairs": [*state.repairs, record]}
    except SchemaError as e:
        rec.summary = {"schema_error": str(e)}
        if whole:
            update["schema_error"] = str(e)
        return update | {"repairs": [*state.repairs, record]}

    fresh = wire_to_fields(wire)
    if state.fields is None:
        rec.summary = {"recovered": "schema"}
        return update | {
            "fields": fresh,
            "uncertain": wire_uncertainty(wire),
            "suspicious_text": merge_suspicious(state.suspicious_text, wire.suspicious_text),
            "schema_error": None,
            "repairs": [*state.repairs, record.model_copy(update={"changed": ["*"]})],
        }
    merged, changed = merge(state.fields, fresh, paths, document.text)
    rec.summary = {"reasked": paths, "changed": changed}
    return update | {
        "fields": merged,
        # The same values twice means the document really says that. Asking again burns quota.
        "repair_stalled": not changed,
        "repairs": [*state.repairs, record.model_copy(update={"changed": changed})],
    }


def merge(
    old: InvoiceFields, fresh: InvoiceFields, paths: list[str], text: str | None
) -> tuple[InvoiceFields, list[str]]:
    """Take re-read values, except one that is not printed while the old one was: that is the model
    fixing the arithmetic instead of reading."""
    grounding = Grounding(text) if text else None

    def invented(path: str, before: Any, after: Any) -> bool:
        return (
            grounding is not None
            and grounding.supports(path, after) is False
            and grounding.supports(path, before) is True
        )

    values: dict[str, Any] = {}
    wants_lines = any(p == "line_items" or p.startswith("line_items.") for p in paths)
    for path in (p for p in paths if not p.startswith("line_items")):
        before, after = old.get(path), fresh.get(path)
        if not values_agree(path, before, after) and not invented(path, before, after):
            values[path] = after
    if wants_lines:
        if len(fresh.line_items) == len(old.line_items):
            whole_list = "line_items" in paths
            for i in range(len(old.line_items)):
                for leaf in LINE_LEAVES:
                    path = f"line_items.{i}.{leaf}"
                    if not whole_list and path not in paths:
                        continue
                    before, after = old.get(path), fresh.get(path)
                    if not values_agree(path, before, after) and not invented(path, before, after):
                        values[path] = after
        elif fresh.line_items and not any(
            grounding is not None and grounding.supports(leaf, getattr(line, leaf)) is False
            for line in fresh.line_items
            for leaf in LINE_LEAVES
        ):
            values["line_items"] = [line.model_dump() for line in fresh.line_items]
    return (old.with_values(values) if values else old), list(values)


def _previous(fields: InvoiceFields | None, keys: list[str] | None) -> str:
    assert fields is not None and keys is not None
    dump = fields.model_dump(mode="json")
    paths = {wire: path for path, wire in PATH_TO_WIRE.items()}
    out = {}
    for key in keys:
        if key == "line_items":
            out[key] = dump["line_items"]
        else:
            group, leaf = paths[key].split(".")
            out[key] = dump[group][leaf]
    return json.dumps(out, ensure_ascii=False)
