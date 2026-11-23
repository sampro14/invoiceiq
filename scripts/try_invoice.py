"""Run one or more invoices through the agent in the terminal and watch each step.

    uv run python -m scripts.try_invoice app/samples/clean.pdf
    uv run python -m scripts.try_invoice app/samples/*.pdf
    uv run python -m scripts.try_invoice my-invoice.pdf --json

Without GEMINI_API_KEY this uses the offline rule-based readers (digital PDFs only).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from agent.config import get_settings
from agent.deps import Deps
from agent.graph import build_graph, checkpoint_serde
from agent.state import InvoiceState, all_paths

COLOR = sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


GREEN, AMBER, RED, DIM, BOLD = "32", "33", "31", "2", "1"
BADGE = {"pass": paint("PASS", GREEN), "warn": paint("WARN", AMBER), "fail": paint("FAIL", RED)}
VERDICT = {
    "approved": paint("APPROVED", GREEN + ";" + BOLD),
    "review": paint("NEEDS A PERSON", AMBER + ";" + BOLD),
    "failed": paint("FAILED", RED + ";" + BOLD),
}


async def run_file(path: Path, deps: Deps, as_json: bool) -> None:
    graph = build_graph(deps, InMemorySaver(serde=checkpoint_serde()))
    run_id = path.name
    deps.store.put(run_id, await asyncio.to_thread(path.read_bytes))
    config: Any = {"configurable": {"thread_id": run_id}}
    print(paint(f"\n━━ {path} ━━", BOLD))
    async for event in graph.astream({"run_id": run_id}, config, stream_mode="custom"):
        if event["type"] == "node_end":
            summary = ", ".join(f"{k}={v}" for k, v in event["summary"].items())
            print(f"  ✓ {event['node']:<14}{event['latency_ms']:>6} ms  {paint(summary, DIM)}")
        elif event["type"] == "node_paused":
            print(f"  ⏸ {event['node']:<14}  {paint('paused, waiting for a person', AMBER)}")
    deps.store.drop(run_id)
    snapshot = await graph.aget_state(config)
    s = InvoiceState.model_validate(snapshot.values)

    if as_json:
        print(json.dumps(s.fields.model_dump(mode="json") if s.fields else None, indent=2))
    elif s.fields:
        print(paint("\n  Extracted fields", BOLD))
        for p in all_paths(s.fields):
            value = s.fields.get(p)
            if value is None:
                continue
            c = s.confidence.get(p)
            conf = (
                ""
                if c is None
                else paint(f"{c:4.0%}", GREEN if c >= deps.settings.confidence_threshold else AMBER)
            )
            flag = paint("  ← the two readings differ", AMBER) if p in s.disagreements else ""
            print(f"    {p:<26} {str(value)[:48]:<48} {conf}{flag}")

    print(paint("\n  Checks", BOLD))
    for check in s.checks:
        print(f"    {BADGE[check.status]}  {check.rule:<20} {paint(check.detail[:90], DIM)}")

    print(f"\n  Result: {VERDICT.get(s.decision or 'failed')}")
    for reason in s.review_reasons:
        print(f"    - {reason[:140]}")
    if s.error:
        print(f"    - {s.error}")
    m = s.metrics
    print(
        paint(f"  {m.latency_ms} ms · {m.tokens_in + m.tokens_out} tokens · ${m.cost_usd:.5f}", DIM)
    )


async def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument(
        "--json", action="store_true", help="print the ERP-ready JSON instead of the field list"
    )
    args = p.parse_args()
    settings = get_settings()
    deps = Deps.from_settings(settings)
    print(paint(f"InvoiceIQ · {settings.mode} mode", BOLD), end="")
    print(
        paint(
            "  (rule-based readers, no AI model; digital PDFs only)" if not settings.live else "",
            DIM,
        )
    )
    try:
        for path in args.files:
            if not await asyncio.to_thread(path.is_file):
                print(paint(f"\n{path}: no such file", RED))
                continue
            await run_file(path, deps, args.json)
    finally:
        await deps.router.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
