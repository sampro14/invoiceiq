"""Record a real run of each sample, to replay when the free quota is used up.

    python -m scripts.cache_samples

Refuses to run offline: a recording is shown to visitors as a live result, so it has to be one.
"""

import asyncio
import json
import sys

from langgraph.checkpoint.memory import InMemorySaver

from agent.config import get_settings
from agent.deps import Deps
from agent.graph import build_graph, checkpoint_serde
from app.runs import RunManager
from app.samples import all_samples

KEEP = {"node_start", "node_end", "node_paused", "partial", "run_started"}


async def main() -> int:
    settings = get_settings()
    if not settings.live:
        print("Set GEMINI_API_KEY and GROQ_API_KEY first.", file=sys.stderr)
        return 1
    deps = Deps.from_settings(settings)
    saver = InMemorySaver(serde=checkpoint_serde())
    manager = RunManager(deps, build_graph(deps, saver), saver)
    problems = 0
    for sample in all_samples().values():
        run = manager.start(sample.path.read_bytes(), f"sample:{sample.id}")
        events = [e async for e in manager.events(run.run_id) if e and e["type"] in KEEP]
        assert run.result is not None
        if run.result.status == "failed":
            print(f"{sample.id}: failed, not recorded ({run.result.error})")
            problems += 1
            continue
        sample.cache.parent.mkdir(exist_ok=True)
        result = run.result.model_dump(mode="json")
        sample.cache.write_text(json.dumps({"events": events, "result": result}, indent=1) + "\n")
        print(f"{sample.id}: {run.result.decision}")
    await asyncio.to_thread(deps.tracer.shutdown)
    await deps.router.aclose()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
