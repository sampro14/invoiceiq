"""Runs: start one, stream its progress, resume it after review, forget it."""

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agent.deps import Deps
from agent.state import HEADER_PATHS, LINE_LEAVES, InvoiceFields, InvoiceState
from app.schemas import MetricsOut, RunResult

log = logging.getLogger("invoiceiq.runs")

RUN_TIMEOUT_S = 120
MAX_RUNS_IN_MEMORY = 500


class RunNotFound(Exception):
    pass


class NotReviewable(Exception):
    pass


class BadCorrection(Exception):
    pass


@dataclass
class Run:
    run_id: str
    created: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    finished: bool = False
    result: RunResult | None = None
    cached: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    async def finish(self, result: RunResult) -> None:
        async with self.changed:
            self.result = result
            self.events.append({"type": "result", "result": result.model_dump(mode="json")})
            self.finished = True
            self.changed.notify_all()


class RunManager:
    def __init__(self, deps: Deps, graph: CompiledStateGraph, checkpointer: Any):
        self.deps = deps
        self.graph = graph
        self.checkpointer = checkpointer
        self._runs: dict[str, Run] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._slots = asyncio.Semaphore(deps.settings.max_concurrent_runs)

    # --- starting -----------------------------------------------------------------------------

    def start(self, data: bytes, source: str) -> Run:
        run = self._new_run()
        self.deps.store.put(run.run_id, data)
        self._spawn(self._execute(run, source, len(data)))
        return run

    def replay(self, recorded: dict[str, Any]) -> Run:
        run = self._new_run()
        run.cached = True
        self._spawn(self._replay(run, recorded))
        return run

    def _new_run(self) -> Run:
        while len(self._runs) >= MAX_RUNS_IN_MEMORY:
            self._runs.pop(next(iter(self._runs)))
        run = Run(uuid.uuid4().hex)
        self._runs[run.run_id] = run
        return run

    def _spawn(self, coro: Any) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(self, run: Run, source: str, size: int) -> None:
        run_id, tracer = run.run_id, self.deps.tracer
        started = time.perf_counter()
        try:
            async with self._slots:
                tracer.start_run(
                    run_id, {"source": source, "bytes": size, "mode": self.deps.settings.mode}
                )
                await run.publish({"type": "run_started", "run_id": run_id})
                async with asyncio.timeout(RUN_TIMEOUT_S):
                    await self._stream(run, {"run_id": run_id})
            result = await self._result(run_id, wall_ms=int((time.perf_counter() - started) * 1000))
        except Exception as e:
            log.exception("run %s failed", run_id)
            message = (
                "The run timed out." if isinstance(e, TimeoutError) else "Something went wrong."
            )
            result = RunResult(
                run_id=run_id, status="failed", decision="failed", mode=self.deps.settings.mode, error=message
            )  # fmt: skip
        finally:
            self.deps.store.drop(run_id)
        result = await self._close_trace(result)
        await run.finish(result)

    async def _stream(self, run: Run, graph_input: Any) -> None:
        chunk: Any
        async for mode, chunk in self.graph.astream(
            graph_input, _config(run.run_id), stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                await run.publish(chunk)
                continue
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                partial: dict[str, Any] = {}
                if isinstance(update.get("fields"), InvoiceFields):
                    partial["fields"] = update["fields"].model_dump(mode="json")
                if update.get("confidence"):
                    partial["confidence"] = update["confidence"]
                if update.get("checks"):
                    partial["checks"] = [c.model_dump() for c in update["checks"]]
                if partial:
                    await run.publish({"type": "partial", **partial})

    async def _replay(self, run: Run, recorded: dict[str, Any]) -> None:
        for event in recorded["events"]:
            # Replays keep the recorded pacing, capped, so the graph is still watchable.
            await asyncio.sleep(min(event.get("latency_ms", 60) / 1000, 1.5))
            await run.publish(event)
        result = RunResult.model_validate(recorded["result"])
        await run.finish(result.model_copy(update={"run_id": run.run_id, "cached": True}))

    # --- reading ------------------------------------------------------------------------------

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    async def result(self, run_id: str) -> RunResult:
        run = self._runs.get(run_id)
        if run is not None:
            if run.result is not None:
                return run.result
            return RunResult(run_id=run_id, status="running", mode=self.deps.settings.mode)
        # Not in memory (the process restarted): rebuild from the checkpoint if it survived.
        return await self._result(run_id, wall_ms=None)

    async def events(self, run_id: str) -> AsyncIterator[dict[str, Any] | None]:
        """Every event so far, then new ones as they happen. None is a heartbeat."""
        run = self._runs.get(run_id)
        if run is None:
            raise RunNotFound(run_id)
        sent = 0
        while True:
            async with run.changed:
                if sent >= len(run.events) and not run.finished:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(run.changed.wait(), timeout=15)
                batch = run.events[sent:]
                done = run.finished
            if not batch and not done:
                yield None
                continue
            for event in batch:
                yield event
            sent += len(batch)
            if done and sent >= len(run.events):
                return

    async def _result(self, run_id: str, wall_ms: int | None) -> RunResult:
        snapshot = await self.graph.aget_state(_config(run_id))
        if not snapshot.values:
            raise RunNotFound(run_id)
        s = InvoiceState.model_validate(snapshot.values)
        waiting = "human_review" in snapshot.next
        status: Literal["awaiting_review", "completed", "failed"] = "completed"
        if waiting:
            status = "awaiting_review"
        elif s.decision == "failed":
            status = "failed"
        return RunResult(
            run_id=run_id,
            status=status,
            mode=self.deps.settings.mode,
            decision=s.decision,
            file_kind=s.file_kind,
            page_count=s.page_count,
            fields=s.fields,
            confidence=s.confidence,
            disagreements=s.disagreements,
            second_opinion=s.second_detail,
            checks=s.checks,
            review_reasons=s.review_reasons,
            reviewed_by_human=s.reviewed_by_human,
            corrections=s.corrections,
            overrides=s.overrides,
            repairs=s.repairs,
            path=s.path,
            metrics=MetricsOut(
                tokens_in=s.metrics.tokens_in,
                tokens_out=s.metrics.tokens_out,
                cost_usd=round(s.metrics.cost_usd, 6),
                latency_ms=s.metrics.latency_ms if wall_ms is None else wall_ms,
                nodes=s.metrics.nodes,
                llm_calls=s.metrics.llm_calls,
            ),
            prompt_version=s.prompt_version,
            error=s.error,
        )  # fmt: skip

    async def _close_trace(self, result: RunResult) -> RunResult:
        tracer, run_id = self.deps.tracer, result.run_id
        output = result.model_dump(
            mode="json", include={"decision", "fields", "confidence", "checks", "path"}
        )
        tracer.end_run(run_id, output)
        if result.decision:
            tracer.score(
                run_id, "decision", result.decision, "; ".join(result.review_reasons)[:500]
            )
        url = await asyncio.to_thread(tracer.trace_url, run_id)
        return result.model_copy(update={"trace_url": url})

    # --- human review -------------------------------------------------------------------------

    async def review(self, run_id: str, action: str, corrections: dict[str, Any]) -> RunResult:
        run = self._runs.get(run_id)
        if run is not None and run.cached:
            raise NotReviewable("This is a cached result, so there is no paused run to resume.")
        if run is None:
            run = Run(run_id, finished=True)
            snapshot = await self.graph.aget_state(_config(run_id))
            if not snapshot.values:
                raise RunNotFound(run_id)
            run = self._runs.setdefault(run_id, run)
        async with run.lock:
            snapshot = await self.graph.aget_state(_config(run_id))
            if "human_review" not in snapshot.next:
                raise NotReviewable("This run is not waiting for review.")
            current = InvoiceState.model_validate(snapshot.values).fields or InvoiceFields()
            check_corrections(current, corrections)
            tracer = self.deps.tracer
            tracer.start_run(run_id, {"source": "human_review"})
            await self.graph.ainvoke(
                Command(resume={"action": action, "corrections": corrections}), _config(run_id)
            )
            wall = run.result.metrics.latency_ms if run.result and run.result.metrics else None
            result = await self._result(run_id, wall_ms=wall)
            result = await self._close_trace(result)
            tracer.score(run_id, "human_corrections", len(result.corrections),
                         ", ".join(c.path for c in result.corrections))  # fmt: skip
            run.result = result
            return result

    # --- housekeeping -------------------------------------------------------------------------

    async def purge(self) -> int:
        """Forget runs older than the TTL, in memory and in the checkpoint database."""
        ttl = self.deps.settings.run_ttl_s
        for run_id, run in list(self._runs.items()):
            if run.finished and run.created < time.time() - ttl:
                self._runs.pop(run_id, None)
        cutoff = (datetime.now(UTC) - timedelta(seconds=ttl)).isoformat()
        latest: dict[str, str] = {}
        async for item in self.checkpointer.alist(None):
            thread = item.config["configurable"]["thread_id"]
            latest[thread] = max(latest.get(thread, ""), item.checkpoint["ts"])
        stale = [t for t, ts in latest.items() if ts < cutoff]
        for thread in stale:
            await self.checkpointer.adelete_thread(thread)
        return len(stale)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def check_corrections(fields: InvoiceFields, corrections: dict[str, Any]) -> None:
    rows = len(fields.line_items)
    for path, value in corrections.items():
        if path == "line_items":
            if not isinstance(value, list) or len(value) > 200:
                raise BadCorrection("line_items must be a list of at most 200 rows")
            for row in value:
                if not isinstance(row, dict) or set(row) - set(LINE_LEAVES):
                    raise BadCorrection(f"Each line may only have {', '.join(LINE_LEAVES)}")
            continue
        parts = path.split(".")
        is_row = (
            len(parts) == 3
            and parts[0] == "line_items"
            and parts[1].isdecimal()
            and int(parts[1]) < rows
            and parts[2] in LINE_LEAVES
        )
        if path not in HEADER_PATHS and not is_row:
            raise BadCorrection(f"Unknown field: {path[:80]}")
        if value is not None and (
            not isinstance(value, str | int | float) or len(str(value)) > 500
        ):
            raise BadCorrection(f"{path} must be text or a number")
    if "line_items" in corrections and any(p.startswith("line_items.") for p in corrections):
        raise BadCorrection("Send either line_items or single line fields, not both")
    try:
        after = fields.with_values(corrections)
    except (ValueError, KeyError, IndexError) as e:
        raise BadCorrection("The corrected values do not fit the schema") from e
    for path, value in corrections.items():
        if path != "line_items" and value not in (None, "") and after.get(path) is None:
            raise BadCorrection(f"{path}: {value!r} is not a valid value")


def _config(run_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": run_id}}
