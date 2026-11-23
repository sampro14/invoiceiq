import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import aiosqlite
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from starlette.formparsers import MultiPartParser

from agent.config import Settings, get_settings
from agent.deps import Deps
from agent.documents import sniff
from agent.graph import build_graph, checkpoint_serde
from agent.llm.router import system_ssl_context
from app.limits import RateLimiter
from app.middleware import BodyLimit, SecurityHeaders
from app.runs import BadCorrection, NotReviewable, RunManager, RunNotFound
from app.samples import all_samples
from app.schemas import ConfigOut, ReviewRequest, RunCreated, RunResult, Sample

log = logging.getLogger("invoiceiq")
HERE = Path(__file__).parent
BUNDLED_REPORT = HERE.parent / "eval" / "results" / "latest.json"
MULTIPART_OVERHEAD = 64 * 1024
RUN_ID = re.compile(r"^[0-9a-f]{32}$")

# Starlette spools uploads over 1 MB to a temp file. The demo promises they never touch disk, so
# keep anything up to the body limit in memory.
MultiPartParser.spool_max_size = 64 * 1024 * 1024


def create_app(settings: Settings | None = None, deps: Deps | None = None) -> FastAPI:
    settings = settings or (deps.settings if deps else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        d = deps or Deps.from_settings(settings)
        async with aiosqlite.connect(settings.checkpoint_db) as conn:
            saver = AsyncSqliteSaver(conn, serde=checkpoint_serde())
            await saver.setup()
            manager = RunManager(d, build_graph(d, saver), saver)
            app.state.runs = manager
            app.state.report_cache = None
            janitor = asyncio.create_task(_janitor(manager))
            log.info("InvoiceIQ started in %s mode", settings.mode)
            try:
                yield
            finally:
                janitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await janitor
                await manager.close()
                await asyncio.to_thread(d.tracer.shutdown)
                await d.router.aclose()

    app = FastAPI(
        title="InvoiceIQ",
        version="0.1.0",
        description="Invoice in, validated ERP-ready JSON out, with confidence, checks, cost and latency on every run.",
        lifespan=lifespan,
    )
    app.state.limiter = RateLimiter(settings.runs_per_ip_per_hour, settings.runs_per_day)
    app.add_middleware(BodyLimit, max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD)
    app.add_middleware(SecurityHeaders)
    if settings.landing_origin:
        app.add_middleware(
            CORSMiddleware, allow_origins=[settings.landing_origin], allow_methods=["GET"]
        )

    def manager() -> RunManager:
        runs: RunManager = app.state.runs
        return runs

    def client_ip(request: Request) -> str:
        if settings.trusted_ip_header:
            value = request.headers.get(settings.trusted_ip_header)
            if value:
                return value.split(",")[0].strip()[:64]
        return request.client.host if request.client else "unknown"

    def valid_id(run_id: str) -> str:
        if not RUN_ID.match(run_id):
            raise HTTPException(404, "No such run")
        return run_id

    # --- API ----------------------------------------------------------------------------------

    @app.post("/v1/extract", response_model=RunCreated, status_code=202, tags=["runs"])
    async def extract(
        request: Request,
        file: Annotated[UploadFile | None, File(description="PDF, PNG or JPG")] = None,
        sample_id: Annotated[
            str | None, Form(description="A bundled sample instead of a file")
        ] = None,
    ) -> Any:
        """Upload one document (or name a sample) and start a run."""
        if (file is None) == (sample_id is None):
            raise HTTPException(422, "Send either a file or a sample_id")
        sample = None
        if sample_id is not None:
            sample = all_samples().get(sample_id)
            if sample is None:
                raise HTTPException(404, "No such sample")
            data, source = sample.path.read_bytes(), f"sample:{sample.id}"
        else:
            assert file is not None
            if not settings.uploads_enabled:
                raise HTTPException(
                    403, "Uploads are switched off right now. The samples still work."
                )
            data, source = await file.read(settings.max_upload_bytes + 1), "upload"
            await file.close()
            if len(data) > settings.max_upload_bytes:
                mb = settings.max_upload_bytes // (1024 * 1024)
                raise HTTPException(413, f"The file is larger than {mb} MB.")
            if sniff(data) is None:
                raise HTTPException(415, "Only PDF, PNG and JPG files are supported.")
        denied = app.state.limiter.check(client_ip(request))
        if denied is not None:
            message, wait = denied
            recorded = sample.cached_run() if sample else None
            if recorded is not None:
                return RunCreated(run_id=manager().replay(recorded).run_id, cached=True)
            return JSONResponse(
                {
                    "detail": f"{message} The sample invoices may have cached results.",
                    "retry_after": wait,
                },
                status_code=429,
                headers={"retry-after": str(wait)},
            )
        return RunCreated(run_id=manager().start(data, source).run_id)

    @app.get("/v1/runs/{run_id}/events", tags=["runs"])
    async def events(run_id: str) -> StreamingResponse:
        """Live node-by-node progress as Server-Sent Events, ending with the result."""
        if manager().get(valid_id(run_id)) is None:
            raise HTTPException(404, "No such run")

        async def stream() -> AsyncIterator[str]:
            async for event in manager().events(run_id):
                if event is None:
                    yield ": keep-alive\n\n"
                else:
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-store", "x-accel-buffering": "no"},
        )

    @app.get("/v1/runs/{run_id}", response_model=RunResult, tags=["runs"])
    async def get_run(run_id: str) -> RunResult:
        try:
            return await manager().result(valid_id(run_id))
        except RunNotFound as e:
            raise HTTPException(404, "No such run") from e

    @app.post("/v1/runs/{run_id}/review", response_model=RunResult, tags=["runs"])
    async def review(run_id: str, body: ReviewRequest) -> RunResult:
        """Approve or correct a paused run, which resumes the graph from its checkpoint."""
        try:
            return await manager().review(valid_id(run_id), body.action, body.corrections)
        except RunNotFound as e:
            raise HTTPException(404, "No such run") from e
        except NotReviewable as e:
            raise HTTPException(409, str(e)) from e
        except BadCorrection as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/v1/samples", response_model=list[Sample], tags=["samples"])
    async def samples() -> list[Sample]:
        return [
            Sample.model_validate(
                {
                    "id": s.id,
                    "title": s.title,
                    "blurb": s.blurb,
                    "kind": s.kind,
                    "preview_url": f"/v1/samples/{s.id}/preview",
                    "cached": s.cache.exists(),
                }
            )
            for s in all_samples().values()
        ]

    @app.get("/v1/samples/{sample_id}/preview", tags=["samples"])
    async def sample_preview(sample_id: str) -> FileResponse:
        sample = all_samples().get(sample_id)
        if sample is None:
            raise HTTPException(404, "No such sample")
        return FileResponse(sample.preview, headers={"cache-control": "public, max-age=86400"})

    @app.get("/v1/eval/latest", tags=["eval"])
    async def eval_latest() -> JSONResponse:
        """Latest eval report published by CI."""
        report = await _eval_report(app, settings)
        if report is None:
            raise HTTPException(404, "No eval report has been published yet")
        return JSONResponse(report, headers={"cache-control": "public, max-age=300"})

    @app.get("/v1/config", response_model=ConfigOut, tags=["service"])
    async def config() -> ConfigOut:
        models = (
            {
                "reader": settings.primary_model,
                "second_text": settings.second_model if settings.groq_api_key else "not configured",
                "second_vision": settings.second_vision_model
                if settings.groq_api_key
                else "not configured",
            }
            if settings.live
            else {
                "reader": "offline-rules",
                "second_text": "offline-anchors",
                "second_vision": "not configured",
            }
        )
        return ConfigOut(
            mode=settings.mode,
            models=models,
            uploads_enabled=settings.uploads_enabled,
            max_upload_mb=settings.max_upload_bytes // (1024 * 1024),
            max_pages=settings.max_pages,
            confidence_threshold=settings.confidence_threshold,
            runs_per_ip_per_hour=settings.runs_per_ip_per_hour,
            runs_left_today=app.state.limiter.remaining_today(),
            tracing=bool(settings.langfuse_public_key and settings.langfuse_secret_key),
        )

    @app.head("/healthz", include_in_schema=False)
    @app.get("/healthz", tags=["service"])
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(HERE / "static" / "index.html", headers={"cache-control": "no-cache"})

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app


async def _janitor(manager: RunManager) -> None:
    while True:
        try:
            await manager.purge()
        except Exception:
            log.exception("purge failed")
        await asyncio.sleep(600)


async def _eval_report(app: FastAPI, settings: Settings) -> dict[str, Any] | None:
    cached = app.state.report_cache
    if cached and cached[0] > time.time() - 300:
        report: dict[str, Any] | None = cached[1]
        return report
    report = None
    if settings.eval_report_url:
        try:
            async with httpx.AsyncClient(timeout=5, verify=system_ssl_context()) as http:
                r = await http.get(settings.eval_report_url)
                r.raise_for_status()
                report = r.json()
        except (httpx.HTTPError, ValueError):
            log.warning("could not fetch the published eval report", exc_info=True)
    if report is None and BUNDLED_REPORT.exists():
        report = json.loads(BUNDLED_REPORT.read_text())
    app.state.report_cache = (time.time(), report)
    return report


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
