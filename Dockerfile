FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /srv
COPY --from=build /srv/.venv .venv
COPY agent agent
COPY app app
# Holds latest.json once a live eval has been published (.dockerignore drops everything else).
COPY eval/results eval/results
RUN mkdir -p data && chown app data
USER app
ENV PATH=/srv/.venv/bin:$PATH PYTHONUNBUFFERED=1 CHECKPOINT_DB=/srv/data/checkpoints.sqlite
EXPOSE 8000
# One worker: runs, rate limits and uploaded documents live in this process's memory.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log --timeout-graceful-shutdown 20"]
