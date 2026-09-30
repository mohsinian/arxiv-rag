# --- build stage: install all dependencies, CPU-only torch ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_PREFER_BINARY=1

WORKDIR /build
# torch first from the CPU index so the app install sees it satisfied
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# --- runtime stage: same interpreter, no build tooling, non-root ---
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Migrations + CLI wiring (arxiv-rag for one-off jobs)
COPY alembic.ini ./
COPY migrations ./migrations
COPY eval ./eval
COPY docker/entrypoint.sh /app/entrypoint.sh

RUN useradd --create-home appuser \
    && chmod +x /app/entrypoint.sh \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# Schema migrations apply on every container start, then hand off to the
# command (API server by default, or any `arxiv-rag` CLI invocation)
ENTRYPOINT ["/app/entrypoint.sh"]
EXPOSE 8000
CMD ["uvicorn", "arxiv_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
