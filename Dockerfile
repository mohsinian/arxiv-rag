FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package first so dependency layers cache independently of code
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Migrations + CLI wiring (alembic runs on boot, arxiv-rag for one-off jobs)
COPY alembic.ini ./
COPY migrations ./migrations
COPY eval ./eval

RUN useradd --create-home appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 8000
CMD ["uvicorn", "arxiv_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
