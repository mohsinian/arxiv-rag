# arxiv-rag

Ingest papers from the arXiv API, store them in a relational database, serve
aggregated data through a visualization API, and answer natural-language
questions about the dataset with a Retrieval-Augmented Generation (RAG)
pipeline.

## Status

- [x] Data ingestion (arXiv API, pagination, throttling, safe re-runs)
- [x] Preprocessing (Atom/XML parsing, normalization)
- [x] Storage (relational schema + migrations)
- [x] Docker stack (database + API, no host Python needed)
- [ ] Visualization API (aggregations for charting)
- [ ] RAG pipeline (embeddings, retrieval, `/ask` endpoint)
- [ ] Retrieval eval (hit@k on hand-written questions)

## Run the stack (Docker)

Requires only Docker and a `.env`:

```bash
cp .env.example .env
docker compose up -d --build
curl http://localhost:8000/health   # {"status":"ok","db":"ok"}
```

The `api` service waits for a healthy database, applies Alembic migrations
on boot, then serves with uvicorn. The database is Postgres 16 with the
pgvector extension (auto-enabled on first start).

## Ingest & manage data (Docker)

The API image doubles as the CLI runner — no host Python involved:

```bash
docker compose run --rm api arxiv-rag ingest --categories cs.AI,cs.LG,cs.CL --limit 1000
docker compose run --rm api arxiv-rag wipe                     # full reset
docker compose run --rm api arxiv-rag wipe --embeddings-only   # drop the RAG index only
```

- Requests are throttled to one per 3 seconds (arXiv's etiquette guideline).
- Re-running is safe: unchanged papers are skipped, changed ones updated —
  the summary reports `N new, N updated, N unchanged`.
- A crash mid-ingest keeps every already-fetched page; re-running continues.

## Browsing the data

- **Adminer** (web UI): `make ui` → http://localhost:8081
  (server `db`, user/password/database `arxiv_rag`).
- **Terminal**: `make db-shell` (pgcli on the host, reads `DATABASE_URL`).

## Development (host)

Requires Python ≥3.10 and Docker (WSL integration enabled on WSL2).

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # or: uv venv + uv pip install
cp .env.example .env
make up-db         # database only
make test          # unit tests, no Docker needed
make test-integration
make lint
```

Code layout: `src/arxiv_rag/` (`ingest/` fetch+parse+clean, `db/` models+repo,
`api/` FastAPI app, `cli.py`), `migrations/` Alembic, `tests/` unit+integration.

## Resetting everything

```bash
docker compose down -v   # also removes the database volume
```
