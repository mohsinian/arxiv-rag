# arxiv-rag

Ingest papers from the arXiv API, store them in a relational database, serve
aggregated data through a visualization API, and answer natural-language
questions about the dataset with a Retrieval-Augmented Generation (RAG)
pipeline.

## Status

Project scaffold — see the roadmap below as it fills in.

- [x] Data ingestion (arXiv API, pagination, throttling, safe re-runs)
- [x] Preprocessing (Atom/XML parsing, normalization)
- [x] Storage (relational schema + migrations)
- [ ] Visualization API (aggregations for charting)
- [ ] RAG pipeline (embeddings, retrieval, `/ask` endpoint)
- [ ] docker-compose, `.env.example`, tests

## Quick start (dev)

Requires Python ≥3.10 and Docker (with WSL integration enabled if on WSL2).

```bash
cp .env.example .env          # adjust if you like; defaults just work
make up-db                    # Postgres 16 + pgvector on localhost:5432 (healthy when done)
make test                     # unit tests — no Docker required
```

The database container auto-enables the `vector` extension on first start
(`db/init/001_enable_vector.sql`). The API service joins `docker-compose.yml`
as soon as there is an app to run.

## Ingesting data

With the database up (`make up-db`) and migrations applied
(`alembic upgrade head`):

```bash
.venv/bin/arxiv-rag ingest --categories cs.AI,cs.LG,cs.CL --limit 1000
```

- Requests are throttled to one per 3 seconds (arXiv's etiquette guideline).
- Re-running is safe: unchanged papers are skipped, changed ones updated —
  the summary line reports `N new, N updated, N unchanged`.
- A crash mid-ingest keeps every already-fetched page; re-running continues.
- `python -m arxiv_rag ingest …` works the same way.

## Resetting the dataset

```bash
.venv/bin/arxiv-rag wipe                # papers, authors, categories, embeddings
.venv/bin/arxiv-rag wipe --embeddings-only   # keep papers, drop the RAG index
```
