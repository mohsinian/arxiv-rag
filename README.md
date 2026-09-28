# arxiv-rag

Ingest papers from the arXiv API, store them in a relational database, serve
aggregated data through a visualization API, and answer natural-language
questions about the dataset with a Retrieval-Augmented Generation (RAG)
pipeline.

## Status

Project scaffold — see the roadmap below as it fills in.

- [ ] Data ingestion (arXiv API, pagination, throttling, safe re-runs)
- [ ] Preprocessing (Atom/XML parsing, normalization)
- [ ] Storage (relational schema + migrations)
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
