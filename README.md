# arxiv-rag

Ingest papers from the arXiv API, store them in a relational database, serve
aggregated data through a visualization API, and answer natural-language
questions about the dataset with a Retrieval-Augmented Generation (RAG)
pipeline.

```
arXiv API ──► ingest (parse → clean → upsert) ──► PostgreSQL + pgvector
                                                    │            │
                                       stats API (SQL aggregations) │
                                                    │   embeddings (gte-modernbert-base, local)
                                                    ▼            ▼
                                          GET /api/v1/…   POST /api/v1/ask
                                                          retrieve top-k → LLM answer + sources
```

## Quick start (Docker, ~5 minutes)

Requires only Docker and a `.env`. No host Python.

```bash
cp .env.example .env            # then set LLM_API_KEY (any OpenAI-compatible provider)
docker compose up -d --build    # db (Postgres 16 + pgvector) + api (migrates on boot)
curl http://localhost:8000/health
# {"status":"ok","db":"ok"}
```

Then ingest, index, and ask:

```bash
docker compose run --rm api arxiv-rag ingest --categories cs.AI,cs.LG,cs.CL --limit 1000
docker compose run --rm api arxiv-rag reindex
curl -X POST localhost:8000/api/v1/ask -H 'Content-Type: application/json' \
     -d '{"question":"What is CARGO?"}'
```

Notes:
- arXiv requests are throttled to one per 3 seconds; ingesting 1,000 papers takes ~6 minutes.
- Re-running ingest is idempotent (`N new, N updated, N unchanged`); reindex only embeds
  new/changed papers.
- The first `reindex` downloads the embedding model (~600 MB) into the `hf-cache` volume.
- The API image doubles as the CLI runner — everything happens inside Docker.

## API

All endpoints are served by the `api` container (`API_PORT`, default 8000).
Interactive docs: `http://localhost:8000/docs`.

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "db": "ok"}
```

### `GET /api/v1/stats/papers-by-month?category=cs.AI&months=12`

Paper counts per publication month (computed in the database). `category` and `months` are optional.

```json
[{"month": "2026-09", "count": 100}]
```

### `GET /api/v1/stats/top-categories?limit=5`

```json
[
  {"category": "cs.AI", "count": 100},
  {"category": "cs.CL", "count": 100},
  {"category": "cs.LG", "count": 100},
  {"category": "cs.CV", "count": 7},
  {"category": "cs.IR", "count": 4}
]
```

### `GET /api/v1/stats/top-authors?limit=3`

```json
[
  {"author": "Binfeng Xu", "count": 2},
  {"author": "David Ababio Awuni", "count": 2},
  {"author": "Elvis Gyasi Owusu", "count": 2}
]
```

### `GET /api/v1/stats/summary`

```json
{"papers": 100, "authors": 488, "categories": 18, "avg_authors_per_paper": 5.03}
```

### `POST /api/v1/ask`

```bash
curl -X POST localhost:8000/api/v1/ask -H 'Content-Type: application/json' \
     -d '{"question":"How do AI agents use memory to stay stable while learning new tasks?"}'
```

```json
{
  "answer": "The passages don't directly answer this question. The closest is [1], which studies the stability-plasticity tradeoff in agent memory: ...",
  "matched": true,
  "sources": [
    {"arxiv_id": "2609.30558", "title": "Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms", "score": 0.7677},
    {"arxiv_id": "2609.29233", "title": "Post-Training Leaves Behavioral Shadows on Unrelated Decisions", "score": 0.6564}
  ]
}
```

The LLM must return a structured reply (`answer`, `cited_passages`,
`sufficient_context`); citation numbers are validated in code and **only the
passages the model actually cited become `sources`**. Prompts wrap passages as
untrusted XML with metadata (authors, date, category) and instruct the model
that it sees a *sample* of the dataset — no corpus-level claims, no following
instructions hidden in abstracts, partial answers allowed. When retrieval finds
nothing above `MIN_SCORE`, the LLM is never called: `matched` is `false`,
`sources` is empty, and `answer` says so. Blank/oversized questions return 422;
LLM provider failures return 502 — never a fabricated answer.

## Schema

```
papers(id, arxiv_id UNIQUE, title, abstract, published_at, updated_at,
       doi NULL, journal_ref NULL, primary_category_id → categories)
authors(id, name UNIQUE)
categories(id, arxiv_category UNIQUE)
paper_authors(paper_id, author_id, position)     -- author order preserved
paper_categories(paper_id, category_id)          -- cross-listings
paper_embeddings(paper_id, VECTOR(768), model, embedded_at)   -- RAG index, separable
```

- One paper ↔ many authors/categories via join tables; the primary category is a FK.
- Embeddings live in their own table so the RAG index can be rebuilt or wiped
  (`wipe --embeddings-only`) without touching papers.
- Schema is reproducible from scratch: `alembic upgrade head` (the migration
  also enables the `vector` extension).

## Retrieval quality (hit@k)

`eval/retrieval_questions.yaml` holds 19 hand-written questions with the
arxiv_ids a good system should retrieve — deliberately paraphrased, not title
copies, plus one unanswerable question exposing the similarity floor. The
`eval` command runs retrieval only (no LLM), labels its own setup, and reports
hit@1/3/5, MRR, and no-match accuracy:

```bash
docker compose run --rm api arxiv-rag eval
# Retrieval setup: vector-only · Alibaba-NLP/gte-modernbert-base · top 10 · min_score 0.55
# Evaluated 19 questions (18 answerable):
#   hit@1: 83.3%   hit@3: 88.9%   hit@5: 88.9%
#   MRR: 0.861
#   no-match accuracy: 1/1 unanswerable questions rejected
```

How to read it (both indexes, vector-only retrieval):

| Index (cs.AI+cs.LG+cs.CL) | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| 1,000 papers (default ingest) | 88.9% | 88.9% | 88.9% | — |
| 3,000 papers | 83.3% | 88.9% | 88.9% | 0.861 |

(hit rates over 18 answerable questions; the 1,000-paper row was measured
before MRR was added.) Tripling the corpus cost one rank-1 position — a gold
paper pushed to rank 2 by a new competing paper — while recall held: the
retrieved-set quality is stable, which is the reranker's exact use case. The
two persistent misses are genuine recall failures, not ranking ones: at
top-30 the gold papers still don't appear, and by-hand analysis shows the
questions' phrasing matching other papers more closely than the gold (e.g.
the gold "post-training recipe" paper scores 0.72 against its question while
related post-training papers score 0.78+). That is the query-rewriting gap,
not a reranking gap. `MIN_SCORE=0.55` was tuned on the score separation
between real matches (0.63–0.81) and the unanswerable question's best match
(0.50), which is what the no-match accuracy row verifies. This is a sanity
harness — not a benchmark claim (19 questions means one question ≈ 5 points;
confidence intervals are wide).

## Environment variables

See [.env.example](.env.example) for the full list with comments. Highlights:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL (compose sets it inside the network) |
| `ARXIV_SLEEP_SECONDS` | Throttle (arXiv etiquette: ≥3s) |
| `EMBEDDING_MODEL` / `EMBED_DIM` | Local embedding model and its dimension |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Any OpenAI-compatible chat endpoint (OpenAI, xkiro, OpenRouter, …) |
| `RAG_TOP_K` / `MIN_SCORE` | Retrieval breadth and the similarity floor for a "match" |
| `POSTGRES_PORT` / `API_PORT` | Host ports (bump if 5432/8000 clash) |

## Development (host)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env && make up-db    # database only
make test                             # unit tests (no Docker needed)
make test-integration                 # needs `make up-db`
make lint                             # ruff lint + format check
make ui                               # Adminer web DB UI on :8081
make db-shell                         # pgcli terminal shell
```

Code layout: `src/arxiv_rag/` — `ingest/` (client, parser, preprocessing,
pipeline), `db/` (models, repo), `rag/` (embeddings, indexer, retrieve,
answer, evaluation), `api/` (FastAPI app), `cli.py`; `migrations/` Alembic;
`tests/` unit + integration; `eval/` retrieval question set.

## Known limitations & next steps

1. **Reranking with a cross-encoder** (e.g. bge-reranker) over the top-50
   candidates — the single biggest quality win available before touching anything else.
2. **Query rewriting** before retrieval (expand/clarify the question with the LLM
   before embedding it).
3. **Chunking full PDFs** — only title+abstract is embedded; full-text chunking
   with per-chunk vectors and paper-level aggregation would cover papers whose
   abstracts don't carry the answer.
4. **Auth and rate limiting on `/ask`** — API keys + per-key quotas before any
   public exposure; `/ask` is unauthenticated.
5. **Caching** — embedding cache keyed on text hash; short-TTL answer cache for
   repeated questions; HTTP caching for stats endpoints.
6. **Author-name dedup** is exact-match; two different researchers sharing a name
   collapse into one row (no ORCID in the arXiv API).
7. **No ANN index** (HNSW) on the embedding column — exact scan is fine at ≤5k
   vectors; needed at scale.
8. The eval set is intentionally small (sanity harness, not a benchmark;
   one question ≈ 5 points, and questions were authored after reading the
   papers — an authoring-bias caveat until the set grows and gains
   independently-written questions).
9. Citation validation is mechanical (every cited number must be a provided
   passage); whether cited passages actually entail each answer claim is not
   automatically checked.
