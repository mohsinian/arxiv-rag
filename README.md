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
- Re-ingestion refreshes new and changed papers returned by the selected arXiv query window
  (the newest `--limit` papers); it is idempotent (`N new, N updated, N unchanged`), but a
  changed older paper outside that window is not rediscovered — there is no update-date
  query or full-refresh mode yet. Reindex only embeds new/changed papers (and re-embeds
  all papers after an embedding-model change).
- The API loads the embedding model during startup; the first startup downloads it (~600 MB)
  into the `hf-cache` volume, so `/health` may take a few minutes on a fresh volume.
- The API image doubles as the CLI runner — everything happens inside Docker. Both the API
  and every `docker compose run --rm api arxiv-rag …` invocation apply schema migrations
  first (image entrypoint), so CLI commands work against a fresh database too.

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
`sufficient_context`); citation numbers are validated in code against the
passages actually shown to the model, and **only the passages the model
actually cited become `sources`**. A parsed reply that cites nothing (or
reports `sufficient_context: false`) is returned as `matched: false` with no
sources — an ungrounded answer is never dressed up as a cited one. The one
exception: a reply that breaks the JSON contract entirely (plain text,
truncated JSON, non-boolean `sufficient_context`) degrades to a conservative
fallback — the raw text with all provided passages — because retrieval did
match; this is logged as a warning. Prompts wrap passages as
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

`eval/retrieval_questions.yaml` holds 30 hand-written questions in two batches:

- **Batch A** (19, assistant-authored): paraphrased from sampled papers, deliberately avoiding title wording; 1 unanswerable probe.
- **Batch B** (11, **independently authored**): written via a local authoring UI by a human reader with no knowledge of the retrieval system; 3 unanswerable probes. Gold lists are post-adjudication — where the originally-mapped paper did not actually answer the question, or several papers genuinely answer it, `expected` lists the acceptable papers.

`eval` runs retrieval only (no LLM), labels its setup, and reports hit@1/3/5, MRR, and no-match accuracy:

```bash
docker compose run --rm api arxiv-rag eval
# Retrieval setup: vector-only · Alibaba-NLP/gte-modernbert-base · top 10 · min_score 0.55
# Evaluated 30 questions (26 answerable):
#   hit@1: 73.1%   hit@3: 80.8%   hit@5: 84.6%
#   MRR: 0.774
#   no-match accuracy: 3/4 unanswerable questions rejected
```

Per batch (3,000-paper index, vector-only, top 10):

| Batch | hit@1 | hit@3 | hit@5 | MRR | Notes |
|---|---|---|---|---|---|
| A — assistant-authored (18 answerable) | 83.3% | 88.9% | 88.9% | 0.861 | author knew the retrieval design |
| B — independent, **strict** user golds (8 answerable) | 0% | 0% | 0% | 0.079 | hits: 3/8 within top 10 |
| B — independent, **post-adjudication** golds | 50.0% | 75.0% | 75.0% | 0.579 | 7/8 within top 10 |
| Combined (26 answerable) | 73.1% | 80.8% | 84.6% | 0.774 | reported by `eval` |

How to read this honestly:

- **Independent questions are much harder** than assistant-authored ones (3/8 vs 15/18 on strict golds). Manual adjudication of the 5 misses found only 1 genuine retrieval failure; 3 were mapping errors (the question's paper didn't actually answer it, while retrieval's top results did), and 1 had several valid papers with the gold outside the top 10. Authoring bias cuts both ways — an eval needs independent *questions* *and* validated *golds*.
- The 1 remaining Batch-A recall miss plus the "type-safe LLM" miss are the query-rewriting gap: the gold papers sit below rank 10 even at top-30 (e.g. the post-training-recipe gold scores 0.72 against its question while related papers score 0.78+).
- **No-match accuracy**: 3/4 unanswerable questions are correctly rejected at `MIN_SCORE=0.55`; one probe scored 0.58 and slips through — the threshold is a tunable compromise, and this metric exists to expose exactly that.
- This is a sanity harness, not a benchmark claim: 26 answerable questions means one question ≈ 4 points, and confidence intervals are wide.

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
   vectors; needed at scale. If `EMBEDDING_MODEL` changes, run `reindex --rebuild`
   (or regular `reindex`, which detects the stored model name) before serving queries.
8. The eval set is intentionally small (sanity harness, not a benchmark;
   one question ≈ 4 points). Batch B is independently authored, but golds
   were adjudicated by the assistant — a fully blind labeling pass would
   remove the last bit of authoring asymmetry. MIN_SCORE was tuned on a
   handful of unanswerable probes; one of four still slips through.
9. Citation validation is mechanical (every cited number must be a provided
   passage); whether cited passages actually entail each answer claim is not
   automatically checked.
10. **Ingestion refreshes only the current query window** — the newest
   `--limit` papers by submission date. A changed older paper outside that
   window is never rediscovered; an update-date query (`sortBy=updatedDate`)
   or full-refresh mode would close the gap.
11. **The API eagerly loads the embedding model at startup** (fail-fast:
   a broken model download stops the boot rather than surfacing at request
   time). The tradeoff is that `/health` and the stats endpoints are coupled
   to model availability and a fresh volume takes minutes before serving. A
   lazy or background loader would decouple them — at the cost of a slow
   first `/ask` and late discovery of a bad download, which is why the
   eager load was kept for this deployment.
