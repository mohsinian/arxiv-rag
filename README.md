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
cp .env.example .env   # then set LLM_API_KEY (any OpenAI-compatible provider)
make up                # build + start db (Postgres 16 + pgvector) and the api
make smoke             # → {"status":"ok","db":"ok"}
```

Then ingest, index, and ask:

```bash
make ingest            # newest 1,000 papers across cs.AI ∪ cs.LG ∪ cs.CL (~1 min)
make reindex           # embed them (CPU, ~20 min; only new/changed papers after that)
make ask QUESTION="What is CARGO?"
```

Every target wraps the raw `docker compose` / CLI command — `make help` lists
them all, and the [CLI section](#cli-arxiv-rag) shows the underlying commands
(without make) if you prefer to type them longhand.

Notes:
- arXiv requests are throttled to one per 3 seconds; ingesting 1,000 papers took
  under a minute in our runs (ten pages of 100). Reindexing them on CPU took ~20 minutes.
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
{"papers": 1031, "authors": 5072, "categories": 79, "avg_authors_per_paper": 5.39}
```

### `POST /api/v1/ask`

A grounded, cited answer (real response; `[n]` in the answer refers to
`sources[n - 1]`):

```bash
curl -X POST localhost:8000/api/v1/ask -H 'Content-Type: application/json' \
     -d '{"question":"How can hallucination detection use the topology of a model internal context?"}'
```

```json
{
  "answer": "According to the retrieved passages, hallucination detection can analyze the topology of information-flow patterns in attention graphs: by computing Forman-Ricci curvature to identify structural patterns indicating information bottlenecks, and capturing semi-local and global information-flow characteristics of attention heads in a single pass. This distinguishes hallucinated from non-hallucinated responses because hallucinations are associated with impaired context sharing—over-reliance on self-attention, diffused context retrieval from earlier tokens, or information over-squashing, especially in the final transformer layer [1].",
  "matched": true,
  "sources": [
    {"arxiv_id": "2609.21096", "title": "Detecting Hallucination in LLMs: Tracing the Topological Signatures of Impaired Context Sharing", "score": 0.8307}
  ]
}
```

And an honest no-match — nothing clears `MIN_SCORE`, so the LLM is never
called and the response is instant:

```bash
curl -X POST localhost:8000/api/v1/ask -H 'Content-Type: application/json' \
     -d '{"question":"What is the best sourdough bread recipe?"}'
```

```json
{
  "answer": "I could not find relevant papers in the ingested dataset for that question, so I would rather not guess. Try rephrasing, or ingest more papers on the topic.",
  "matched": false,
  "sources": []
}
```

> The examples above were generated with `z-ai/glm-5.3-flash` via xkiro
> (`LLM_BASE_URL=https://api.xkiro.com/v1`). The default configuration is
> `gpt-4o-mini` against `api.openai.com` — any OpenAI-compatible endpoint
> works, and answer phrasing varies by model.

The LLM must return a structured reply (`answer`, `cited_passages`,
`sufficient_context`); citation numbers are validated in code against the
passages actually shown to the model, and **only the passages the model
actually cited become `sources`** — markers are renumbered so `[n]` always
means `sources[n - 1]`. A reply that reports `sufficient_context: false`
keeps its answer and cited sources but returns `matched: false` (partial
coverage); a reply that cites nothing valid is a full no-match with no
sources — an ungrounded answer is never dressed up as a cited one. The one
exception: a reply that breaks the JSON contract entirely (plain text,
truncated JSON, non-boolean `sufficient_context`, non-array `cited_passages`)
degrades to a conservative fallback — the raw text with the passages shown
to the model, with any marker that cannot resolve against them removed —
because retrieval did match; this is logged as a warning. Prompts wrap passages as
untrusted XML with metadata (authors, date, category) and instruct the model
that it sees a *sample* of the dataset — no corpus-level claims, no following
instructions hidden in abstracts, partial answers allowed. When retrieval finds
nothing above `MIN_SCORE`, the LLM is never called: `matched` is `false`,
`sources` is empty, and `answer` says so. Blank/oversized questions return 422;
LLM provider failures return 502 — never a fabricated answer.

## CLI (`arxiv-rag`)

Every operation is also a CLI subcommand, run from the same image — each
invocation applies schema migrations first, so no API server needs to be up:

```bash
docker compose run --rm api arxiv-rag <command>
```

| Command | What it does |
|---|---|
| `ingest --categories cs.AI,cs.LG,cs.CL --limit 1000` | fetch the newest papers in any of the categories |
| `ingest --ids 2609.30558,…` or `--ids-from eval/retrieval_questions.yaml` | fetch specific papers by id (reproducible eval corpus) |
| `reindex` (`--rebuild`) | embed new/changed papers; detects embedding-model switches |
| `ask "question"` (`--api`) | full RAG answer in the terminal — `[n]` matches the printed source list; `--api` asks the running server (warm model, seconds) instead of loading one |
| `eval` | retrieval-only eval: hit@1/3/5, MRR, no-match accuracy |
| `wipe` (`--embeddings-only`) | drop the dataset (or just the embeddings) |

`make ask` in action (asks the running API — real output, ~7 s end to end):

```text
$ make ask QUESTION="how can I get better performance out an LLM model?"
The retrieved papers suggest several approaches:

(1) use self-evolution carefully but guard against degeneration—ATRI reweights samples and
halts training when learnable information gain stays low, since naive self-training can
plateau or decline [1];

(2) fine-tune with optimization-specific rewards, as in OptiScribe, where GRPO with
solver-feedback rewards in a challenger-solver self-play setup substantially improved
performance on generating hard MILP instances [2];

Sources:
  [1] 2609.36535  When Updating Stops Being Learning: Rethinking LLM Self-Evolution via…  (0.72)
  [2] 2609.37356  Teaching LLMs to Generate Challenging MILP Instances via Solver Feedback  (0.73)
```

The Makefile wraps the common flows — `make up`, `make down`, `make smoke`,
`make test`, `make ask QUESTION="…"` (warm API) and `make ask-cli QUESTION="…"`
(standalone container) — run `make help` for the full list.

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

`eval` runs retrieval only (no LLM), labels its setup, and reports hit@1/3/5, MRR, and no-match accuracy. Gold papers are fixed arXiv ids while `ingest` pulls the newest papers, so score the eval on a corpus that contains them:

```bash
docker compose run --rm api arxiv-rag ingest --ids-from eval/retrieval_questions.yaml
docker compose run --rm api arxiv-rag reindex
docker compose run --rm api arxiv-rag eval
# Retrieval setup: vector-only · Alibaba-NLP/gte-modernbert-base · top 10 · min_score 0.55
# Evaluated 30 questions (26 answerable):
#   hit@1: 84.6%   hit@3: 88.5%   hit@5: 92.3%
#   MRR: 0.871
#   no-match accuracy: 3/4 unanswerable questions rejected
```

Questions whose gold papers are missing from the dataset are skipped and
reported (never silently scored as misses).

Per batch (1,031-paper index — newest 1,000 across `cs.AI`∪`cs.LG`∪`cs.CL`
plus the eval golds; vector-only, top 10):

| Batch | hit@1 | hit@3 | hit@5 | MRR | Notes |
|---|---|---|---|---|---|
| A — assistant-authored (18 answerable) | 94.4% | 94.4% | 94.4% | 0.951 | author knew the retrieval design |
| B — independent, post-adjudication golds (8 answerable) | 62.5% | 75.0% | 87.5% | 0.692 | 7/8 within top 5 |
| Combined (26 answerable) | 84.6% | 88.5% | 92.3% | 0.871 | reported by `eval` |

How to read this honestly:

- **Independent questions are harder** than assistant-authored ones (62.5% vs 94.4% at hit@1). During adjudication on the previous corpus, the strict pre-adjudication golds scored 0% at top 10 — manual review found most "misses" were mapping errors (the question's mapped paper didn't actually answer it), not retrieval failures. Authoring bias cuts both ways — an eval needs independent *questions* *and* validated *golds*.
- The two remaining hit@5 misses are the query-rewriting gap: the post-training-recipe gold sits at rank 9 (0.72 against a 0.78 top match), and the "type-safe LLM" gold does not appear in the top 30 at all — the question's phrasing simply doesn't embed near its paper.
- **No-match accuracy**: 3/4 unanswerable questions are correctly rejected at `MIN_SCORE=0.55`; the penalty-shootout probe scores 0.59 and slips through — the threshold is a tunable compromise, and this metric exists to expose exactly that.
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
