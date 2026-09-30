"""Command-line interface: ingest, wipe, reindex, eval, ask."""

import json
import re
import textwrap
import urllib.request
from pathlib import Path

import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from arxiv_rag.config import get_settings
from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.client import ArxivClient
from arxiv_rag.ingest.pipeline import IngestPipeline
from arxiv_rag.logging_config import configure_logging
from arxiv_rag.rag.answer import LLMError
from arxiv_rag.rag.ask import build_ask_handler
from arxiv_rag.rag.embeddings import build_embedder
from arxiv_rag.rag.evaluation import gold_ids, load_questions, run_eval, split_by_coverage
from arxiv_rag.rag.indexer import EmbeddingIndexer
from arxiv_rag.rag.retrieve import PaperRetriever

app = typer.Typer(help="Ingest arXiv papers into Postgres and manage the dataset.")

DEFAULT_CATEGORIES = "cs.AI,cs.LG,cs.CL"
DEFAULT_QUESTIONS_PATH = "eval/retrieval_questions.yaml"


def _open_session() -> Session:
    settings = get_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    return Session(engine)


@app.command()
def ingest(
    categories: str = typer.Option(
        DEFAULT_CATEGORIES, "--categories", "-c", help="Comma-separated arXiv categories."
    ),
    limit: int = typer.Option(
        1000, "--limit", "-l", min=1, help="Maximum number of papers to fetch."
    ),
    ids: str = typer.Option(
        "", "--ids", help="Comma-separated arXiv ids to fetch instead of a category query."
    ),
    ids_from: Path | None = typer.Option(
        None,
        "--ids-from",
        help="Fetch the gold papers of an eval question set (YAML) instead of a category query.",
    ),
) -> None:
    """Fetch papers from the arXiv API and upsert them into the database.

    By default fetches the newest `--limit` papers in any of `--categories`.
    With `--ids` / `--ids-from`, fetches exactly those papers instead (e.g.
    `--ids-from eval/retrieval_questions.yaml` makes `eval` reproducible).
    Safe to re-run: unchanged papers are skipped, changed ones are updated,
    and nothing is ever duplicated.
    """
    settings = get_settings()
    try:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        if ids_from is not None:
            wanted += gold_ids(load_questions(ids_from))
    except Exception as exc:
        typer.echo(f"ingest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    session = _open_session()
    try:
        pipeline = IngestPipeline(ArxivClient(settings), Repository(session))
        if ids or ids_from is not None:
            summary = pipeline.run_ids(wanted, page_size=settings.arxiv_page_size)
        else:
            parsed = [c.strip() for c in categories.split(",") if c.strip()]
            summary = pipeline.run(parsed, limit, page_size=settings.arxiv_page_size)
    except Exception as exc:
        typer.echo(f"ingest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()
    typer.echo(f"Ingest complete: {summary}")


@app.command()
def wipe(
    embeddings_only: bool = typer.Option(
        False, "--embeddings-only", help="Only clear embeddings (papers are kept)."
    ),
) -> None:
    """Wipe the dataset so a fresh import can be run from scratch."""
    session = _open_session()
    try:
        Repository(session).wipe(embeddings_only=embeddings_only)
    except Exception as exc:
        typer.echo(f"wipe failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()
    what = (
        "Embeddings wiped"
        if embeddings_only
        else "Dataset wiped (papers, authors, categories, embeddings)"
    )
    typer.echo(what)


@app.command()
def reindex(
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Drop all embeddings first, then re-embed everything."
    ),
) -> None:
    """Generate embeddings for papers that are new or changed since indexing."""
    settings = get_settings()
    session = _open_session()
    try:
        indexer = EmbeddingIndexer(
            Repository(session), build_embedder(settings), expected_dim=settings.embed_dim
        )
        summary = indexer.reindex(rebuild=rebuild)
    except Exception as exc:
        typer.echo(f"reindex failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()
    typer.echo(f"Reindex complete: {summary}")


@app.command()
def eval(
    question_set: Path = typer.Argument(
        DEFAULT_QUESTIONS_PATH, exists=False, help="YAML file with questions + expected arxiv_ids."
    ),
    top_k: int = typer.Option(
        10, "--top-k", min=1, help="How many papers to retrieve per question."
    ),
) -> None:
    """Run the retrieval evaluation: hit@1/3/5, MRR, no-match accuracy (no LLM)."""
    settings = get_settings()
    session = _open_session()
    try:
        questions = load_questions(Path(question_set))
        repo = Repository(session)
        indexed = repo.existing_arxiv_ids(gold_ids(questions))
        scorable, skipped = split_by_coverage(questions, indexed)
        embedder = build_embedder(settings)
        retriever = PaperRetriever(repo, embedder, top_k=top_k, min_score=settings.min_score)
        report = run_eval(scorable, lambda q: [s.arxiv_id for s in retriever.retrieve(q).sources])
    except Exception as exc:
        typer.echo(f"eval failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()

    answerable = len(report.answerable)
    typer.echo(
        f"Retrieval setup: vector-only · {settings.embedding_model} · "
        f"top {top_k} · min_score {settings.min_score}"
    )
    if skipped:
        typer.echo(
            f"Skipped {len(skipped)} question(s) whose gold papers are not in the dataset — "
            f"run `arxiv-rag ingest --ids-from {question_set}` to score them:"
        )
        for question in skipped:
            typer.echo(f"  - {question.question}  (gold: {', '.join(question.expected)})")
    typer.echo(f"Evaluated {len(report.results)} questions ({answerable} answerable):")
    for k in report.ks:
        typer.echo(f"  hit@{k}: {report.mean_hit_at(k):.1%}")
    typer.echo(f"  MRR: {report.mrr():.3f}")
    correct, total = report.no_match_stats()
    if total:
        typer.echo(f"  no-match accuracy: {correct}/{total} unanswerable questions rejected")
    if misses := report.misses(k=report.ks[-1]):
        typer.echo("Misses:")
        for miss in misses:
            typer.echo(
                f"  - {miss.question.question}  (expected: {', '.join(miss.question.expected)})"
            )


def _render_answer(answer: str, matched: bool, sources: list[tuple[str, str, float]]) -> None:
    """Print an ask result readably: enumerated findings get their own
    paragraph, everything wraps at 90 columns, and sources are numbered to
    match the answer's [n] markers."""
    items = [item for item in re.split(r"(?=\(\d{1,2}\)\s)", answer) if item.strip()]
    if len(items) > 1:
        typer.echo("\n\n".join(textwrap.fill(item.strip(), width=90) for item in items))
    else:
        typer.echo(textwrap.fill(answer.strip(), width=90))
    if not matched and sources:
        typer.echo("\n(partial answer: the retrieved papers do not fully cover the question)")
    if sources:
        typer.echo("\nSources:")
        for number, (arxiv_id, title, score) in enumerate(sources, 1):
            title = title if len(title) <= 72 else title[:69] + "…"
            typer.echo(f"  [{number}] {arxiv_id}  {title}  ({score:.2f})")


def _ask_via_api(question: str, base_url: str = "http://127.0.0.1:8000") -> dict:
    """POST the question to the running API — reuses its warm embedding model."""
    request = urllib.request.Request(
        f"{base_url}/api/v1/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer from the ingested papers."),
    api: bool = typer.Option(
        False, "--api", help="Ask the running API server instead of loading the model here."
    ),
) -> None:
    """Answer a question with the full RAG pipeline.

    By default loads the embedding model in this process (works standalone).
    With --api, sends the question to the running API instead — the model
    stays warm there, so answers come back in seconds with no load chatter.
    Either way, `[n]` markers in the answer match the numbered source list
    printed below it, exactly like the API response.
    """
    if api:
        try:
            result = _ask_via_api(question)
        except Exception as exc:
            typer.echo(
                f"ask failed: could not reach the API ({exc}); run `docker compose up -d` "
                "first or drop --api to load the model locally",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        _render_answer(
            result["answer"],
            result["matched"],
            [(s["arxiv_id"], s["title"], s["score"]) for s in result["sources"]],
        )
        return

    settings = get_settings()
    session = _open_session()
    try:
        handler = build_ask_handler(settings, build_embedder(settings))(Repository(session))
        answered = handler(question)
    except LLMError as exc:
        typer.echo(f"ask failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()
    _render_answer(
        answered.answer,
        answered.matched,
        [(s.arxiv_id, s.title, s.score) for s in answered.sources],
    )


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()
