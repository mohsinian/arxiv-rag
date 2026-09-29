"""Command-line interface: ingest, wipe, reindex, eval."""

import logging
from pathlib import Path

import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from arxiv_rag.config import get_settings
from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.client import ArxivClient
from arxiv_rag.ingest.pipeline import IngestPipeline
from arxiv_rag.rag.embeddings import build_embedder
from arxiv_rag.rag.evaluation import load_questions, run_eval
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
) -> None:
    """Fetch papers from the arXiv API and upsert them into the database.

    Safe to re-run: unchanged papers are skipped, changed ones are updated,
    and nothing is ever duplicated.
    """
    parsed = [c.strip() for c in categories.split(",") if c.strip()]
    settings = get_settings()
    session = _open_session()
    try:
        pipeline = IngestPipeline(ArxivClient(settings), Repository(session))
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
        embedder = build_embedder(settings)
        retriever = PaperRetriever(
            Repository(session), embedder, top_k=top_k, min_score=settings.min_score
        )
        report = run_eval(questions, lambda q: [s.arxiv_id for s in retriever.retrieve(q).sources])
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
