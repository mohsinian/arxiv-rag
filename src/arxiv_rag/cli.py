"""Command-line interface: ingest, wipe, reindex (eval arrives with Phase 7)."""

import logging

import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from arxiv_rag.config import get_settings
from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.client import ArxivClient
from arxiv_rag.ingest.pipeline import IngestPipeline
from arxiv_rag.rag.embeddings import build_embedder
from arxiv_rag.rag.indexer import EmbeddingIndexer

app = typer.Typer(help="Ingest arXiv papers into Postgres and manage the dataset.")

DEFAULT_CATEGORIES = "cs.AI,cs.LG,cs.CL"


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
