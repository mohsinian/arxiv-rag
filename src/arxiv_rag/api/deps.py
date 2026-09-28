"""Shared FastAPI dependencies."""

from collections.abc import Iterator

from fastapi import HTTPException, Request

from arxiv_rag.db.repo import Repository


def get_repo(request: Request) -> Iterator[Repository]:
    """One repository (backed by its own session) per request."""
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise HTTPException(status_code=503, detail="database not configured")
    session = session_factory()
    try:
        yield Repository(session)
    finally:
        session.close()
