"""Application factory."""

from collections.abc import Callable

from fastapi import FastAPI
from sqlalchemy.orm import Session

import arxiv_rag
from arxiv_rag.api.routes.ask import router as ask_router
from arxiv_rag.api.routes.health import router as health_router
from arxiv_rag.api.routes.stats import router as stats_router

SessionFactory = Callable[[], Session]
AskFactory = Callable[[Session], Callable[[str], object]]


def create_app(
    session_factory: SessionFactory | None = None,
    ask_factory: AskFactory | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `session_factory` creates short-lived Sessions for request handlers
    (and /health's liveness probe). `ask_factory` maps a request-scoped
    session to the /ask handler. Tests inject their own; production wiring
    lives in `arxiv_rag.api.main`.
    """
    app = FastAPI(
        title="arxiv-rag",
        version=arxiv_rag.__version__,
        description="arXiv paper stats and RAG question answering over an ingested dataset.",
    )
    app.state.session_factory = session_factory
    app.state.ask_factory = ask_factory
    app.include_router(health_router)
    app.include_router(stats_router)
    app.include_router(ask_router)
    return app
