"""Production wiring for uvicorn: `uvicorn arxiv_rag.api.main:app`.

Loads the local embedding model once at import (a few seconds on CPU);
LLM settings come from the environment and are only used by /ask.
"""

from arxiv_rag.api.app import create_app
from arxiv_rag.config import get_settings
from arxiv_rag.db.engine import create_session_factory
from arxiv_rag.rag.ask import build_ask_handler
from arxiv_rag.rag.embeddings import build_embedder

settings = get_settings()
session_factory = create_session_factory(settings.database_url)
embedder = build_embedder(settings)
app = create_app(session_factory=session_factory, ask_factory=build_ask_handler(settings, embedder))
