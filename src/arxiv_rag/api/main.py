"""Production wiring for uvicorn: `uvicorn arxiv_rag.api.main:app`."""

from arxiv_rag.api.app import create_app
from arxiv_rag.config import get_settings
from arxiv_rag.db.engine import create_session_factory

settings = get_settings()
session_factory = create_session_factory(settings.database_url)
app = create_app(session_factory=session_factory)
