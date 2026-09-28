import pytest

# Every environment variable Settings reads — cleared before each test so
# machine-level config never leaks into test outcomes.
SETTINGS_ENV_VARS = (
    "DATABASE_URL",
    "ARXIV_SLEEP_SECONDS",
    "ARXIV_PAGE_SIZE",
    "ARXIV_MAX_RETRIES",
    "EMBEDDING_MODEL",
    "EMBED_DIM",
    "EMBED_BATCH_SIZE",
    "RAG_TOP_K",
    "MIN_SCORE",
    "MAX_CONTEXT_CHARS",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def isolate_settings_env(monkeypatch):
    for var in SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sqlite_session():
    """Fresh in-memory database with all tables created; no cleanup needed."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import Session

    from arxiv_rag.db.models import Base

    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()
