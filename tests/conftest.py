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
