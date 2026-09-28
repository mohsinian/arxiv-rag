from pathlib import Path

import pytest

from arxiv_rag.config import Settings

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[2]


def make_settings(**overrides):
    # _env_file=None: never let a developer's local .env leak into tests.
    return Settings(_env_file=None, **overrides)


def test_defaults_match_the_plan():
    s = make_settings()
    assert s.database_url == "postgresql+psycopg://arxiv_rag:arxiv_rag@localhost:5432/arxiv_rag"
    assert s.arxiv_sleep_seconds == 3.0  # arXiv asks for at most 1 request per 3 seconds
    assert s.arxiv_page_size == 100
    assert s.arxiv_max_retries == 3
    assert s.embedding_model == "Alibaba-NLP/gte-modernbert-base"
    assert s.embed_dim == 768
    assert s.embed_batch_size == 32
    assert s.rag_top_k == 5
    assert s.min_score == 0.25
    assert s.max_context_chars == 12000
    assert s.llm_base_url == "https://api.openai.com/v1"
    assert s.llm_api_key == ""  # only /ask needs it; must not break startup
    assert s.llm_model == "gpt-4o-mini"
    assert s.log_level == "INFO"


def test_env_overrides_are_parsed_and_applied(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/other")
    monkeypatch.setenv("ARXIV_SLEEP_SECONDS", "5")
    monkeypatch.setenv("EMBED_DIM", "1024")
    monkeypatch.setenv("MIN_SCORE", "0.4")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    s = make_settings()
    assert s.database_url == "postgresql+psycopg://u:p@db:5432/other"
    assert s.arxiv_sleep_seconds == 5.0
    assert s.embed_dim == 1024
    assert s.min_score == 0.4
    assert s.llm_api_key == "sk-test"


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("SOMETHING_ELSE", "1")
    s = make_settings()
    assert s.arxiv_sleep_seconds == 3.0


def test_env_example_documents_every_setting():
    env_example = (REPO_ROOT / ".env.example").read_text()
    for field in Settings.model_fields:
        assert field.upper() in env_example, f".env.example is missing {field}"
