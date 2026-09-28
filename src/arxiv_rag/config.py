"""Application settings, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage (Postgres with the pgvector extension; see docker-compose.yml)
    database_url: str = "postgresql+psycopg://arxiv_rag:arxiv_rag@localhost:5432/arxiv_rag"

    # arXiv ingestion — arXiv asks for at most 1 request per 3 seconds
    arxiv_sleep_seconds: float = 3.0
    arxiv_page_size: int = 100
    arxiv_max_retries: int = 3

    # Embeddings (local, via sentence-transformers)
    embedding_model: str = "Alibaba-NLP/gte-modernbert-base"
    embed_dim: int = 768
    embed_batch_size: int = 32

    # RAG retrieval / answering
    rag_top_k: int = 5
    min_score: float = 0.55  # tuned on eval: relevant matches score ~0.6-0.8, off-topic <0.5
    max_context_chars: int = 12000

    # LLM — any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, Ollama, ...)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""  # only /ask needs it; empty must not break startup
    llm_model: str = "gpt-4o-mini"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the environment (and .env) is read once per process."""
    return Settings()
