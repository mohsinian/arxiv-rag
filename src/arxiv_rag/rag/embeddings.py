"""Embedding providers: a protocol plus the local sentence-transformers implementation."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from arxiv_rag.config import Settings
from arxiv_rag.db.models import EMBEDDING_DIM


class EmbeddingConfigError(Exception):
    """Raised when the embedding model cannot produce vectors the schema can store."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can embed passages and queries into fixed-size vectors."""

    model_name: str
    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


def passage_text(title: str, abstract: str) -> str:
    """Canonical text for one paper's embedding: title, blank line, abstract."""
    return f"{title}\n\n{abstract}".strip()


def check_embedding_dimension(model_name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise EmbeddingConfigError(
            f"model {model_name} produces {actual}-dimensional vectors but the schema "
            f"expects {expected} (EMBED_DIM); align EMBED_DIM/migrations and reindex"
        )


class SentenceTransformerEmbedder:
    """Local embedder backed by sentence-transformers, CPU-only."""

    def __init__(self, model_name: str, expected_dim: int, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        dimension = int(self._model.get_sentence_embedding_dimension())
        check_embedding_dimension(model_name, dimension, expected_dim)
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def build_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    # The schema bakes the vector width into the column type (Vector(768)), so
    # a different EMBED_DIM would pass the model check yet fail on every insert
    if settings.embed_dim != EMBEDDING_DIM:
        raise EmbeddingConfigError(
            f"EMBED_DIM={settings.embed_dim} but the schema fixes the vector width at "
            f"{EMBEDDING_DIM}; changing it requires a new migration plus reindex, "
            "not just the environment variable"
        )
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        expected_dim=settings.embed_dim,
        batch_size=settings.embed_batch_size,
    )
