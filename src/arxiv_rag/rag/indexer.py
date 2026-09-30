"""Embedding index maintenance: embed papers whose text is new or changed."""

import logging
from dataclasses import dataclass
from itertools import islice

from arxiv_rag.db.repo import Repository
from arxiv_rag.rag.embeddings import (
    EmbeddingProvider,
    check_embedding_dimension,
    passage_text,
)

logger = logging.getLogger(__name__)


@dataclass
class ReindexSummary:
    embedded: int = 0

    def __str__(self) -> str:
        return f"{self.embedded} embedded"


class EmbeddingIndexer:
    """Keeps `paper_embeddings` in sync with paper text.

    A paper is stale when it has no embedding row, or when its `updated_at`
    is newer than the row's `embedded_at` (e.g. arXiv served a new version).
    Upserts already delete embeddings of changed papers, so in practice the
    staleness check mainly catches out-of-band edits — belt and suspenders.
    """

    def __init__(self, repo: Repository, embedder: EmbeddingProvider, expected_dim: int) -> None:
        check_embedding_dimension(embedder.model_name, embedder.dimension, expected_dim)
        self._repo = repo
        self._embedder = embedder

    def reindex(self, rebuild: bool = False) -> ReindexSummary:
        """Embed stale papers in batches; `rebuild` drops the index first."""
        if rebuild:
            self._repo.wipe(embeddings_only=True)
            logger.info("embedding index cleared for rebuild")

        summary = ReindexSummary()
        papers = self._repo.papers_needing_embedding(self._embedder.model_name)
        while True:
            batch = list(islice(papers, 64))
            if not batch:
                break
            vectors = self._embedder.embed_texts(
                [passage_text(title, abstract) for _id, title, abstract in batch]
            )
            self._repo.store_embeddings(
                [
                    (paper_id, vector, self._embedder.model_name)
                    for (paper_id, _t, _a), vector in zip(batch, vectors, strict=True)
                ]
            )
            summary.embedded += len(batch)
            logger.info("reindex progress: %s", summary)
        logger.info("reindex finished: %s", summary)
        return summary
