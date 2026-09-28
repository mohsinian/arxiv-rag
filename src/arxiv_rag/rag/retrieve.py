"""Top-k semantic retrieval over the paper embeddings (pgvector cosine distance)."""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import Select, bindparam, select
from sqlalchemy.orm import Session

from arxiv_rag.db import models
from arxiv_rag.db.repo import Repository


class QueryEmbedder(Protocol):
    model_name: str
    dimension: int

    def embed_query(self, query: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievedPaper:
    arxiv_id: str
    title: str
    abstract: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    sources: list[RetrievedPaper]
    matched: bool


def similar_papers_query(k: int) -> Select:
    """Papers joined with their embeddings, nearest first (cosine distance).

    The query vector is a bind parameter supplied at execution time
    (`{"query_vector": [...]}`), so the SQL text is testable without a DB.
    """
    return (
        select(
            models.Paper.arxiv_id,
            models.Paper.title,
            models.Paper.abstract,
            (1 - models.PaperEmbedding.embedding.cosine_distance(bindparam("query_vector"))).label(
                "score"
            ),
        )
        .join(models.PaperEmbedding, models.PaperEmbedding.paper_id == models.Paper.id)
        .order_by(models.PaperEmbedding.embedding.cosine_distance(bindparam("query_vector")).asc())
        .limit(k)
    )


class PaperRetriever:
    """Embeds a question and returns the most similar papers with similarity scores."""

    def __init__(
        self,
        repo: Repository,
        embedder: QueryEmbedder,
        top_k: int,
        min_score: float,
        session: Session | None = None,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score
        self._session = session

    def retrieve(self, question: str) -> RetrievalResult:
        vector = self._embedder.embed_query(question)
        rows = self._repo.execute_similar_papers(similar_papers_query(k=self._top_k), vector)
        sources = [
            RetrievedPaper(
                arxiv_id=row.arxiv_id,
                title=row.title,
                abstract=row.abstract,
                score=round(row.score, 4),
            )
            for row in rows
        ]
        sources = [source for source in sources if source.score >= self._min_score]
        return RetrievalResult(sources=sources, matched=bool(sources))
