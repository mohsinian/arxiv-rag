from datetime import datetime, timezone

import pytest

from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.parser import PaperRecord
from arxiv_rag.rag.retrieve import PaperRetriever

pytestmark = pytest.mark.integration

UTC = timezone.utc


def sparse(first: float, second: float = 0.0) -> list[float]:
    vector = [0.0] * 768
    vector[0] = first
    vector[1] = second
    return vector


class FixedQueryEmbedder:
    """Always embeds the query as the given vector."""

    model_name = "fake-model"
    dimension = 768

    def __init__(self, vector):
        self._vector = vector

    def embed_query(self, query):
        return self._vector

    def embed_texts(self, texts):
        return [self._vector for _ in texts]


def rec(arxiv_id: str, title: str) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=title,
        abstract=f"Abstract for {title}.",
        authors=("Alice Researcher",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published=datetime(2024, 1, 10, tzinfo=UTC),
        updated=datetime(2024, 1, 10, tzinfo=UTC),
        doi=None,
        journal_ref=None,
    )


@pytest.fixture
def planted(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records(
        [
            rec("2401.00001", "Exact Direction Paper"),
            rec("2401.00002", "Diagonal Paper"),
            rec("2401.00003", "Orthogonal Paper"),
        ]
    )
    repo.store_embeddings(
        [
            (1, sparse(1.0), "fake"),  # cosine sim vs [1,0]: 1.0
            (2, sparse(0.7071067811865476, 0.7071067811865476), "fake"),  # ≈ 0.7071
            (3, sparse(0.0, 1.0), "fake"),  # 0.0
        ]
    )
    return repo


def test_results_are_ordered_by_cosine_similarity(planted, pg_session):
    retriever = PaperRetriever(planted, FixedQueryEmbedder(sparse(1.0)), top_k=3, min_score=0.0)
    result = retriever.retrieve("anything")
    assert [s.arxiv_id for s in result.sources] == ["2401.00001", "2401.00002", "2401.00003"]
    assert result.sources[0].score == pytest.approx(1.0)
    assert result.sources[1].score == pytest.approx(0.7071, abs=1e-3)
    assert result.sources[2].score == pytest.approx(0.0, abs=1e-6)
    assert result.matched


def test_top_k_limits_the_number_of_sources(planted):
    retriever = PaperRetriever(planted, FixedQueryEmbedder(sparse(1.0)), top_k=2, min_score=0.0)
    result = retriever.retrieve("anything")
    assert [s.arxiv_id for s in result.sources] == ["2401.00001", "2401.00002"]


def test_min_score_filters_weak_matches(planted):
    retriever = PaperRetriever(planted, FixedQueryEmbedder(sparse(1.0)), top_k=3, min_score=0.9)
    result = retriever.retrieve("anything")
    assert [s.arxiv_id for s in result.sources] == ["2401.00001"]


def test_nothing_above_threshold_is_an_honest_miss(planted):
    # query points opposite to every stored vector: all scores ≤ 0
    retriever = PaperRetriever(planted, FixedQueryEmbedder(sparse(-1.0)), top_k=3, min_score=0.25)
    result = retriever.retrieve("anything")
    assert result.sources == []
    assert not result.matched


def test_empty_index_is_an_honest_miss(pg_session):
    Repository(pg_session).upsert_records([rec("2401.00001", "Lonely Paper")])  # no embeddings
    retriever = PaperRetriever(
        Repository(pg_session), FixedQueryEmbedder(sparse(1.0)), top_k=3, min_score=0.25
    )
    result = retriever.retrieve("anything")
    assert result.sources == []
    assert not result.matched


def test_retrieved_papers_carry_metadata(planted):
    retriever = PaperRetriever(planted, FixedQueryEmbedder(sparse(1.0)), top_k=1, min_score=0.0)
    result = retriever.retrieve("anything")
    source = result.sources[0]
    assert source.published == "2024-01-10"
    assert source.category == "cs.LG"
    assert source.authors == "Alice Researcher"
