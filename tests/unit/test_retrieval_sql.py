import pytest
from sqlalchemy.dialects import postgresql

from arxiv_rag.rag.retrieve import similar_papers_query

pytestmark = pytest.mark.unit


def test_query_uses_pgvector_cosine_distance_with_limit():
    compiled = str(similar_papers_query(k=5).compile(dialect=postgresql.dialect()))
    assert "<=>" in compiled  # pgvector cosine distance operator
    assert "LIMIT" in compiled


def test_query_vector_is_a_bind_parameter():
    compiled = str(similar_papers_query(k=1).compile(dialect=postgresql.dialect()))
    assert "query_vector" in compiled


def test_query_joins_papers_with_their_embeddings():
    compiled = str(similar_papers_query(k=1).compile(dialect=postgresql.dialect()))
    assert "papers" in compiled
    assert "paper_embeddings" in compiled
