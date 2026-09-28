from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select, text, update

from arxiv_rag.db import models
from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.parser import PaperRecord
from arxiv_rag.rag.embeddings import EmbeddingConfigError
from arxiv_rag.rag.indexer import EmbeddingIndexer

pytestmark = pytest.mark.integration

UTC = timezone.utc


def rec(arxiv_id: str) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract=f"Abstract about topic {arxiv_id}.",
        authors=("Alice Researcher",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published=datetime(2024, 1, 10, tzinfo=UTC),
        updated=datetime(2024, 1, 10, tzinfo=UTC),
        doi=None,
        journal_ref=None,
    )


class FakeEmbedder:
    model_name = "fake-model"
    dimension = 768

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] + [0.0] * 767 for t in texts]

    def embed_query(self, query):
        return self.embed_texts([query])[0]


@pytest.fixture
def seeded(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records([rec(f"2401.0000{i}") for i in (1, 2, 3)])
    return repo


def embedding_count(session) -> int:
    return session.scalar(select(func.count()).select_from(models.PaperEmbedding))


def test_reindex_embeds_all_papers_and_round_trips_vectors(seeded, pg_session):
    embedder = FakeEmbedder()
    summary = EmbeddingIndexer(seeded, embedder, expected_dim=768).reindex()

    assert summary.embedded == 3
    assert embedding_count(pg_session) == 3
    stored = pg_session.execute(select(models.PaperEmbedding)).scalars().all()
    assert all(row.model == "fake-model" for row in stored)
    expected_first = float(len("Title 2401.00001\n\nAbstract about topic 2401.00001."))
    assert list(stored[0].embedding) == pytest.approx([expected_first] + [0.0] * 767)


def test_reindex_is_a_noop_when_everything_is_fresh(seeded, pg_session):
    embedder = FakeEmbedder()
    indexer = EmbeddingIndexer(seeded, embedder, expected_dim=768)
    indexer.reindex()

    summary = indexer.reindex()

    assert summary.embedded == 0
    assert len(embedder.calls) == 1  # only the first pass embedded anything
    assert embedding_count(pg_session) == 3


def test_changed_paper_is_re_embedded(seeded, pg_session):
    EmbeddingIndexer(seeded, FakeEmbedder(), expected_dim=768).reindex()

    seeded.upsert_records([_with_abstract(rec("2401.00002"), "A revised abstract.")])
    embedder = FakeEmbedder()
    summary = EmbeddingIndexer(seeded, embedder, expected_dim=768).reindex()

    assert summary.embedded == 1  # only the changed paper
    assert embedder.calls[0] == ["Title 2401.00002\n\nA revised abstract."]


def _with_abstract(record, abstract):
    from dataclasses import replace

    return replace(record, abstract=abstract, updated=datetime(2025, 6, 1, tzinfo=UTC))


def test_paper_updated_after_embedding_is_stale(seeded, pg_session):
    EmbeddingIndexer(seeded, FakeEmbedder(), expected_dim=768).reindex()
    pg_session.execute(
        update(models.Paper)
        .where(models.Paper.arxiv_id == "2401.00001")
        .values(updated_at=func.now() + text("interval '1 hour'"))
    )
    pg_session.commit()

    summary = EmbeddingIndexer(seeded, FakeEmbedder(), expected_dim=768).reindex()

    assert summary.embedded == 1


def test_rebuild_wipes_and_re_embeds_everything(seeded, pg_session):
    EmbeddingIndexer(seeded, FakeEmbedder(), expected_dim=768).reindex()

    summary = EmbeddingIndexer(seeded, FakeEmbedder(), expected_dim=768).reindex(rebuild=True)

    assert summary.embedded == 3
    assert embedding_count(pg_session) == 3


def test_dimension_mismatch_is_rejected_before_touching_the_db(seeded, pg_session):
    class WrongDimEmbedder(FakeEmbedder):
        dimension = 999

    with pytest.raises(EmbeddingConfigError, match="999"):
        EmbeddingIndexer(seeded, WrongDimEmbedder(), expected_dim=768)
    assert embedding_count(pg_session) == 0
