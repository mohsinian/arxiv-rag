from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from arxiv_rag.db import models
from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.parser import PaperRecord

pytestmark = pytest.mark.integration

ALL_TABLES = (
    models.Paper,
    models.Author,
    models.Category,
    models.paper_authors,
    models.paper_categories,
    models.PaperEmbedding,
)


def record(arxiv_id: str) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title="A Title",
        abstract="An abstract.",
        authors=("Alice Researcher",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published=datetime(2024, 1, 15, tzinfo=timezone.utc),
        updated=datetime(2024, 2, 1, tzinfo=timezone.utc),
        doi=None,
        journal_ref=None,
    )


def count(session, table) -> int:
    return session.scalar(select(func.count()).select_from(table))


@pytest.fixture
def seeded_session(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records([record("2401.00001"), record("2401.00002")])
    paper = pg_session.execute(
        select(models.Paper).where(models.Paper.arxiv_id == "2401.00001")
    ).scalar_one()
    pg_session.add(
        models.PaperEmbedding(paper_id=paper.id, embedding=[0.5] * 768, model="test-model")
    )
    pg_session.commit()
    return pg_session


def test_wipe_clears_every_table(seeded_session):
    Repository(seeded_session).wipe()

    assert all(count(seeded_session, table) == 0 for table in ALL_TABLES)


def test_wipe_embeddings_only_keeps_papers(seeded_session):
    Repository(seeded_session).wipe(embeddings_only=True)

    assert count(seeded_session, models.PaperEmbedding) == 0
    assert count(seeded_session, models.Paper) == 2
    assert count(seeded_session, models.Author) == 1
    assert count(seeded_session, models.paper_authors) == 2


def test_wipe_on_empty_database_is_a_noop(pg_session):
    Repository(pg_session).wipe()  # must not raise
    Repository(pg_session).wipe(embeddings_only=True)

    assert all(count(pg_session, table) == 0 for table in ALL_TABLES)
