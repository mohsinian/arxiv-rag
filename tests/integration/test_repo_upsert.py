from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from arxiv_rag.db import models
from arxiv_rag.db.repo import IngestSummary, Repository
from arxiv_rag.ingest.parser import PaperRecord

pytestmark = pytest.mark.integration

UTC = timezone.utc


def record(**overrides) -> PaperRecord:
    defaults = dict(
        arxiv_id="2401.00001",
        title="A Paper Title",
        abstract="A paper abstract.",
        authors=("Alice Researcher", "Bob Engineer"),
        categories=("cs.LG", "cs.AI"),
        primary_category="cs.LG",
        published=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
        updated=datetime(2024, 2, 1, 10, 30, tzinfo=UTC),
        doi="10.1234/test",
        journal_ref="J. Test 1, 2 (2024)",
    )
    defaults.update(overrides)
    return PaperRecord(**defaults)


def count(session, table) -> int:
    return session.scalar(select(func.count()).select_from(table))


def paper_by_arxiv_id(session, arxiv_id) -> models.Paper:
    return session.execute(
        select(models.Paper).where(models.Paper.arxiv_id == arxiv_id)
    ).scalar_one()


def author_names_in_order(session, paper_id: models.Paper) -> list[str]:
    return list(
        session.execute(
            select(models.Author.name)
            .join(models.paper_authors, models.paper_authors.c.author_id == models.Author.id)
            .where(models.paper_authors.c.paper_id == paper_id)
            .order_by(models.paper_authors.c.position)
        ).scalars()
    )


def category_names_for(session, paper_id: models.Paper) -> list[str]:
    return list(
        session.execute(
            select(models.Category.arxiv_category)
            .join(
                models.paper_categories,
                models.paper_categories.c.category_id == models.Category.id,
            )
            .where(models.paper_categories.c.paper_id == paper_id)
        ).scalars()
    )


def test_inserts_new_record_with_all_relations(pg_session):
    repo = Repository(pg_session)
    summary = repo.upsert_records([record()])

    assert summary == IngestSummary(new=1, updated=0, unchanged=0)
    paper = paper_by_arxiv_id(pg_session, "2401.00001")
    assert paper.title == "A Paper Title"
    assert paper.doi == "10.1234/test"
    assert author_names_in_order(pg_session, paper.id) == ["Alice Researcher", "Bob Engineer"]
    assert category_names_for(pg_session, paper.id) == ["cs.LG", "cs.AI"]
    primary = pg_session.get(models.Category, paper.primary_category_id)
    assert primary.arxiv_category == "cs.LG"
    assert count(pg_session, models.paper_authors) == 2
    assert count(pg_session, models.paper_categories) == 2


def test_re_upserting_identical_record_is_unchanged(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records([record()])
    summary = repo.upsert_records([record()])

    assert summary == IngestSummary(new=0, updated=0, unchanged=1)
    assert count(pg_session, models.Paper) == 1
    assert count(pg_session, models.Author) == 2
    assert count(pg_session, models.Category) == 2
    assert count(pg_session, models.paper_authors) == 2
    assert count(pg_session, models.paper_categories) == 2


def test_changed_record_updates_fields_and_replaces_joins(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records([record()])

    summary = repo.upsert_records(
        [
            record(
                title="A Paper Title (v2)",
                abstract="A revised abstract.",
                authors=("Alice Researcher", "Carol Scientist", "Dan New"),
                categories=("cs.CL",),
                primary_category="cs.CL",
                updated=datetime(2024, 3, 1, 8, 0, tzinfo=UTC),
            )
        ]
    )

    assert summary == IngestSummary(new=0, updated=1, unchanged=0)
    paper = paper_by_arxiv_id(pg_session, "2401.00001")
    assert paper.title == "A Paper Title (v2)"
    assert paper.abstract == "A revised abstract."
    assert paper.updated_at == datetime(2024, 3, 1, 8, 0, tzinfo=UTC)
    assert author_names_in_order(pg_session, paper.id) == [
        "Alice Researcher",
        "Carol Scientist",
        "Dan New",
    ]
    assert category_names_for(pg_session, paper.id) == ["cs.CL"]
    # join rows are replaced, not accumulated
    assert count(pg_session, models.paper_authors) == 3
    assert count(pg_session, models.paper_categories) == 1


def test_author_shared_across_papers_is_stored_once(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records([record(arxiv_id="2401.00001")])
    repo.upsert_records([record(arxiv_id="2401.00002", authors=("Alice Researcher",))])

    assert count(pg_session, models.Author) == 2  # Alice + Bob, Alice not duplicated
    counts = dict(
        pg_session.execute(
            select(models.Author.name, func.count(models.paper_authors.c.paper_id))
            .join(models.paper_authors, models.paper_authors.c.author_id == models.Author.id)
            .group_by(models.Author.name)
        ).all()
    )
    assert counts["Alice Researcher"] == 2
    assert counts["Bob Engineer"] == 1


def test_changed_paper_loses_its_stale_embedding(pg_session):
    repo = Repository(pg_session)
    repo.upsert_records([record()])
    paper = paper_by_arxiv_id(pg_session, "2401.00001")
    pg_session.add(
        models.PaperEmbedding(paper_id=paper.id, embedding=[0.1] * 768, model="test-model")
    )
    pg_session.commit()

    repo.upsert_records([record(title="Changed title", updated=datetime(2024, 3, 1, tzinfo=UTC))])

    assert count(pg_session, models.PaperEmbedding) == 0


def test_failing_record_rolls_back_the_whole_batch(pg_session):
    repo = Repository(pg_session)
    # category longer than the String(64) column → database error mid-batch
    too_long = "x" * 65
    with pytest.raises(Exception, match="value too long"):
        repo.upsert_records([record(), record(arxiv_id="2401.00002", categories=(too_long,))])

    assert count(pg_session, models.Paper) == 0  # the earlier record was not half-committed


def test_duplicate_authors_within_one_record_do_not_crash(pg_session):
    # regression: a real arXiv paper listed the same author twice, which
    # violated the paper_authors primary key
    repo = Repository(pg_session)
    summary = repo.upsert_records(
        [record(authors=("Alice Researcher", "Bob Engineer", "Alice Researcher"))]
    )

    assert summary.new == 1
    paper = paper_by_arxiv_id(pg_session, "2401.00001")
    assert author_names_in_order(pg_session, paper.id) == ["Alice Researcher", "Bob Engineer"]


def test_category_order_in_the_feed_does_not_trigger_an_update(pg_session):
    # regression: stored rows come back in index order, not feed order, which
    # made every multi-category paper look "changed" on every re-ingest
    repo = Repository(pg_session)
    repo.upsert_records([record(categories=("cs.LG", "cs.AI", "cs.CL"))])

    summary = repo.upsert_records([record(categories=("cs.CL", "cs.AI", "cs.LG"))])

    assert summary == IngestSummary(new=0, updated=0, unchanged=1)
