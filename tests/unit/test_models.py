from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from arxiv_rag.db import models

pytestmark = pytest.mark.unit

UTC = timezone.utc


def make_category(session, name="cs.LG") -> models.Category:
    category = models.Category(arxiv_category=name)
    session.add(category)
    session.flush()
    return category


def make_author(session, name="Alice Researcher") -> models.Author:
    author = models.Author(name=name)
    session.add(author)
    session.flush()
    return author


def make_paper(session, category, arxiv_id="2401.00001", doi=None) -> models.Paper:
    paper = models.Paper(
        arxiv_id=arxiv_id,
        title="A Title",
        abstract="An abstract.",
        published_at=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
        updated_at=datetime(2024, 2, 1, 10, 30, tzinfo=UTC),
        doi=doi,
        primary_category_id=category.id,
    )
    session.add(paper)
    session.flush()
    return paper


def test_duplicate_arxiv_id_is_rejected(sqlite_session):
    s = sqlite_session
    category = make_category(s)
    make_paper(s, category, arxiv_id="2401.00001")
    with pytest.raises(IntegrityError):
        make_paper(s, category, arxiv_id="2401.00001")


def test_doi_and_journal_ref_are_optional(sqlite_session):
    s = sqlite_session
    category = make_category(s)
    paper = make_paper(s, category, doi=None)
    s.commit()
    refreshed = s.get(models.Paper, paper.id)
    assert refreshed.doi is None
    assert refreshed.journal_ref is None


def test_duplicate_author_name_is_rejected(sqlite_session):
    s = sqlite_session
    make_author(s, name="Alice Researcher")
    with pytest.raises(IntegrityError):
        make_author(s, name="Alice Researcher")


def test_duplicate_category_is_rejected(sqlite_session):
    s = sqlite_session
    make_category(s, name="cs.LG")
    with pytest.raises(IntegrityError):
        make_category(s, name="cs.LG")


def test_paper_author_join_rejects_duplicates_and_preserves_position(sqlite_session):
    s = sqlite_session
    category = make_category(s)
    paper = make_paper(s, category)
    first = make_author(s, name="First Author")
    second = make_author(s, name="Second Author")

    s.execute(
        models.paper_authors.insert().values(paper_id=paper.id, author_id=first.id, position=0)
    )
    s.execute(
        models.paper_authors.insert().values(paper_id=paper.id, author_id=second.id, position=1)
    )
    with pytest.raises(IntegrityError):
        s.execute(
            models.paper_authors.insert().values(paper_id=paper.id, author_id=second.id, position=2)
        )

    rows = (
        s.execute(
            select(models.Author.name)
            .join(models.paper_authors, models.paper_authors.c.author_id == models.Author.id)
            .where(models.paper_authors.c.paper_id == paper.id)
            .order_by(models.paper_authors.c.position)
        )
        .scalars()
        .all()
    )
    assert rows == ["First Author", "Second Author"]


def test_paper_categories_join_rejects_duplicates(sqlite_session):
    s = sqlite_session
    category = make_category(s)
    paper = make_paper(s, category)
    s.execute(models.paper_categories.insert().values(paper_id=paper.id, category_id=category.id))
    with pytest.raises(IntegrityError):
        s.execute(
            models.paper_categories.insert().values(paper_id=paper.id, category_id=category.id)
        )


def test_embedding_rows_are_tied_to_papers(sqlite_session):
    s = sqlite_session
    category = make_category(s)
    paper = make_paper(s, category)
    s.add(models.PaperEmbedding(paper_id=paper.id, embedding=[0.1, 0.2], model="test-model"))
    s.commit()
    count = s.scalar(select(func.count()).select_from(models.PaperEmbedding))
    assert count == 1
    s.execute(delete(models.Paper).where(models.Paper.id == paper.id))
    s.commit()
    count = s.scalar(select(func.count()).select_from(models.PaperEmbedding))
    assert count == 0  # cascade removes the embedding with its paper
