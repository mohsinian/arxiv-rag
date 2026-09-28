from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from arxiv_rag.api.app import create_app
from arxiv_rag.db.models import Base
from arxiv_rag.db.repo import Repository
from arxiv_rag.ingest.parser import PaperRecord

pytestmark = pytest.mark.unit

UTC = timezone.utc


def rec(arxiv_id: str, categories: tuple[str, ...], authors: tuple[str, ...]) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract=f"Abstract {arxiv_id}.",
        authors=authors,
        categories=categories,
        primary_category=categories[0],
        published=datetime(2024, 1, 10, tzinfo=UTC),
        updated=datetime(2024, 1, 10, tzinfo=UTC),
        doi=None,
        journal_ref=None,
    )


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        Repository(session).upsert_records(
            [
                rec("2401.00001", ("cs.LG", "cs.AI"), ("Alice", "Bob")),
                rec("2401.00002", ("cs.LG",), ("Alice",)),
                rec("2401.00003", ("cs.CL",), ("Carol", "Alice", "Bob")),
            ]
        )
    return TestClient(create_app(session_factory=factory))


def test_top_categories_orders_by_count_then_name(client):
    response = client.get("/api/v1/stats/top-categories")
    assert response.status_code == 200
    assert response.json() == [
        {"category": "cs.LG", "count": 2},
        {"category": "cs.AI", "count": 1},
        {"category": "cs.CL", "count": 1},
    ]


def test_top_categories_respects_limit(client):
    response = client.get("/api/v1/stats/top-categories", params={"limit": 2})
    assert response.status_code == 200
    assert response.json() == [
        {"category": "cs.LG", "count": 2},
        {"category": "cs.AI", "count": 1},
    ]


def test_top_authors_orders_by_count(client):
    response = client.get("/api/v1/stats/top-authors")
    assert response.status_code == 200
    assert response.json() == [
        {"author": "Alice", "count": 3},
        {"author": "Bob", "count": 2},
        {"author": "Carol", "count": 1},
    ]


def test_summary_counts_and_average_authors_per_paper(client):
    response = client.get("/api/v1/stats/summary")
    assert response.status_code == 200
    assert response.json() == {
        "papers": 3,
        "authors": 3,
        "categories": 3,
        "avg_authors_per_paper": 2.0,  # (2 + 1 + 3) / 3
    }


def test_summary_on_empty_database_is_zeroed():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    client = TestClient(create_app(session_factory=factory))

    response = client.get("/api/v1/stats/summary")

    assert response.status_code == 200
    assert response.json() == {
        "papers": 0,
        "authors": 0,
        "categories": 0,
        "avg_authors_per_paper": 0,
    }


def test_limits_must_be_positive(client):
    assert client.get("/api/v1/stats/top-categories", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/stats/top-authors", params={"limit": -1}).status_code == 422
