from datetime import datetime, timedelta, timezone

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


def rec(arxiv_id: str, published: datetime, categories: tuple[str, ...]) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract=f"Abstract {arxiv_id}.",
        authors=("Alice Researcher",),
        categories=categories,
        primary_category=categories[0],
        published=published,
        updated=published,
        doi=None,
        journal_ref=None,
    )


@pytest.fixture
def client():
    """TestClient over a fresh seeded SQLite database."""
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
                rec("2401.00001", datetime(2024, 1, 10, tzinfo=UTC), ("cs.LG",)),
                rec(
                    "2401.00002",
                    datetime(2024, 1, 20, tzinfo=UTC),
                    ("cs.LG", "cs.AI"),
                ),
                rec("2401.00003", datetime(2024, 2, 5, tzinfo=UTC), ("cs.CL",)),
            ]
        )
    return TestClient(create_app(session_factory=factory))


def test_buckets_papers_by_month(client):
    response = client.get("/api/v1/stats/papers-by-month")
    assert response.status_code == 200
    assert response.json() == [
        {"month": "2024-01", "count": 2},
        {"month": "2024-02", "count": 1},
    ]


def test_category_filter_counts_only_matching_papers(client):
    response = client.get("/api/v1/stats/papers-by-month", params={"category": "cs.AI"})
    assert response.status_code == 200
    assert response.json() == [{"month": "2024-01", "count": 1}]


def test_unknown_category_yields_empty_list(client):
    response = client.get("/api/v1/stats/papers-by-month", params={"category": "astro-ph"})
    assert response.status_code == 200
    assert response.json() == []


def test_months_window_excludes_old_papers():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    with factory() as session:
        Repository(session).upsert_records(
            [
                rec("2501.00001", now - timedelta(days=5), ("cs.LG",)),
                rec("2101.00001", now - timedelta(days=400), ("cs.LG",)),
            ]
        )
    client = TestClient(create_app(session_factory=factory))

    response = client.get("/api/v1/stats/papers-by-month", params={"months": 12})

    assert response.status_code == 200
    months = [bucket["month"] for bucket in response.json()]
    assert len(months) == 1  # only the recent paper; the 400-day-old one is excluded


def test_empty_database_returns_empty_list():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    client = TestClient(create_app(session_factory=factory))

    response = client.get("/api/v1/stats/papers-by-month")

    assert response.status_code == 200
    assert response.json() == []


def test_months_must_be_positive(client):
    response = client.get("/api/v1/stats/papers-by-month", params={"months": 0})
    assert response.status_code == 422
