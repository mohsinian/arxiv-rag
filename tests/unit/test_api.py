import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from arxiv_rag.api.app import create_app
from arxiv_rag.db.models import Base

pytestmark = pytest.mark.unit


class BrokenSession:
    def execute(self, *_args, **_kwargs):
        raise RuntimeError("db is down")

    def close(self):
        pass


@pytest.fixture
def sqlite_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_create_app_returns_fastapi_with_health_route():
    app = create_app()
    assert isinstance(app, FastAPI)
    assert "/health" in app.openapi()["paths"]


def test_health_reports_ok_without_configured_db():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "skipped"}


def test_health_checks_the_database(sqlite_factory):
    client = TestClient(create_app(session_factory=sqlite_factory))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


def test_health_reports_503_when_db_is_down():
    client = TestClient(create_app(session_factory=lambda: BrokenSession()))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"]["db"] == "error"


def test_unknown_route_returns_fastapi_404_shape():
    client = TestClient(create_app())
    response = client.get("/nope")
    assert response.status_code == 404
    assert "detail" in response.json()
