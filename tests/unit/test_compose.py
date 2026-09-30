from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[2]


def load_compose():
    return yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())


def test_compose_defines_pgvector_db_service():
    db = load_compose()["services"]["db"]
    assert db["image"].startswith("pgvector/pgvector:")
    assert "5432" in str(db["ports"])


def test_db_service_has_healthcheck_and_init_script_mount():
    db = load_compose()["services"]["db"]
    assert "healthcheck" in db
    mounts = " ".join(db["volumes"])
    assert "/docker-entrypoint-initdb.d" in mounts


def test_init_sql_enables_vector_extension():
    sql = (REPO_ROOT / "db" / "init" / "001_enable_vector.sql").read_text()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql


def test_compose_defines_api_service_built_from_repo():
    api = load_compose()["services"]["api"]
    assert api["build"]["context"] == "."
    assert "8000" in str(api["ports"])


def test_api_service_waits_for_healthy_db():
    api = load_compose()["services"]["api"]
    assert api["depends_on"]["db"]["condition"] == "service_healthy"


def test_compose_defaults_match_documented_rag_and_ingest_settings():
    api = load_compose()["services"]["api"]
    environment = api["environment"]
    assert environment["MIN_SCORE"] == "${MIN_SCORE:-0.55}"
    assert environment["ARXIV_SLEEP_SECONDS"] == "${ARXIV_SLEEP_SECONDS:-3}"
    assert environment["ARXIV_PAGE_SIZE"] == "${ARXIV_PAGE_SIZE:-100}"
    assert environment["ARXIV_MAX_RETRIES"] == "${ARXIV_MAX_RETRIES:-3}"


def test_database_port_is_host_local_only():
    db = load_compose()["services"]["db"]
    assert db["ports"] == ["127.0.0.1:${POSTGRES_PORT:-5432}:5432"]


def test_api_port_is_host_local_only():
    # /ask is unauthenticated and spends the LLM key — no LAN exposure by default
    api = load_compose()["services"]["api"]
    assert api["ports"] == ["127.0.0.1:${API_PORT:-8000}:8000"]


def test_image_entrypoint_migrates_schema_for_api_and_cli_runs():
    # `docker compose run` overrides the service command, so migrations must
    # live in the image ENTRYPOINT to also cover one-off CLI invocations
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    entrypoint = (REPO_ROOT / "docker" / "entrypoint.sh").read_text()
    assert 'ENTRYPOINT ["/app/entrypoint.sh"]' in dockerfile
    assert "alembic upgrade head" in entrypoint
    assert 'exec "$@"' in entrypoint
