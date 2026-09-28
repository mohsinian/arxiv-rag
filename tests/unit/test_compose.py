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
