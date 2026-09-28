"""Fixtures for integration tests: a throwaway Postgres database per session.

Requires the composed database (`make up-db`). Tests never touch the dev
`arxiv_rag` database — they get `arxiv_rag_test`, created and dropped here,
with the schema managed by Alembic exactly as production would.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).parents[2]
TEST_DB_NAME = "arxiv_rag_test"


def _database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url
    from arxiv_rag.config import get_settings

    return get_settings().database_url


def _run_alembic(url: str, *args: str) -> None:
    env = {**os.environ, "DATABASE_URL": url}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(REPO_ROOT / "alembic.ini"), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic {' '.join(args)} failed:\n{result.stderr}"


@pytest.fixture(scope="session")
def pg_test_db_url() -> str:
    base = make_url(_database_url())
    admin_url = base.set(database="postgres")
    test_url = base.set(database=TEST_DB_NAME)

    admin = create_engine(admin_url)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin.dispose()

    # str(url) would render the password as literal '***' (SQLAlchemy 2.0 safety
    # default) — the subprocess needs the real credentials.
    yield test_url.render_as_string(hide_password=False)

    admin = create_engine(admin_url)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def migrated_engine(pg_test_db_url):
    """Schema upgraded for the test, downgraded after it."""
    _run_alembic(pg_test_db_url, "upgrade", "head")
    engine = create_engine(pg_test_db_url)
    yield engine
    engine.dispose()
    _run_alembic(pg_test_db_url, "downgrade", "base")


@pytest.fixture
def pg_session(migrated_engine):
    with Session(migrated_engine) as session:
        yield session


@pytest.fixture
def alembic(pg_test_db_url):
    """Callable: run an alembic command against the test database."""

    def _run(*args: str) -> None:
        _run_alembic(pg_test_db_url, *args)

    return _run
