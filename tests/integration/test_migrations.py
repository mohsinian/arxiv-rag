import pytest
from sqlalchemy import create_engine, inspect

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "papers",
    "authors",
    "categories",
    "paper_authors",
    "paper_categories",
    "paper_embeddings",
}


def test_upgrade_creates_all_expected_tables(migrated_engine):
    names = set(inspect(migrated_engine).get_table_names())
    assert names >= EXPECTED_TABLES


def test_embedding_column_is_a_768_dimension_vector(migrated_engine):
    columns = {c["name"]: c for c in inspect(migrated_engine).get_columns("paper_embeddings")}
    type_repr = str(columns["embedding"]["type"]).upper()
    assert type_repr.startswith("VECTOR")
    assert "768" in type_repr


def test_unique_constraints_exist(migrated_engine):
    inspector = inspect(migrated_engine)
    papers_uniques = {tuple(u["column_names"]) for u in inspector.get_unique_constraints("papers")}
    assert ("arxiv_id",) in papers_uniques
    authors_uniques = {
        tuple(u["column_names"]) for u in inspector.get_unique_constraints("authors")
    }
    assert ("name",) in authors_uniques


def test_downgrade_base_drops_all_tables(alembic, pg_test_db_url):
    alembic("upgrade", "head")
    alembic("downgrade", "base")
    engine = create_engine(pg_test_db_url)
    try:
        names = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert names <= {"alembic_version"}
