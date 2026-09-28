from datetime import datetime, timezone
from pathlib import Path

import pytest

from arxiv_rag.ingest.parser import parse_atom

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[1] / "fixtures" / "atom"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_parses_full_modern_entry():
    records = parse_atom(load_fixture("two_entries.xml"))
    assert len(records) == 2
    r = records[0]
    assert r.arxiv_id == "2401.12345"  # version suffix stripped
    assert r.title == "Attention Is All We Need for Cheap Embeddings"
    assert r.authors == ("Alice Researcher", "Bob Engineer", "Carol Scientist")
    assert r.primary_category == "cs.LG"
    assert r.categories == ("cs.LG", "cs.AI", "stat.ML")
    assert r.published == datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
    assert r.updated == datetime(2024, 2, 1, 10, 30, tzinfo=timezone.utc)
    assert r.doi == "10.1234/test.2024.5678"
    assert r.journal_ref == "J. Test. Comput. 12, 34 (2024)"
    assert "cheap method" in r.abstract


def test_parses_old_style_entry_without_doi_or_journal_ref():
    records = parse_atom(load_fixture("two_entries.xml"))
    r = records[1]
    assert r.arxiv_id == "cs/0501001"  # archive prefix kept, version stripped
    assert r.title == "Neural Nets & Beyond"  # XML entity decoded
    assert r.authors == ("Old Timer",)
    assert r.primary_category == "cs.AI"
    assert r.categories == ("cs.AI",)
    assert r.doi is None
    assert r.journal_ref is None


def test_records_preserve_document_order():
    records = parse_atom(load_fixture("two_entries.xml"))
    assert [r.arxiv_id for r in records] == ["2401.12345", "cs/0501001"]


def test_abstract_keeps_raw_text_until_preprocessing():
    r = parse_atom(load_fixture("two_entries.xml"))[0]
    # newlines and irregular spacing survive parsing; preprocessing cleans them
    assert "\n" in r.abstract or "  " in r.abstract
