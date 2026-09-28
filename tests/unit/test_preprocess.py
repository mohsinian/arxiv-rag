from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from arxiv_rag.ingest.parser import PaperRecord
from arxiv_rag.ingest.preprocess import clean_record

pytestmark = pytest.mark.unit

UTC = timezone.utc
PLUS5 = timezone(timedelta(hours=5))


def dirty_record() -> PaperRecord:
    return PaperRecord(
        arxiv_id="2401.12345",
        title="  A   Title\nwith\twhitespace  ",
        abstract=""" Abstract text
            with     irregular spacing
            and newlines.  """,
        authors=("  Alice   Researcher ", "", "Bob\nEngineer"),
        categories=("cs.LG", "cs.AI", "cs.LG"),
        primary_category="cs.LG",
        published=datetime(2024, 1, 15, 10, 30),  # naive — must be assumed UTC
        updated=datetime(2024, 2, 1, 15, 30, tzinfo=PLUS5),  # 10:30 UTC
        doi="  10.1234/test.2024.5678  ",
        journal_ref="   ",
    )


def test_title_and_abstract_whitespace_is_collapsed():
    r = clean_record(dirty_record())
    assert r.title == "A Title with whitespace"
    assert r.abstract == "Abstract text with irregular spacing and newlines."


def test_dates_are_normalized_to_utc():
    r = clean_record(dirty_record())
    assert r.published.tzinfo == UTC
    assert r.published == datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
    assert r.updated == datetime(2024, 2, 1, 10, 30, tzinfo=UTC)


def test_author_names_are_normalized_and_empties_dropped():
    r = clean_record(dirty_record())
    assert r.authors == ("Alice Researcher", "Bob Engineer")


def test_categories_are_deduped_preserving_order():
    r = clean_record(dirty_record())
    assert r.categories == ("cs.LG", "cs.AI")


def test_doi_is_stripped_and_blank_journal_ref_becomes_none():
    r = clean_record(dirty_record())
    assert r.doi == "10.1234/test.2024.5678"
    assert r.journal_ref is None


def test_input_record_is_not_mutated():
    original = dirty_record()
    clean_record(original)
    assert original.title == "  A   Title\nwith\twhitespace  "
    assert original.authors == ("  Alice   Researcher ", "", "Bob\nEngineer")


def test_duplicate_author_names_within_a_record_are_deduped():
    # real arXiv data: consortium papers can list the same author twice
    record = replace(
        dirty_record(), authors=("Alice Researcher", "Bob Engineer", "Alice Researcher")
    )
    r = clean_record(record)
    assert r.authors == ("Alice Researcher", "Bob Engineer")
