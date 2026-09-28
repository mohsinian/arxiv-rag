"""Text and date normalization for parsed arXiv records."""

import unicodedata
from datetime import datetime, timezone

from arxiv_rag.ingest.parser import PaperRecord


def clean_record(record: PaperRecord) -> PaperRecord:
    """Return a normalized copy of `record`; the input is left untouched.

    - title/abstract: collapse all whitespace runs (newlines included) to one space
    - author names: unicode-normalized (NFKC), whitespace collapsed, empties dropped
      — this is also the form the DB stores, so name dedup stays consistent
    - dates: converted to UTC; naive datetimes are assumed UTC
    - doi/journal_ref: stripped; blank values become None
    """
    published = _to_utc(record.published)
    updated = _to_utc(record.updated)
    return PaperRecord(
        arxiv_id=record.arxiv_id.strip(),
        title=_collapse_ws(record.title),
        abstract=_collapse_ws(record.abstract),
        authors=tuple(a for a in (_clean_name(n) for n in record.authors) if a),
        categories=tuple(dict.fromkeys(c.strip() for c in record.categories)),
        primary_category=record.primary_category.strip(),
        published=published,
        updated=updated,
        doi=_clean_optional(record.doi),
        journal_ref=_clean_optional(record.journal_ref),
    )


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _clean_name(name: str) -> str:
    return _collapse_ws(unicodedata.normalize("NFKC", name))


def _clean_optional(text: str | None) -> str | None:
    cleaned = _collapse_ws(text) if text else ""
    return cleaned or None


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
