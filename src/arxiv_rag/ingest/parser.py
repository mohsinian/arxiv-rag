"""Parse arXiv Atom/XML API responses into clean records.

The parser is strictly structural: it extracts fields faithfully (including
whitespace as served) and leaves text/date normalization to `preprocess`.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

_ABS_URL_PREFIX = "http://arxiv.org/abs/"
_VERSION_SUFFIX = re.compile(r"v\d+$")


class ArxivParseError(Exception):
    """Raised when an arXiv Atom response cannot be parsed into records."""


@dataclass(frozen=True)
class PaperRecord:
    """One paper as extracted from the feed, before database storage."""

    arxiv_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str
    published: datetime
    updated: datetime
    doi: str | None
    journal_ref: str | None


def parse_atom(xml: str | bytes) -> list[PaperRecord]:
    """Parse an arXiv Atom feed into records, in document order."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ArxivParseError(f"could not parse Atom XML: {exc}") from exc

    if root.tag != f"{{{ATOM_NS}}}feed":
        raise ArxivParseError(f"expected an Atom <feed> root element, got {root.tag!r}")

    return [_parse_entry(entry) for entry in root.findall(f"{{{ATOM_NS}}}entry")]


def _parse_entry(entry: ET.Element) -> PaperRecord:
    entry_id = entry.findtext(f"{{{ATOM_NS}}}id")
    title = entry.findtext(f"{{{ATOM_NS}}}title")
    summary = entry.findtext(f"{{{ATOM_NS}}}summary")
    published = _parse_date(entry.findtext(f"{{{ATOM_NS}}}published"))
    updated = _parse_date(entry.findtext(f"{{{ATOM_NS}}}updated"))

    missing = [
        name
        for name, value in [
            ("id", entry_id),
            ("title", title),
            ("summary", summary),
            ("published", published),
            ("updated", updated),
        ]
        if not value
    ]
    if missing:
        raise ArxivParseError(f"entry is missing required field(s): {', '.join(missing)}")

    assert entry_id is not None and title is not None and summary is not None
    assert published is not None and updated is not None

    authors = tuple(
        name.text
        for author in entry.findall(f"{{{ATOM_NS}}}author")
        if (name := author.find(f"{{{ATOM_NS}}}name")) is not None and name.text
    )
    categories = tuple(
        category.get("term")
        for category in entry.findall(f"{{{ATOM_NS}}}category")
        if category.get("term")
    )

    primary = entry.find(f"{{{ARXIV_NS}}}primary_category")
    primary_category = primary.get("term") if primary is not None else None
    if not primary_category:
        if not categories:
            raise ArxivParseError(f"entry {entry_id} has no primary category or categories")
        primary_category = categories[0]

    return PaperRecord(
        arxiv_id=_extract_arxiv_id(entry_id),
        title=title,
        abstract=summary,
        authors=authors,
        categories=categories,
        primary_category=primary_category,
        published=published,
        updated=updated,
        doi=_optional_text(entry.findtext(f"{{{ARXIV_NS}}}doi")),
        journal_ref=_optional_text(entry.findtext(f"{{{ARXIV_NS}}}journal_ref")),
    )


def _extract_arxiv_id(entry_id: str) -> str:
    """Turn `<id>http://arxiv.org/abs/2401.12345v2</id>` into `2401.12345`.

    Old-style ids keep their archive prefix (`cs/0501001`). The version suffix
    is dropped so re-ingesting a new version of the same paper maps to the
    same row instead of duplicating it.
    """
    text = entry_id.strip()
    if text.startswith(_ABS_URL_PREFIX):
        text = text[len(_ABS_URL_PREFIX) :]
    else:
        text = text.rsplit("/", 1)[-1]
    return _VERSION_SUFFIX.sub("", text)


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    # Python 3.10's fromisoformat does not accept the trailing "Z"
    return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))


def _optional_text(raw: str | None) -> str | None:
    text = (raw or "").strip()
    return text or None
