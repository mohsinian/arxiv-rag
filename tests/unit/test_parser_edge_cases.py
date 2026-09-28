import pytest

from arxiv_rag.ingest.parser import ArxivParseError, parse_atom

pytestmark = pytest.mark.unit

MINIMAL_ENTRY = """
<entry xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/abs/2402.99999v1</id>
  <updated>2024-02-20T00:00:00Z</updated>
  <published>2024-02-20T00:00:00Z</published>
  <title>Minimal</title>
  <summary>A minimal but valid entry.</summary>
</entry>
"""


def wrap_feed(inner: str) -> str:
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{inner}</feed>'


def test_empty_feed_returns_empty_list():
    assert parse_atom(wrap_feed("")) == []


def test_malformed_xml_raises_parse_error():
    with pytest.raises(ArxivParseError, match="could not parse"):
        parse_atom("<feed xmlns='http://www.w3.org/2005/Atom'><entry>…")


def test_non_atom_root_raises_parse_error():
    with pytest.raises(ArxivParseError, match="Atom <feed>"):
        parse_atom("<html><body>not a feed</body></html>")


def test_entry_missing_required_field_names_the_field():
    broken = MINIMAL_ENTRY.replace("<summary>A minimal but valid entry.</summary>", "")
    with pytest.raises(ArxivParseError, match="summary"):
        parse_atom(wrap_feed(broken))


def test_author_without_name_is_skipped():
    entry = MINIMAL_ENTRY.replace(
        "</entry>",
        """
        <author><name>Real Author</name></author>
        <author></author>
        <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
        </entry>
        """,
    )
    (record,) = parse_atom(wrap_feed(entry))
    assert record.authors == ("Real Author",)


def test_missing_primary_category_falls_back_to_first_category():
    entry = MINIMAL_ENTRY.replace(
        "</entry>",
        """
        <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
        <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
        </entry>
        """,
    )
    (record,) = parse_atom(wrap_feed(entry))
    assert record.primary_category == "cs.CL"
    assert record.categories == ("cs.CL", "cs.AI")


def test_no_categories_at_all_raises_parse_error():
    with pytest.raises(ArxivParseError, match="no primary category"):
        parse_atom(wrap_feed(MINIMAL_ENTRY))
