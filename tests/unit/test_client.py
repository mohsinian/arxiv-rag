from pathlib import Path

import httpx
import pytest

from arxiv_rag.config import Settings
from arxiv_rag.ingest.client import ArxivClient, ArxivHTTPError

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[1] / "fixtures" / "atom"


def load_atom(name: str) -> str:
    return (FIXTURES / name).read_text()


def make_client(handler, **overrides) -> ArxivClient:
    # sleep=0 so un-throttled tests run instantly; throttling has its own tests
    settings = Settings(_env_file=None, arxiv_sleep_seconds=0.0, **overrides)
    return ArxivClient(settings, transport=httpx.MockTransport(handler))


def ok_handler(captured: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text=load_atom("two_entries.xml"))

    return handler


def test_builds_query_params_for_single_category():
    captured: list[httpx.Request] = []
    client = make_client(ok_handler(captured))
    client.fetch_page(["cs.AI"])
    params = captured[0].url.params
    assert params["search_query"] == "cat:cs.AI"
    assert params["start"] == "0"
    assert params["max_results"] == "100"
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"


def test_builds_or_query_for_multiple_categories():
    captured: list[httpx.Request] = []
    client = make_client(ok_handler(captured))
    client.fetch_page(["cs.AI", "cs.LG", "cs.CL"])
    assert captured[0].url.params["search_query"] == "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"


def test_start_and_max_results_are_passed_through():
    captured: list[httpx.Request] = []
    client = make_client(ok_handler(captured), arxiv_page_size=50)
    client.fetch_page(["cs.AI"], start=150, max_results=50)
    assert captured[0].url.params["start"] == "150"
    assert captured[0].url.params["max_results"] == "50"


def test_requests_go_to_the_arxiv_api_with_identifying_user_agent():
    captured: list[httpx.Request] = []
    client = make_client(ok_handler(captured))
    client.fetch_page(["cs.AI"])
    request = captured[0]
    assert request.url.host == "export.arxiv.org"
    assert request.url.path == "/api/query"
    assert request.headers["user-agent"].startswith("arxiv-rag/")


def test_response_body_is_parsed_into_records():
    client = make_client(lambda request: httpx.Response(200, text=load_atom("two_entries.xml")))
    records = client.fetch_page(["cs.AI"])
    assert [r.arxiv_id for r in records] == ["2401.12345", "cs/0501001"]


def test_client_error_raises_arxiv_http_error():
    client = make_client(lambda request: httpx.Response(400, text="bad request"))
    with pytest.raises(ArxivHTTPError, match="400"):
        client.fetch_page(["cs.AI"])


def test_empty_categories_are_rejected():
    client = make_client(lambda request: httpx.Response(200, text=load_atom("two_entries.xml")))
    with pytest.raises(ValueError, match="categor"):
        client.fetch_page([])


# ---------------------------------------------------------------------------
# Pagination (iter_papers)

ENTRY_TPL = """
<entry>
  <id>http://arxiv.org/abs/{arxiv_id}v1</id>
  <updated>2024-02-01T00:00:00Z</updated>
  <published>2024-02-01T00:00:00Z</published>
  <title>Paper {n}</title>
  <summary>Abstract number {n}.</summary>
  <author><name>Author {n}</name></author>
  <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
</entry>
"""


def make_feed(*arxiv_ids: str) -> str:
    entries = "".join(ENTRY_TPL.format(arxiv_id=i, n=idx + 1) for idx, i in enumerate(arxiv_ids))
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


def queue_handler(pages: list[str], captured: list[httpx.Request]):
    pages = list(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text=pages.pop(0))

    return handler


def test_iter_papers_yields_records_across_pages():
    captured: list[httpx.Request] = []
    client = make_client(
        queue_handler(
            [
                make_feed("2401.00001", "2401.00002"),
                make_feed("2401.00003", "2401.00004"),
                make_feed("2401.00005", "2401.00006"),
            ],
            captured,
        ),
        arxiv_page_size=2,
    )
    records = list(client.iter_papers(["cs.LG"], limit=6))
    assert [r.arxiv_id for r in records] == [
        "2401.00001",
        "2401.00002",
        "2401.00003",
        "2401.00004",
        "2401.00005",
        "2401.00006",
    ]
    starts = [request.url.params["start"] for request in captured]
    assert starts == ["0", "2", "4"]


def test_iter_papers_stops_when_a_page_comes_back_short():
    captured: list[httpx.Request] = []
    client = make_client(
        queue_handler([make_feed("2401.00001", "2401.00002"), make_feed("2401.00003")], captured),
        arxiv_page_size=2,
    )
    records = list(client.iter_papers(["cs.LG"], limit=100))
    assert [r.arxiv_id for r in records] == ["2401.00001", "2401.00002", "2401.00003"]
    assert len(captured) == 2  # a short page means the server is exhausted


def test_iter_papers_respects_limit_without_overfetching():
    captured: list[httpx.Request] = []
    client = make_client(
        queue_handler([make_feed("2401.00001", "2401.00002"), make_feed("2401.00003")], captured),
        arxiv_page_size=2,
    )
    records = list(client.iter_papers(["cs.LG"], limit=3))
    assert len(records) == 3
    assert captured[0].url.params["max_results"] == "2"
    assert captured[1].url.params["max_results"] == "1"  # last page shrunk to the remainder


def test_iter_papers_stops_on_empty_page():
    empty_feed = "<feed xmlns='http://www.w3.org/2005/Atom'/>"
    captured: list[httpx.Request] = []
    client = make_client(
        queue_handler([make_feed("2401.00001", "2401.00002"), empty_feed], captured),
        arxiv_page_size=2,
    )
    records = list(client.iter_papers(["cs.LG"], limit=100))
    assert [r.arxiv_id for r in records] == ["2401.00001", "2401.00002"]
    assert len(captured) == 2  # an empty page after a full one ends iteration


def test_iter_papers_rejects_non_positive_limit():
    client = make_client(lambda request: httpx.Response(200, text=make_feed()))
    with pytest.raises(ValueError, match="limit"):
        list(client.iter_papers(["cs.LG"], limit=0))


# Fetching specific papers (id_list)


def test_fetch_by_ids_uses_id_list_instead_of_a_search_query():
    captured: list[httpx.Request] = []
    client = make_client(queue_handler([make_feed("2401.00001", "2401.00002")], captured))
    records = client.fetch_by_ids(["2401.00001", " 2401.00002 ", "2401.00001"])
    assert [r.arxiv_id for r in records] == ["2401.00001", "2401.00002"]
    params = captured[0].url.params
    assert params["id_list"] == "2401.00001,2401.00002"  # stripped and deduplicated
    assert params["max_results"] == "2"
    assert "search_query" not in params


def test_fetch_by_ids_rejects_an_empty_list():
    client = make_client(lambda request: httpx.Response(200, text=make_feed()))
    with pytest.raises(ValueError, match="id"):
        client.fetch_by_ids(["", "  "])


def test_iter_papers_by_ids_requests_one_page_of_ids_at_a_time():
    captured: list[httpx.Request] = []
    client = make_client(
        queue_handler([make_feed("2401.00001", "2401.00002"), make_feed("2401.00003")], captured),
        arxiv_page_size=2,
    )
    records = list(client.iter_papers_by_ids(["2401.00001", "2401.00002", "2401.00003"]))
    assert [r.arxiv_id for r in records] == ["2401.00001", "2401.00002", "2401.00003"]
    assert [r.url.params["id_list"] for r in captured] == [
        "2401.00001,2401.00002",
        "2401.00003",
    ]


def test_iter_papers_by_ids_logs_ids_arxiv_did_not_return(caplog):
    client = make_client(queue_handler([make_feed("2401.00001")], []))
    with caplog.at_level("WARNING", logger="arxiv_rag.ingest.client"):
        records = list(client.iter_papers_by_ids(["2401.00001", "2401.99999"]))
    assert [r.arxiv_id for r in records] == ["2401.00001"]
    assert "2401.99999" in caplog.text
