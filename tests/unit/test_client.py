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


def test_builds_and_query_for_multiple_categories():
    captured: list[httpx.Request] = []
    client = make_client(ok_handler(captured))
    client.fetch_page(["cs.AI", "cs.LG", "cs.CL"])
    assert captured[0].url.params["search_query"] == "cat:cs.AI AND cat:cs.LG AND cat:cs.CL"


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
