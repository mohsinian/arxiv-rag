"""HTTP client for the arXiv API (https://export.arxiv.org/api/query)."""

import logging
from collections.abc import Iterator

import httpx

import arxiv_rag
from arxiv_rag.config import Settings
from arxiv_rag.ingest.parser import PaperRecord, parse_atom

logger = logging.getLogger(__name__)

ARXIV_API_BASE_URL = "https://export.arxiv.org"
ARXIV_API_PATH = "/api/query"


class ArxivHTTPError(Exception):
    """Raised when the arXiv API responds with an unrecoverable failure."""


class ArxivClient:
    """Fetches paper records from the arXiv Atom API.

    Pass `transport=httpx.MockTransport(...)` in tests; `sleep_fn`/`now_fn`
    are injectable so rate-limiting can be tested without real waiting.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn=None,
        now_fn=None,
    ) -> None:
        self._settings = settings
        self._sleep = sleep_fn
        self._now = now_fn
        self._last_request_at: float | None = None
        self._http = httpx.Client(
            base_url=ARXIV_API_BASE_URL,
            transport=transport,
            timeout=30.0,
            headers={"User-Agent": f"arxiv-rag/{arxiv_rag.__version__}"},
        )

    def fetch_page(
        self,
        categories: list[str],
        start: int = 0,
        max_results: int | None = None,
    ) -> list[PaperRecord]:
        """Fetch one page of papers for the given categories."""
        if max_results is None:
            max_results = self._settings.arxiv_page_size
        params = self._page_params(categories, start, max_results)
        logger.debug("arXiv GET %s?%s", ARXIV_API_PATH, params)
        response = self._http.get(ARXIV_API_PATH, params=params)
        if response.status_code != 200:
            raise ArxivHTTPError(f"arXiv API returned HTTP {response.status_code}")
        return parse_atom(response.text)

    @staticmethod
    def _page_params(
        categories: list[str],
        start: int,
        max_results: int | None,
    ) -> dict[str, str | int]:
        cleaned = [c.strip() for c in categories if c and c.strip()]
        if not cleaned:
            raise ValueError("at least one category is required, e.g. ['cs.AI']")
        return {
            "search_query": " AND ".join(f"cat:{c}" for c in cleaned),
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

    def iter_papers(self, categories: list[str], limit: int) -> Iterator[PaperRecord]:
        """Yield up to `limit` papers, transparently paginating through the API.

        Stops early when a page comes back shorter than requested or empty —
        both mean the result set is exhausted.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        fetched = 0
        while fetched < limit:
            page_size = min(self._settings.arxiv_page_size, limit - fetched)
            records = self.fetch_page(categories, start=fetched, max_results=page_size)
            if not records:
                return
            yield from records
            fetched += len(records)
            if len(records) < page_size:
                return

    def close(self) -> None:
        self._http.close()
