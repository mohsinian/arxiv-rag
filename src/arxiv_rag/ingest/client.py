"""HTTP client for the arXiv API (https://export.arxiv.org/api/query)."""

import itertools
import logging
import time
from collections.abc import Iterator

import httpx

import arxiv_rag
from arxiv_rag.config import Settings
from arxiv_rag.ingest.parser import PaperRecord, parse_atom

logger = logging.getLogger(__name__)

ARXIV_API_BASE_URL = "https://export.arxiv.org"
ARXIV_API_PATH = "/api/query"
_BACKOFF_CAP_SECONDS = 30.0


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
        sleep_fn=time.sleep,
        now_fn=time.monotonic,
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
        """Fetch one page of papers, retrying transient failures.

        arXiv asks for at most one request per 3 seconds; every request
        (retries included) is spaced accordingly. 429/5xx responses and
        network errors are retried with exponential backoff; other 4xx
        responses fail immediately.
        """
        if max_results is None:
            max_results = self._settings.arxiv_page_size
        params = self._page_params(categories, start, max_results)

        extra_wait = 0.0
        for attempt in itertools.count():
            self._respect_rate_limit(extra_wait)
            extra_wait = 0.0
            try:
                logger.debug("arXiv GET %s?%s", ARXIV_API_PATH, params)
                response = self._http.get(ARXIV_API_PATH, params=params)
            except httpx.TransportError as exc:
                if attempt >= self._settings.arxiv_max_retries:
                    raise ArxivHTTPError(
                        f"arXiv API unreachable after {attempt + 1} attempts: {exc}"
                    ) from exc
                extra_wait = self._backoff_seconds(attempt)
                logger.warning("arXiv request failed (%s); retrying", exc)
                continue
            if response.status_code == 200:
                return parse_atom(response.text)
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and attempt < self._settings.arxiv_max_retries:
                extra_wait = self._retry_wait(response, attempt)
                logger.warning(
                    "arXiv API returned HTTP %d; retrying in %.1fs",
                    response.status_code,
                    extra_wait,
                )
                continue
            raise ArxivHTTPError(
                f"arXiv API returned HTTP {response.status_code} after {attempt + 1} attempt(s)"
            )

    def _respect_rate_limit(self, extra_wait: float = 0.0) -> None:
        """Sleep just enough that consecutive requests stay ≥ the interval apart."""
        remaining = 0.0
        if self._last_request_at is not None:
            remaining = self._settings.arxiv_sleep_seconds - (self._now() - self._last_request_at)
        wait = max(extra_wait, remaining)
        if wait > 0:
            self._sleep(wait)
        self._last_request_at = self._now()

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        return float(min(2**attempt, _BACKOFF_CAP_SECONDS))

    @classmethod
    def _retry_wait(cls, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        return cls._backoff_seconds(attempt)

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
