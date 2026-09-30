"""Ingestion orchestration: arXiv fetch → clean → upsert, committed per page."""

import logging
from collections.abc import Iterator
from itertools import islice

from arxiv_rag.db.repo import IngestSummary, Repository
from arxiv_rag.ingest.client import ArxivClient
from arxiv_rag.ingest.parser import PaperRecord
from arxiv_rag.ingest.preprocess import clean_record

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Runs the ingestion pipeline with per-page commit granularity.

    A crash or API failure mid-ingest never loses already-fetched pages:
    each chunk of records is its own transaction. The failing exception
    propagates to the caller; records from the in-flight chunk are simply
    re-fetched on the next run.
    """

    def __init__(self, client: ArxivClient, repo: Repository) -> None:
        self._client = client
        self._repo = repo

    def run(self, categories: list[str], limit: int, page_size: int = 100) -> IngestSummary:
        """Ingest the newest `limit` papers in any of `categories`."""
        return self._ingest(self._client.iter_papers(categories, limit=limit), page_size)

    def run_ids(self, arxiv_ids: list[str], page_size: int = 100) -> IngestSummary:
        """Ingest specific papers by arXiv id (e.g. an eval set's gold papers)."""
        return self._ingest(self._client.iter_papers_by_ids(arxiv_ids), page_size)

    def _ingest(self, records: Iterator[PaperRecord], page_size: int) -> IngestSummary:
        total = IngestSummary()
        while True:
            batch = list(islice(records, page_size))
            if not batch:
                break
            summary = self._repo.upsert_records([clean_record(r) for r in batch])
            total = total + summary
            logger.info("ingest progress: %s", total)
        logger.info("ingest finished: %s", total)
        return total
