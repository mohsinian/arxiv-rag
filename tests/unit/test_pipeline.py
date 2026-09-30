from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from arxiv_rag.db.repo import IngestSummary
from arxiv_rag.ingest.parser import PaperRecord
from arxiv_rag.ingest.pipeline import IngestPipeline

pytestmark = pytest.mark.unit

UTC = timezone.utc


def rec(arxiv_id: str) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=f"  Title   for   {arxiv_id}\n",
        abstract=f" Abstract\n for   {arxiv_id}. ",
        authors=("Alice Researcher",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 2, tzinfo=UTC),
        doi=None,
        journal_ref=None,
    )


class FakeClient:
    def __init__(self, pages: list[list[PaperRecord]]):
        self.pages = pages
        self.calls: list[tuple[list[str], int]] = []

    def iter_papers(self, categories: list[str], limit: int) -> Iterator[PaperRecord]:
        self.calls.append((list(categories), limit))
        for page in self.pages:
            yield from page

    def iter_papers_by_ids(self, arxiv_ids: list[str]) -> Iterator[PaperRecord]:
        self.calls.append((list(arxiv_ids), len(arxiv_ids)))
        for page in self.pages:
            yield from page

    def explode_after_pages(self, pages_before_failure: int) -> None:
        original = self.iter_papers

        def failing(categories: list[str], limit: int) -> Iterator[PaperRecord]:
            self.calls.append((list(categories), limit))
            for index, page in enumerate(self.pages):
                if index >= pages_before_failure:
                    raise RuntimeError("arXiv exploded")
                yield from page

        self.iter_papers = failing  # type: ignore[method-assign]
        _ = original


class FakeRepo:
    def __init__(self, results: list[IngestSummary] | None = None):
        self.results = list(results or [])
        self.batches: list[list[PaperRecord]] = []

    def upsert_records(self, records: list[PaperRecord]) -> IngestSummary:
        self.batches.append(list(records))
        return self.results.pop(0) if self.results else IngestSummary(new=len(records))


def test_cleans_records_before_upsert():
    repo = FakeRepo()
    pipeline = IngestPipeline(FakeClient([[rec("2401.00001")]]), repo)

    pipeline.run(["cs.AI"], limit=10)

    (batch,) = repo.batches
    assert batch[0].title == "Title for 2401.00001"  # whitespace collapsed
    assert batch[0].abstract == "Abstract for 2401.00001."


def test_passes_categories_and_limit_to_client():
    client, repo = FakeClient([]), FakeRepo()
    IngestPipeline(client, repo).run(["cs.AI", "cs.LG"], limit=1000)
    assert client.calls == [(["cs.AI", "cs.LG"], 1000)]


def test_commits_once_per_page_and_aggregates_summaries():
    pages = [[rec(f"2401.0000{i}"), rec(f"2401.0000{i + 1}")] for i in range(1, 6, 2)]  # 3 pages
    repo = FakeRepo(
        results=[
            IngestSummary(new=2),
            IngestSummary(new=1, updated=1),
            IngestSummary(updated=2),
        ]
    )

    summary = IngestPipeline(FakeClient(pages), repo).run(["cs.AI"], limit=6, page_size=2)

    assert [len(batch) for batch in repo.batches] == [2, 2, 2]  # committed page by page
    assert summary == IngestSummary(new=3, updated=3, unchanged=0)


def test_mid_ingest_failure_keeps_committed_pages():
    client = FakeClient([[rec("2401.00001"), rec("2401.00002")], [rec("2401.00003")]])
    client.explode_after_pages(1)
    repo = FakeRepo()

    with pytest.raises(RuntimeError, match="arXiv exploded"):
        IngestPipeline(client, repo).run(["cs.AI"], limit=10, page_size=2)

    assert len(repo.batches) == 1  # first page was already committed
    assert [r.arxiv_id for r in repo.batches[0]] == ["2401.00001", "2401.00002"]


def test_no_results_returns_zero_summary():
    summary = IngestPipeline(FakeClient([]), FakeRepo()).run(["cs.AI"], limit=10)
    assert summary == IngestSummary()


def test_run_ids_fetches_the_given_papers_and_commits_per_page():
    client = FakeClient([[rec("2401.00001"), rec("2401.00002")], [rec("2401.00003")]])
    repo = FakeRepo()

    summary = IngestPipeline(client, repo).run_ids(
        ["2401.00001", "2401.00002", "2401.00003"], page_size=2
    )

    assert client.calls == [(["2401.00001", "2401.00002", "2401.00003"], 3)]
    assert [len(batch) for batch in repo.batches] == [2, 1]
    assert repo.batches[0][0].title == "Title for 2401.00001"  # cleaned like any ingest
    assert summary == IngestSummary(new=3)
