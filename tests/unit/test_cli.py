import pytest
from typer.testing import CliRunner

import arxiv_rag.cli as cli
from arxiv_rag.db.repo import IngestSummary

pytestmark = pytest.mark.unit

runner = CliRunner()


class FakePipeline:
    last: "FakePipeline | None" = None

    def __init__(self, client, repo):
        self.client = client
        self.repo = repo
        self.calls: list[tuple[list[str], int, int]] = []
        FakePipeline.last = self

    def run(self, categories, limit, page_size):
        self.calls.append((list(categories), limit, page_size))
        return IngestSummary(new=limit)


class FakeRepo:
    last: "FakeRepo | None" = None

    def __init__(self, session):
        self.session = session
        self.wipe_calls: list[dict] = []
        FakeRepo.last = self

    def wipe(self, embeddings_only=False):
        self.wipe_calls.append({"embeddings_only": embeddings_only})


class FakeClient:
    last: "FakeClient | None" = None

    def __init__(self, settings):
        self.settings = settings
        FakeClient.last = self


class FakeSession:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fakes(monkeypatch):
    FakePipeline.last = None
    FakeRepo.last = None
    FakeClient.last = None
    monkeypatch.setattr(cli, "_open_session", FakeSession)
    monkeypatch.setattr(cli, "IngestPipeline", FakePipeline)
    monkeypatch.setattr(cli, "Repository", FakeRepo)
    monkeypatch.setattr(cli, "ArxivClient", FakeClient)


def test_ingest_passes_parsed_categories_and_limit(fakes):
    result = runner.invoke(cli.app, ["ingest", "--categories", "cs.AI,cs.LG", "--limit", "500"])

    assert result.exit_code == 0, result.output
    (categories, limit, _page_size) = FakePipeline.last.calls[0]
    assert categories == ["cs.AI", "cs.LG"]
    assert limit == 500
    assert "500 new" in result.output


def test_ingest_has_sensible_defaults(fakes):
    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 0, result.output
    (categories, limit, _page_size) = FakePipeline.last.calls[0]
    assert categories == ["cs.AI", "cs.LG", "cs.CL"]
    assert limit == 1000


def test_ingest_failure_exits_nonzero(monkeypatch, fakes):
    class ExplodingPipeline(FakePipeline):
        def run(self, categories, limit, page_size):
            raise RuntimeError("network down")

    monkeypatch.setattr(cli, "IngestPipeline", ExplodingPipeline)
    result = runner.invoke(cli.app, ["ingest"])

    assert result.exit_code == 1
    assert "network down" in result.output


def test_wipe_calls_repo_wipe(fakes):
    result = runner.invoke(cli.app, ["wipe"])

    assert result.exit_code == 0, result.output
    assert FakeRepo.last.wipe_calls == [{"embeddings_only": False}]
    assert "wiped" in result.output


def test_wipe_embeddings_only(fakes):
    result = runner.invoke(cli.app, ["wipe", "--embeddings-only"])

    assert result.exit_code == 0, result.output
    assert FakeRepo.last.wipe_calls == [{"embeddings_only": True}]
