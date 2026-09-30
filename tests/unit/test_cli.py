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

    def run_ids(self, arxiv_ids, page_size):
        self.id_calls = [list(arxiv_ids)]
        return IngestSummary(new=len(arxiv_ids))


class FakeRepo:
    last: "FakeRepo | None" = None
    indexed: set[str] | None = None  # None: every requested id is present

    def __init__(self, session):
        self.session = session
        self.wipe_calls: list[dict] = []
        FakeRepo.last = self

    def wipe(self, embeddings_only=False):
        self.wipe_calls.append({"embeddings_only": embeddings_only})

    def existing_arxiv_ids(self, arxiv_ids):
        return set(arxiv_ids) if self.indexed is None else self.indexed & set(arxiv_ids)


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
    FakeRepo.indexed = None
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


def test_ingest_ids_fetches_exactly_those_papers(fakes):
    result = runner.invoke(cli.app, ["ingest", "--ids", "2401.00001, 2401.00002"])

    assert result.exit_code == 0, result.output
    assert FakePipeline.last.id_calls == [["2401.00001", "2401.00002"]]
    assert FakePipeline.last.calls == []  # no category query
    assert "2 new" in result.output


def test_ingest_ids_from_fetches_eval_gold_papers(tmp_path, fakes):
    question_set = tmp_path / "questions.yaml"
    question_set.write_text(
        "questions:\n"
        "  - question: q1\n    expected: ['2401.00001', '2401.00002']\n"
        "  - question: q2\n    expected: ['2401.00002']\n"
        "  - question: sourdough?\n    expected: ['none']\n"
    )
    result = runner.invoke(cli.app, ["ingest", "--ids-from", str(question_set)])

    assert result.exit_code == 0, result.output
    assert FakePipeline.last.id_calls == [["2401.00001", "2401.00002"]]  # no "none", no dupes


def test_ingest_ids_from_missing_file_exits_nonzero(fakes):
    result = runner.invoke(cli.app, ["ingest", "--ids-from", "/nope/questions.yaml"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_wipe_calls_repo_wipe(fakes):
    result = runner.invoke(cli.app, ["wipe"])

    assert result.exit_code == 0, result.output
    assert FakeRepo.last.wipe_calls == [{"embeddings_only": False}]
    assert "wiped" in result.output


def test_wipe_embeddings_only(fakes):
    result = runner.invoke(cli.app, ["wipe", "--embeddings-only"])

    assert result.exit_code == 0, result.output
    assert FakeRepo.last.wipe_calls == [{"embeddings_only": True}]


class FakeIndexer:
    last: "FakeIndexer | None" = None

    def __init__(self, repo, embedder, expected_dim):
        self.repo = repo
        self.embedder = embedder
        self.expected_dim = expected_dim
        self.calls: list[dict] = []
        FakeIndexer.last = self

    def reindex(self, rebuild=False):
        from arxiv_rag.rag.indexer import ReindexSummary

        self.calls.append({"rebuild": rebuild})
        return ReindexSummary(embedded=7)


@pytest.fixture
def reindex_fakes(monkeypatch, fakes):
    monkeypatch.setattr(cli, "EmbeddingIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "build_embedder", lambda settings: object())


def test_reindex_builds_indexer_with_settings_dim(reindex_fakes):
    result = runner.invoke(cli.app, ["reindex"])

    assert result.exit_code == 0, result.output
    assert FakeIndexer.last.expected_dim == 768
    assert FakeIndexer.last.calls == [{"rebuild": False}]
    assert "7 embedded" in result.output


def test_reindex_rebuild_flag(reindex_fakes):
    result = runner.invoke(cli.app, ["reindex", "--rebuild"])

    assert result.exit_code == 0, result.output
    assert FakeIndexer.last.calls == [{"rebuild": True}]


def test_reindex_failure_exits_nonzero(monkeypatch, reindex_fakes):
    class ExplodingIndexer(FakeIndexer):
        def reindex(self, rebuild=False):
            raise RuntimeError("model missing")

    monkeypatch.setattr(cli, "EmbeddingIndexer", ExplodingIndexer)
    result = runner.invoke(cli.app, ["reindex"])

    assert result.exit_code == 1
    assert "model missing" in result.output


@pytest.fixture
def eval_fakes(monkeypatch, reindex_fakes):
    from arxiv_rag.rag.evaluation import EvalQuestion, EvalReport, QuestionResult

    questions = [
        EvalQuestion(question="q1", expected=["a"]),
        EvalQuestion(question="q2", expected=["b"]),
    ]
    monkeypatch.setattr(cli, "load_questions", lambda path: questions)

    def fake_run_eval(questions, retrieve, ks=(1, 3, 5)):
        retrieved = {"q1": ["a", "x"], "q2": ["x", "y"]}
        results = [QuestionResult(question=q, retrieved=retrieved[q.question]) for q in questions]
        return EvalReport(results=results, ks=ks)

    monkeypatch.setattr(cli, "run_eval", fake_run_eval)
    return questions


def test_eval_prints_hit_rates(eval_fakes):
    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0, result.output
    assert "hit@1" in result.output
    assert "50.0%" in result.output  # 1 of 2 questions hit at k=1


def test_eval_lists_misses(eval_fakes):
    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0
    assert "q2" in result.output  # the miss is shown with its expected ids


def test_eval_with_missing_file_exits_nonzero(monkeypatch, eval_fakes):
    monkeypatch.setattr(
        cli, "load_questions", lambda path: (_ for _ in ()).throw(FileNotFoundError("nope"))
    )
    result = runner.invoke(cli.app, ["eval", "/nope/path.yaml"])

    assert result.exit_code == 1
    assert "nope" in result.output


def test_eval_skips_questions_whose_gold_papers_are_not_ingested(eval_fakes):
    FakeRepo.indexed = {"a"}  # q2's gold paper "b" was never ingested
    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0, result.output
    assert "Skipped 1 question" in result.output
    assert "ingest --ids-from" in result.output
    assert "Evaluated 1 questions" in result.output
    assert "hit@1: 100.0%" in result.output  # the skipped question is not a miss
