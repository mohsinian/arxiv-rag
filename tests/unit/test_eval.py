from pathlib import Path

import pytest

from arxiv_rag.rag.evaluation import (
    EvalQuestion,
    EvalReport,
    QuestionResult,
    gold_ids,
    hit_at_k,
    load_questions,
    split_by_coverage,
)

pytestmark = pytest.mark.unit


# -- hit@k math ---------------------------------------------------------------


def test_hit_at_one():
    assert hit_at_k(["a", "b", "c"], expected=["a"], k=1) is True


def test_miss_at_one_but_hit_at_three():
    retrieved = ["x", "y", "a"]
    assert hit_at_k(retrieved, expected=["a"], k=1) is False
    assert hit_at_k(retrieved, expected=["a"], k=3) is True


def test_complete_miss():
    assert hit_at_k(["x", "y", "z"], expected=["a"], k=5) is False


def test_any_expected_in_top_k_counts():
    assert hit_at_k(["x", "b"], expected=["a", "b"], k=2) is True


def test_k_larger_than_retrieved_list_is_safe():
    assert hit_at_k(["a"], expected=["a"], k=10) is True
    assert hit_at_k([], expected=["a"], k=10) is False


def test_retrieved_beyond_k_is_ignored():
    assert hit_at_k(["x", "a"], expected=["a"], k=1) is False


# -- question set loading ------------------------------------------------------

VALID_YAML = """
questions:
  - question: How do agents use memory?
    expected: ["2609.30558"]
  - question: What is CARGO?
    expected: ["2609.30471", "2609.99999"]
"""


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(content)
    return path


def test_loads_valid_question_set(tmp_path):
    questions = load_questions(write(tmp_path, VALID_YAML))
    assert questions == [
        EvalQuestion(question="How do agents use memory?", expected=["2609.30558"]),
        EvalQuestion(question="What is CARGO?", expected=["2609.30471", "2609.99999"]),
    ]


def test_blank_question_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="question"):
        load_questions(write(tmp_path, "questions:\n  - question: '   '\n    expected: ['x']\n"))


def test_missing_expected_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        load_questions(write(tmp_path, "questions:\n  - question: q\n    expected: []\n"))


def test_duplicate_questions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(
            write(
                tmp_path,
                "questions:\n"
                "  - question: q\n    expected: ['a']\n"
                "  - question: q\n    expected: ['b']\n",
            )
        )


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "nope.yaml")


# -- report aggregation ---------------------------------------------------------


def make_result(question: str, retrieved: list[str], expected: list[str]) -> QuestionResult:
    return QuestionResult(
        question=EvalQuestion(question=question, expected=expected), retrieved=retrieved
    )


def test_report_aggregates_mean_hit_at_k():
    report = EvalReport(
        results=[
            make_result("q1", ["a", "x", "y"], ["a"]),  # hit@1
            make_result("q2", ["x", "y", "a"], ["a"]),  # hit@3 only
            make_result("q3", ["x", "y", "z"], ["a"]),  # never
        ]
    )
    assert report.mean_hit_at(1) == pytest.approx(1 / 3)
    assert report.mean_hit_at(3) == pytest.approx(2 / 3)
    assert report.mean_hit_at(5) == pytest.approx(2 / 3)


def test_report_ks_default_to_1_3_5():
    report = EvalReport(results=[])
    assert report.ks == (1, 3, 5)


def test_report_lists_misses():
    report = EvalReport(results=[make_result("hard question", ["x"], ["a"])])
    misses = report.misses(k=3)
    assert len(misses) == 1
    assert misses[0].question.question == "hard question"


# -- unanswerable questions + MRR ------------------------------------------------


def test_unanswerable_questions_are_excluded_from_hit_rates():
    result = QuestionResult(
        question=EvalQuestion(question="sourdough?", expected=["none"]), retrieved=["x", "y"]
    )
    hit = QuestionResult(question=EvalQuestion(question="real q", expected=["a"]), retrieved=["a"])
    report = EvalReport(results=[result, hit])
    assert report.mean_hit_at(1) == 1.0  # only the answerable question counts


def test_no_match_accuracy_counts_correctly_rejected_questions():
    rejected = QuestionResult(
        question=EvalQuestion(question="q1", expected=["none"]),
        retrieved=[],  # correctly nothing
    )
    forced = QuestionResult(
        question=EvalQuestion(question="q2", expected=["none"]), retrieved=["some-paper"]
    )
    correct, total = EvalReport(results=[rejected, forced]).no_match_stats()
    assert (correct, total) == (1, 2)


def test_mrr_is_reciprocal_rank_of_first_hit():
    results = [
        QuestionResult(question=EvalQuestion(question="q1", expected=["a"]), retrieved=["a", "b"]),
        QuestionResult(question=EvalQuestion(question="q2", expected=["b"]), retrieved=["x", "b"]),
        QuestionResult(question=EvalQuestion(question="q3", expected=["c"]), retrieved=["x", "y"]),
    ]
    assert EvalReport(results=results).mrr() == (1 / 1 + 1 / 2 + 0) / 3


# -- gold coverage (reproducible eval on any dataset) ----------------------------


def test_gold_ids_are_deduplicated_and_skip_unanswerable():
    questions = [
        EvalQuestion(question="q1", expected=["a", "b"]),
        EvalQuestion(question="q2", expected=["b", "c"]),
        EvalQuestion(question="q3", expected=["none"]),
    ]
    assert gold_ids(questions) == ["a", "b", "c"]


def test_questions_without_any_indexed_gold_are_skipped():
    covered = EvalQuestion(question="q1", expected=["a", "missing"])
    uncovered = EvalQuestion(question="q2", expected=["missing"])
    unanswerable = EvalQuestion(question="q3", expected=["none"])
    scorable, skipped = split_by_coverage([covered, uncovered, unanswerable], indexed_ids={"a"})
    assert scorable == [covered, unanswerable]  # one indexed gold is enough
    assert skipped == [uncovered]
