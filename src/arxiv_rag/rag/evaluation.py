"""Retrieval evaluation: hit@k over a hand-written question set.

The question set is deliberately tiny (15-20 questions). It is a sanity
harness that catches retrieval regressions and compares embedding models —
not a benchmark claim.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_KS = (1, 3, 5)


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> bool:
    """True if any expected arxiv_id appears within the top-k retrieved."""
    return bool(set(expected) & set(retrieved[:k]))


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected: list[str]


@dataclass(frozen=True)
class QuestionResult:
    question: EvalQuestion
    retrieved: list[str]

    def hit(self, k: int) -> bool:
        return hit_at_k(self.retrieved, self.question.expected, k)

    @property
    def is_unanswerable(self) -> bool:
        return self.question.expected == ["none"]

    def rank_of_first_hit(self) -> int | None:
        for index, arxiv_id in enumerate(self.retrieved, 1):
            if arxiv_id in self.question.expected:
                return index
        return None


@dataclass
class EvalReport:
    results: list[QuestionResult]
    ks: tuple[int, ...] = DEFAULT_KS

    @property
    def answerable(self) -> list[QuestionResult]:
        return [result for result in self.results if not result.is_unanswerable]

    def mean_hit_at(self, k: int) -> float:
        answerable = self.answerable
        if not answerable:
            return 0.0
        return sum(result.hit(k) for result in answerable) / len(answerable)

    def mrr(self) -> float:
        """Mean reciprocal rank over answerable questions (0 for a total miss)."""
        answerable = self.answerable
        if not answerable:
            return 0.0
        return sum(
            1 / rank if (rank := result.rank_of_first_hit()) else 0.0 for result in answerable
        ) / len(answerable)

    def no_match_stats(self) -> tuple[int, int]:
        """(correctly rejected, total) over unanswerable questions.

        An unanswerable question is correctly rejected when nothing above the
        similarity floor was retrieved for it.
        """
        unanswerable = [result for result in self.results if result.is_unanswerable]
        correct = sum(1 for result in unanswerable if not result.retrieved)
        return correct, len(unanswerable)

    def misses(self, k: int) -> list[QuestionResult]:
        return [result for result in self.answerable if not result.hit(k)]


def load_questions(path: Path) -> list[EvalQuestion]:
    """Load and validate the question set: non-blank questions, ≥1 expected each, unique."""
    if not path.exists():
        raise FileNotFoundError(f"question set not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    questions = raw.get("questions")
    if not questions:
        raise ValueError(f"{path}: no questions defined")

    parsed: list[EvalQuestion] = []
    seen: set[str] = set()
    for entry in questions:
        question = (entry.get("question") or "").strip()
        if not question:
            raise ValueError(f"{path}: every question needs non-blank text")
        expected = [str(arxiv_id) for arxiv_id in (entry.get("expected") or [])]
        if not expected:
            raise ValueError(f"{path}: question {question!r} needs at least one expected arxiv_id")
        if question.lower() in seen:
            raise ValueError(f"{path}: duplicate question {question!r}")
        seen.add(question.lower())
        parsed.append(EvalQuestion(question=question, expected=expected))
    return parsed


def run_eval(
    questions: list[EvalQuestion],
    retrieve,  # Callable[[str], list[str]] — top-k arxiv_ids for a question
    ks: tuple[int, ...] = DEFAULT_KS,
) -> EvalReport:
    """Retrieve once per question and score hits at each k."""
    results = [QuestionResult(question=q, retrieved=retrieve(q.question)) for q in questions]
    return EvalReport(results=results, ks=ks)
