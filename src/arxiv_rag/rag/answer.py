"""Grounded answer generation: numbered context, citations, honest no-match."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from arxiv_rag.rag.retrieve import RetrievalResult

_SYSTEM_PROMPT = """You answer questions about an arXiv paper dataset.

Rules:
- Answer ONLY using the numbered context passages provided by the user.
- Cite the passage numbers you used, in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, say so plainly instead of guessing.
- Be concise: two to four sentences unless the question demands more."""


@runtime_checkable
class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class LLMError(RuntimeError):
    """Raised when the LLM provider fails; callers must not fabricate answers."""


@dataclass(frozen=True)
class Answered:
    answer: str
    matched: bool
    sources: list


class AnswerGenerator:
    """Turns a retrieval result into a grounded answer via an LLM.

    When retrieval found nothing above threshold, the LLM is never called —
    the honest "no relevant papers" answer is returned directly.
    """

    def __init__(self, llm: ChatClient, max_context_chars: int = 12_000) -> None:
        self._llm = llm
        self._max_context = max_context_chars

    def answer(self, retrieval: RetrievalResult, question: str) -> Answered:
        if not retrieval.sources:
            return Answered(
                answer=(
                    "I could not find relevant papers in the ingested dataset for that "
                    "question, so I would rather not guess. Try rephrasing, or ingest "
                    "more papers on the topic."
                ),
                matched=False,
                sources=[],
            )

        user_prompt = self._build_user_prompt(retrieval, question)
        try:
            answer = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            raise LLMError(f"LLM provider failed: {exc}") from exc
        return Answered(answer=answer.strip(), matched=True, sources=list(retrieval.sources))

    def _build_user_prompt(self, retrieval: RetrievalResult, question: str) -> str:
        blocks: list[str] = []
        used = 0
        for number, paper in enumerate(retrieval.sources, start=1):
            block = f"[{number}] {paper.title}\n{paper.abstract}"
            if used + len(block) > self._max_context and blocks:
                break
            blocks.append(block[: max(0, self._max_context - used)])
            used += len(blocks[-1])
        context = "\n\n".join(blocks)
        return f"Context passages:\n\n{context}\n\nQuestion: {question}"
