import pytest

from arxiv_rag.rag.answer import AnswerGenerator, ChatClient
from arxiv_rag.rag.retrieve import RetrievalResult, RetrievedPaper

pytestmark = pytest.mark.unit

SOURCE = RetrievedPaper(
    arxiv_id="2401.00001",
    title="Attention Is All We Need for Cheap Embeddings",
    abstract="We propose a cheap method for producing embeddings.",
    score=0.87,
)


class FakeLLM:
    """Records prompts, returns a canned answer."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.reply


def sources(*papers: RetrievedPaper) -> RetrievalResult:
    return RetrievalResult(sources=list(papers), matched=bool(papers))


def test_system_prompt_forbids_outside_knowledge_and_requires_citations():
    llm = FakeLLM("an answer")
    AnswerGenerator(llm).answer(sources(SOURCE), "What is proposed?")
    system, _user = llm.prompts[0]
    assert "only" in system.lower() and "context" in system.lower()
    assert "[1]" in system  # citation convention explained


def test_user_prompt_contains_numbered_context_blocks_and_question():
    llm = FakeLLM("an answer")
    AnswerGenerator(llm).answer(sources(SOURCE), "What is proposed?")
    _system, user = llm.prompts[0]
    assert "[1]" in user
    assert "Attention Is All We Need for Cheap Embeddings" in user
    assert "cheap method" in user
    assert "What is proposed?" in user


def test_context_is_truncated_to_max_chars():
    llm = FakeLLM("an answer")
    huge = RetrievedPaper(arxiv_id="2401.00002", title="T", abstract="x" * 100_000, score=0.5)
    AnswerGenerator(llm, max_context_chars=500).answer(sources(huge), "q")
    _system, user = llm.prompts[0]
    assert len(user) < 1500  # abstract truncated, not the whole 100k


def test_happy_path_returns_answer_and_sources():
    llm = FakeLLM("They propose a cheap method [1].")
    result = AnswerGenerator(llm).answer(sources(SOURCE), "What is proposed?")
    assert result.answer == "They propose a cheap method [1]."
    assert result.matched
    assert [s.arxiv_id for s in result.sources] == ["2401.00001"]
    assert result.sources[0].score == 0.87


def test_no_retrieval_match_never_calls_the_llm():
    llm = FakeLLM("I should not be called")
    result = AnswerGenerator(llm).answer(sources(), "anything")
    assert result.matched is False
    assert "could not find" in result.answer.lower() or "no relevant" in result.answer.lower()
    assert result.sources == []
    assert llm.prompts == []  # the LLM was never asked


def test_llm_failure_raises_instead_of_fabricating():
    class ExplodingLLM(FakeLLM):
        def complete(self, system, user):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        AnswerGenerator(ExplodingLLM("x")).answer(sources(SOURCE), "q")


def test_chat_client_protocol_is_satisfied_by_minimal_fake():
    assert isinstance(FakeLLM("x"), ChatClient)


def test_chat_client_boots_without_an_api_key_and_only_fails_on_use():
    # the API must start (health/stats) with no LLM key; only /ask may fail
    from arxiv_rag.rag.llm import OpenAICompatibleChat

    client = OpenAICompatibleChat(
        base_url="https://api.openai.com/v1", api_key="", model="gpt-4o-mini"
    )
    with pytest.raises(Exception, match="api_key"):
        client.complete("system", "user")
