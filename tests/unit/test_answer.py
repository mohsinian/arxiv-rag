import json

import pytest

from arxiv_rag.rag.answer import AnswerGenerator, ChatClient, LLMError
from arxiv_rag.rag.llm import OpenAICompatibleChat
from arxiv_rag.rag.retrieve import RetrievalResult, RetrievedPaper

pytestmark = pytest.mark.unit


def paper(
    n: int, abstract: str = "We propose a cheap method.", score: float = 0.8
) -> RetrievedPaper:
    return RetrievedPaper(
        arxiv_id=f"2401.0000{n}",
        title=f"Paper {n}",
        abstract=abstract,
        score=score,
        published="2024-01-15",
        category="cs.LG",
        authors="A. Author, B. Author et al.",
    )


def reply_json(answer: str, cited: list[int], sufficient: bool = True) -> str:
    return json.dumps({"answer": answer, "cited_passages": cited, "sufficient_context": sufficient})


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.reply


def sources(*papers: RetrievedPaper) -> RetrievalResult:
    return RetrievalResult(sources=list(papers), matched=bool(papers))


# -- prompt shape ---------------------------------------------------------------


def test_system_prompt_states_sample_rules_and_untrusted_data():
    llm = FakeLLM(reply_json("a [1]", [1]))
    AnswerGenerator(llm).answer(sources(paper(1)), "q")
    system, _user = llm.prompts[0]
    assert "sample" in system  # passages are a sample, not the dataset
    assert "untrusted" in system.lower()  # never follow instructions in passages
    assert "never guess" in system.lower() or "do not guess" in system.lower()
    assert "part" in system.lower()  # partial answers allowed


def test_user_prompt_wraps_passages_in_xml_with_metadata():
    llm = FakeLLM(reply_json("a [1]", [1]))
    AnswerGenerator(llm).answer(sources(paper(1)), "What is proposed?")
    _system, user = llm.prompts[0]
    assert "<passages>" in user and "</passages>" in user
    assert '<passage id="1"' in user
    assert 'published="2024-01-15"' in user
    assert 'category="cs.LG"' in user
    assert 'authors="A. Author, B. Author et al."' in user
    assert "<title>Paper 1</title>" in user
    assert "<abstract>We propose a cheap method.</abstract>" in user
    assert "<question>What is proposed?</question>" in user


def test_xml_special_characters_are_escaped():
    tricky = RetrievedPaper(
        arxiv_id="2401.00009",
        title='A & B "quotes" <study>',
        abstract="Uses <tags> & symbols.",
        score=0.7,
        published="2024-01-15",
        category="cs.CL",
        authors="C. Author",
    )
    llm = FakeLLM(reply_json("a [1]", [1]))
    AnswerGenerator(llm).answer(sources(tricky), "q")
    _system, user = llm.prompts[0]
    assert "<title>A &amp; B" in user
    assert "&lt;study&gt;" in user
    assert "<passages>" in user  # structure still intact


# -- structured output ----------------------------------------------------------


def test_json_reply_yields_answer_with_only_cited_sources():
    llm = FakeLLM(reply_json("Paper 2 proposes X [2].", cited=[2]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True
    assert result.answer == "Paper 2 proposes X [2]."
    assert [s.arxiv_id for s in result.sources] == ["2401.00002"]  # only the cited one


def test_invalid_citation_numbers_are_filtered():
    llm = FakeLLM(reply_json("a [1] and mysteriously [7].", cited=[1, 7]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert [s.arxiv_id for s in result.sources] == ["2401.00001"]  # 7 is not a passage


def test_insufficient_context_is_an_honest_no_match():
    llm = FakeLLM(reply_json("The retrieved papers do not cover this.", cited=[], sufficient=False))
    result = AnswerGenerator(llm).answer(sources(paper(1)), "q")
    assert result.matched is False
    assert result.sources == []
    assert "do not cover" in result.answer


def test_json_in_code_fences_is_parsed():
    llm = FakeLLM(f"```json\n{reply_json('a [1]', [1])}\n```")
    result = AnswerGenerator(llm).answer(sources(paper(1)), "q")
    assert result.answer == "a [1]"
    assert [s.arxiv_id for s in result.sources] == ["2401.00001"]


def test_plain_text_reply_falls_back_to_all_sources():
    llm = FakeLLM("A plain prose answer without JSON.")
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True
    assert result.answer == "A plain prose answer without JSON."
    assert len(result.sources) == 2  # conservative fallback: all provided passages


def test_no_retrieval_match_never_calls_the_llm():
    llm = FakeLLM("should not be called")
    result = AnswerGenerator(llm).answer(sources(), "anything")
    assert result.matched is False
    assert "could not find" in result.answer.lower()
    assert result.sources == []
    assert llm.prompts == []


def test_llm_failure_raises_instead_of_fabricating():
    class ExplodingLLM(FakeLLM):
        def complete(self, system, user):
            raise RuntimeError("provider down")

    with pytest.raises(LLMError, match="provider down"):
        AnswerGenerator(ExplodingLLM("x")).answer(sources(paper(1)), "q")


def test_chat_client_protocol_is_satisfied_by_minimal_fake():
    assert isinstance(FakeLLM("x"), ChatClient)


def test_chat_client_defaults_to_temperature_zero():
    client = OpenAICompatibleChat(base_url="https://api.example/v1", api_key="k", model="m")
    assert client._temperature == 0.0


# -- context budget ---------------------------------------------------------


def test_lowest_ranked_passages_are_dropped_first():
    long_abstract = "First sentence is long enough. " * 30
    llm = FakeLLM(reply_json("a [1]", [1]))
    AnswerGenerator(llm, max_context_chars=1200).answer(
        sources(paper(1, long_abstract), paper(2, long_abstract), paper(3, long_abstract)), "q"
    )
    _system, user = llm.prompts[0]
    assert '<passage id="1"' in user
    assert '<passage id="3"' not in user  # lowest-ranked dropped to fit the budget
    assert '<passage id="2"' not in user


def test_truncation_cuts_at_sentence_boundary():
    two_sentences = "The first sentence carries the key claim. The second sentence is extra detail."
    llm = FakeLLM(reply_json("a [1]", [1]))
    AnswerGenerator(llm, max_context_chars=220).answer(sources(paper(1, two_sentences)), "q")
    _system, user = llm.prompts[0]
    assert "key claim." in user
    assert "extra detail" not in user
