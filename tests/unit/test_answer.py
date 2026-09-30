import json

import pytest

from arxiv_rag.rag.answer import AnswerGenerator, ChatClient, LLMError, LLMNotConfiguredError
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
    assert result.answer == "Paper 2 proposes X [1]."  # renumbered: [1] is sources[0]
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


# -- citation numbering matches the returned sources ---------------------------


def test_citations_are_renumbered_in_order_of_first_appearance():
    llm = FakeLLM(reply_json("Method A [3] beats B [1]; see also [3].", cited=[1, 3]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2), paper(3)), "q")
    assert [s.arxiv_id for s in result.sources] == ["2401.00003", "2401.00001"]
    assert result.answer == "Method A [1] beats B [2]; see also [1]."


def test_grouped_citations_are_renumbered_together():
    llm = FakeLLM(reply_json("Both agree [4, 2].", cited=[2, 4]))
    result = AnswerGenerator(llm).answer(sources(*(paper(n) for n in range(1, 5))), "q")
    assert [s.arxiv_id for s in result.sources] == ["2401.00004", "2401.00002"]
    assert result.answer == "Both agree [1, 2]."


def test_marker_for_a_passage_that_does_not_exist_is_removed():
    llm = FakeLLM(reply_json("a [1] and mysteriously [7].", cited=[1, 7]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.answer == "a [1] and mysteriously."


def test_cited_passage_without_inline_marker_is_appended_after_marked_ones():
    llm = FakeLLM(reply_json("Mostly from [2].", cited=[1, 2]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert [s.arxiv_id for s in result.sources] == ["2401.00002", "2401.00001"]
    assert result.answer == "Mostly from [1]."


def test_inline_marker_missing_from_cited_list_still_counts_as_a_source():
    llm = FakeLLM(reply_json("Shown in [2].", cited=[]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True
    assert [s.arxiv_id for s in result.sources] == ["2401.00002"]
    assert result.answer == "Shown in [1]."


def test_bracketed_years_are_not_treated_as_citations():
    llm = FakeLLM(reply_json("Introduced in [2023] and extended [2].", cited=[2]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.answer == "Introduced in [2023] and extended [1]."


def test_fallback_sources_exclude_passages_dropped_from_the_prompt():
    long_abstract = "First sentence is long enough. " * 30
    llm = FakeLLM("Plain prose citing [1].")
    result = AnswerGenerator(llm, max_context_chars=1200).answer(
        sources(paper(1, long_abstract), paper(2, long_abstract), paper(3, long_abstract)), "q"
    )
    assert result.matched is True
    assert [s.arxiv_id for s in result.sources] == ["2401.00001"]  # only what the model saw


# -- strict grounding contract ----------------------------------------------


def test_sufficient_reply_with_no_citations_is_an_honest_no_match():
    llm = FakeLLM(reply_json("Something vague.", cited=[]))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is False
    assert result.sources == []
    assert result.answer == "Something vague."


def test_string_sufficient_context_breaks_contract_and_falls_back():
    # bool("false") is True — a non-boolean must not sneak through as sufficient
    reply = json.dumps({"answer": "a [1]", "cited_passages": [1], "sufficient_context": "false"})
    result = AnswerGenerator(FakeLLM(reply)).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True  # retrieval matched; the reply itself was unparseable
    assert len(result.sources) == 2  # conservative fallback: all provided passages


def test_missing_sufficient_context_key_breaks_contract_and_falls_back():
    reply = json.dumps({"answer": "a [1]", "cited_passages": [1]})
    result = AnswerGenerator(FakeLLM(reply)).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True
    assert len(result.sources) == 2


def test_truncated_json_falls_back_to_all_sources():
    result = AnswerGenerator(FakeLLM('{"answer": "partial", "cited_passages": [1')).answer(
        sources(paper(1), paper(2)), "q"
    )
    assert result.matched is True
    assert len(result.sources) == 2


def test_citation_to_passage_dropped_from_context_is_ignored():
    long_abstract = "First sentence is long enough. " * 30
    llm = FakeLLM(reply_json("a [1] [3].", cited=[1, 3]))
    result = AnswerGenerator(llm, max_context_chars=1200).answer(
        sources(paper(1, long_abstract), paper(2, long_abstract), paper(3, long_abstract)), "q"
    )
    assert result.matched is True
    assert [s.arxiv_id for s in result.sources] == ["2401.00001"]  # passage 3 was never shown


def test_citations_only_to_dropped_passages_is_an_honest_no_match():
    long_abstract = "First sentence is long enough. " * 30
    llm = FakeLLM(reply_json("a [3].", cited=[3]))
    result = AnswerGenerator(llm, max_context_chars=1200).answer(
        sources(paper(1, long_abstract), paper(2, long_abstract), paper(3, long_abstract)), "q"
    )
    assert result.matched is False
    assert result.sources == []


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


def test_missing_llm_credentials_propagate_as_not_configured():
    client = OpenAICompatibleChat(base_url="https://api.example/v1", api_key="", model="m")
    with pytest.raises(LLMNotConfiguredError, match="LLM_API_KEY"):
        AnswerGenerator(client).answer(sources(paper(1)), "q")


def test_no_match_needs_no_llm_credentials():
    client = OpenAICompatibleChat(base_url="https://api.example/v1", api_key="", model="m")
    result = AnswerGenerator(client).answer(sources(), "q")
    assert result.matched is False


def test_chat_client_protocol_is_satisfied_by_minimal_fake():
    assert isinstance(FakeLLM("x"), ChatClient)


def test_chat_client_defaults_to_temperature_zero_and_timeout():
    client = OpenAICompatibleChat(base_url="https://api.example/v1", api_key="k", model="m")
    assert client._temperature == 0.0
    assert client._timeout == 30.0


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


def test_insufficient_reply_with_citations_keeps_sources_but_unmatched():
    llm = FakeLLM(reply_json("Partial: it covers X [2] but not Y.", cited=[2], sufficient=False))
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is False
    assert result.answer == "Partial: it covers X [1] but not Y."  # marker still resolves
    assert [s.arxiv_id for s in result.sources] == ["2401.00002"]


def test_insufficient_reply_without_citations_strips_dangling_markers():
    llm = FakeLLM(reply_json("They do not cover it.", cited=[], sufficient=False))
    result = AnswerGenerator(llm).answer(sources(paper(1)), "q")
    assert result.matched is False
    assert result.sources == []
    assert "[" not in result.answer


def test_sufficient_but_uncited_strips_dangling_markers():
    llm = FakeLLM(reply_json("Vague claims [7].", cited=[7]))  # 7 was never shown
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is False
    assert result.sources == []
    assert result.answer == "Vague claims."


def test_fallback_drops_markers_that_cannot_resolve():
    llm = FakeLLM("Unsupported claim [7].")  # only passages 1-2 were shown
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True
    assert result.answer == "Unsupported claim."
    assert len(result.sources) == 2


def test_fallback_keeps_resolvable_markers_verbatim():
    llm = FakeLLM("Shown in [2] and also [1].")
    result = AnswerGenerator(llm).answer(sources(paper(1), paper(2)), "q")
    assert result.answer == "Shown in [2] and also [1]."
    assert [s.arxiv_id for s in result.sources] == ["2401.00001", "2401.00002"]


@pytest.mark.parametrize("cited", ["12", [1, "2"], [True]])
def test_schema_violating_cited_passages_breaks_contract(cited):
    # a string would iterate as [1, 2]; mixed/bool entries are equally invalid
    reply = json.dumps({"answer": "a [1]", "cited_passages": cited, "sufficient_context": True})
    result = AnswerGenerator(FakeLLM(reply)).answer(sources(paper(1), paper(2)), "q")
    assert result.matched is True  # fallback: retrieval matched, reply unparseable
    assert len(result.sources) == 2
