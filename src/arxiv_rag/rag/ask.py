"""Production wiring for the /ask flow: embedder → retriever → LLM answerer."""

from collections.abc import Callable

from arxiv_rag.config import Settings
from arxiv_rag.db.repo import Repository
from arxiv_rag.rag.answer import Answered, AnswerGenerator
from arxiv_rag.rag.embeddings import EmbeddingProvider
from arxiv_rag.rag.llm import OpenAICompatibleChat
from arxiv_rag.rag.retrieve import PaperRetriever


def build_ask_handler(
    settings: Settings, embedder: EmbeddingProvider
) -> Callable[[Repository], Callable[[str], Answered]]:
    """Return a factory mapping a request-scoped Repository to an ask handler."""
    chat = OpenAICompatibleChat(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model
    )
    answerer = AnswerGenerator(chat, max_context_chars=settings.max_context_chars)

    def factory(repo: Repository) -> Callable[[str], Answered]:
        retriever = PaperRetriever(
            repo, embedder, top_k=settings.rag_top_k, min_score=settings.min_score
        )

        def handler(question: str) -> Answered:
            retrieval = retriever.retrieve(question)
            return answerer.answer(retrieval, question)

        return handler

    return factory
