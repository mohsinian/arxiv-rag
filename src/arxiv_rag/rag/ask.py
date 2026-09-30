"""Production wiring for the /ask flow: embedder → retriever → LLM answerer."""

import logging
from collections.abc import Callable

from arxiv_rag.config import Settings
from arxiv_rag.db.repo import Repository
from arxiv_rag.rag.answer import Answered, AnswerGenerator
from arxiv_rag.rag.embeddings import EmbeddingProvider
from arxiv_rag.rag.llm import OpenAICompatibleChat
from arxiv_rag.rag.retrieve import PaperRetriever

logger = logging.getLogger(__name__)


def build_ask_handler(
    settings: Settings, embedder: EmbeddingProvider
) -> Callable[[Repository], Callable[[str], Answered]]:
    """Return a factory mapping a request-scoped Repository to an ask handler."""
    chat = OpenAICompatibleChat(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model
    )
    if chat.configured:
        logger.info("LLM: %s via %s", settings.llm_model, settings.llm_base_url)
    else:
        logger.warning(
            "LLM_API_KEY is not set: /ask still reports no-match questions, but questions "
            "with matching papers will return 503 until it is configured"
        )
    answerer = AnswerGenerator(chat, max_context_chars=settings.max_context_chars)

    def factory(repo: Repository) -> Callable[[str], Answered]:
        retriever = PaperRetriever(
            repo, embedder, top_k=settings.rag_top_k, min_score=settings.min_score
        )

        def handler(question: str) -> Answered:
            retrieval = retriever.retrieve(question)
            logger.debug(
                "retrieval: %d passage(s) above min_score %.2f (top score %s)",
                len(retrieval.sources),
                settings.min_score,
                retrieval.sources[0].score if retrieval.sources else "n/a",
            )
            return answerer.answer(retrieval, question)

        return handler

    return factory
