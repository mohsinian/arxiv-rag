"""Natural-language question answering over the ingested dataset."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from arxiv_rag.api.deps import get_repo
from arxiv_rag.api.schemas import AskRequest, AskResponse
from arxiv_rag.db.repo import Repository
from arxiv_rag.rag.answer import LLMError, LLMNotConfiguredError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@router.post("/api/v1/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request, repo: Repository = Depends(get_repo)) -> AskResponse:
    """Answer a free-text question grounded in the ingested papers.

    Returns the answer, whether a good match was found, and the papers that
    grounded it; `[n]` in the answer refers to `sources[n - 1]`. When nothing
    relevant is indexed, `matched` is false and the LLM is never called.
    """
    ask_factory = getattr(request.app.state, "ask_factory", None)
    if ask_factory is None:
        raise HTTPException(status_code=503, detail="ask pipeline not configured")
    started = time.perf_counter()
    try:
        answered = ask_factory(repo)(payload.question)
    except LLMNotConfiguredError as exc:
        logger.warning("ask rejected: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        # full provider error stays in the server log; clients get a stable message
        logger.error("ask failed after %.0f ms: %s", _elapsed_ms(started), exc)
        raise HTTPException(
            status_code=502, detail="LLM provider request failed; see server logs"
        ) from exc
    logger.info(
        "ask answered in %.0f ms: matched=%s sources=%d question_chars=%d",
        _elapsed_ms(started),
        answered.matched,
        len(answered.sources),
        len(payload.question),
    )
    return AskResponse(
        answer=answered.answer,
        matched=answered.matched,
        sources=[
            {"arxiv_id": s.arxiv_id, "title": s.title, "score": s.score} for s in answered.sources
        ],
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
