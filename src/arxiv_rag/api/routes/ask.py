"""Natural-language question answering over the ingested dataset."""

from fastapi import APIRouter, Depends, HTTPException, Request

from arxiv_rag.api.deps import get_repo
from arxiv_rag.api.schemas import AskRequest, AskResponse
from arxiv_rag.db.repo import Repository
from arxiv_rag.rag.answer import LLMError

router = APIRouter(tags=["ask"])


@router.post("/api/v1/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request, repo: Repository = Depends(get_repo)) -> AskResponse:
    """Answer a free-text question grounded in the ingested papers.

    Returns the answer, whether a good match was found, and the papers that
    grounded it. When nothing relevant is indexed, `matched` is false and the
    LLM is never called.
    """
    ask_factory = getattr(request.app.state, "ask_factory", None)
    if ask_factory is None:
        raise HTTPException(status_code=503, detail="ask pipeline not configured")
    try:
        answered = ask_factory(repo)(payload.question)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(
        answer=answered.answer,
        matched=answered.matched,
        sources=[
            {"arxiv_id": s.arxiv_id, "title": s.title, "score": s.score} for s in answered.sources
        ],
    )
