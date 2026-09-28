"""Aggregation endpoints for charting."""

from fastapi import APIRouter, Depends, Query

from arxiv_rag.api.deps import get_repo
from arxiv_rag.api.schemas import MonthCount
from arxiv_rag.db.repo import Repository

router = APIRouter(prefix="/api/v1/stats", tags=["stats"])


@router.get("/papers-by-month", response_model=list[MonthCount])
def papers_by_month(
    category: str | None = Query(None, description="Filter by arXiv category, e.g. cs.AI"),
    months: int | None = Query(None, ge=1, le=600, description="Only the last N months"),
    repo: Repository = Depends(get_repo),
) -> list[MonthCount]:
    """Paper counts per publication month, ascending — computed in the database."""
    return [MonthCount(**row) for row in repo.papers_by_month(category=category, months=months)]
