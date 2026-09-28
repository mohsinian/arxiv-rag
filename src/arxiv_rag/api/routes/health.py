"""Liveness/readiness endpoint."""

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    """Report app and database health (used by compose healthchecks/orchestrators)."""
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        return {"status": "ok", "db": "skipped"}
    try:
        session = session_factory()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "db": "error"}) from exc
    return {"status": "ok", "db": "ok"}
