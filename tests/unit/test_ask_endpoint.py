import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from arxiv_rag.api.app import create_app
from arxiv_rag.db.models import Base
from arxiv_rag.rag.answer import Answered, LLMError, LLMNotConfiguredError
from arxiv_rag.rag.retrieve import RetrievedPaper

pytestmark = pytest.mark.unit

SOURCE = RetrievedPaper(
    arxiv_id="2401.00001",
    title="A Paper",
    abstract="An abstract.",
    score=0.81,
)


def make_app(ask_result: Answered | Exception, calls: list[str] | None = None):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def fake_ask_factory(repo):
        def handler(question: str) -> Answered:
            if calls is not None:
                calls.append(question)
            if isinstance(ask_result, Exception):
                raise ask_result
            return ask_result

        return handler

    return TestClient(create_app(session_factory=factory, ask_factory=fake_ask_factory))


def test_ask_returns_answer_and_sources():
    answered = Answered(answer="They propose X [1].", matched=True, sources=[SOURCE])
    response = make_app(answered).post("/api/v1/ask", json={"question": "What do they propose?"})
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "answer": "They propose X [1].",
        "matched": True,
        "sources": [{"arxiv_id": "2401.00001", "title": "A Paper", "score": 0.81}],
    }


def test_ask_passes_the_question_through():
    calls: list[str] = []
    make_app(Answered(answer="a", matched=False, sources=[]), calls).post(
        "/api/v1/ask", json={"question": "What is RAG?"}
    )
    assert calls == ["What is RAG?"]


def test_ask_honest_no_match_shape():
    answered = Answered(answer="I could not find relevant papers…", matched=False, sources=[])
    response = make_app(answered).post("/api/v1/ask", json={"question": "something obscure"})
    assert response.status_code == 200
    body = response.json()
    assert body["matched"] is False
    assert body["sources"] == []


def test_empty_question_is_rejected():
    response = make_app(Answered("a", False, [])).post("/api/v1/ask", json={"question": "   "})
    assert response.status_code == 422


def test_missing_question_is_rejected():
    response = make_app(Answered("a", False, [])).post("/api/v1/ask", json={})
    assert response.status_code == 422


def test_overlong_question_is_rejected():
    response = make_app(Answered("a", False, [])).post("/api/v1/ask", json={"question": "x" * 3000})
    assert response.status_code == 422


def test_llm_failure_maps_to_502():
    response = make_app(LLMError("provider down: 401")).post("/api/v1/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "provider" in response.json()["detail"].lower()


def test_llm_failure_details_stay_in_the_server_log(caplog):
    error = LLMError("provider down: Incorrect API key sk-live-abc123")
    with caplog.at_level("ERROR", logger="arxiv_rag.api.routes.ask"):
        response = make_app(error).post("/api/v1/ask", json={"question": "q"})
    assert "sk-live-abc123" not in response.text
    assert "sk-live-abc123" in caplog.text


def test_missing_llm_credentials_map_to_503_with_a_clear_message():
    error = LLMNotConfiguredError("no LLM credentials configured: set LLM_API_KEY")
    response = make_app(error).post("/api/v1/ask", json={"question": "q"})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]


def test_answered_questions_are_logged_with_outcome(caplog):
    answered = Answered(answer="X [1].", matched=True, sources=[SOURCE])
    with caplog.at_level("INFO", logger="arxiv_rag.api.routes.ask"):
        make_app(answered).post("/api/v1/ask", json={"question": "What is X?"})
    assert "matched=True sources=1" in caplog.text
