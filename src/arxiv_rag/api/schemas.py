"""Response schemas for the stats and ask APIs."""

from pydantic import BaseModel, Field, field_validator


class MonthCount(BaseModel):
    month: str  # YYYY-MM
    count: int


class CategoryCount(BaseModel):
    category: str
    count: int


class AuthorCount(BaseModel):
    author: str
    count: int


class Summary(BaseModel):
    papers: int
    authors: int
    categories: int
    avg_authors_per_paper: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_have_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class AskSource(BaseModel):
    arxiv_id: str
    title: str
    score: float


class AskResponse(BaseModel):
    answer: str
    matched: bool
    sources: list[AskSource]
