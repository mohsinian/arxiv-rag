"""Response schemas for the stats and ask APIs."""

from pydantic import BaseModel


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
