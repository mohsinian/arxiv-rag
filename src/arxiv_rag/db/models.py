"""SQLAlchemy models for the arXiv dataset.

Schema (see plan §3): papers / authors / categories with join tables, plus a
separate `paper_embeddings` table so the RAG index can be rebuilt or wiped
without touching paper rows.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768  # must match the default model (gte-modernbert-base); see plan §1.1


class Base(DeclarativeBase):
    pass


paper_authors = Table(
    "paper_authors",
    Base.metadata,
    Column("paper_id", ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
    Column("author_id", ForeignKey("authors.id", ondelete="CASCADE"), primary_key=True),
    # position preserves author order (first author = 0)
    Column("position", Integer, nullable=False),
)

paper_categories = Table(
    "paper_categories",
    Base.metadata,
    Column("paper_id", ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True),
    Column("category_id", ForeignKey("categories.id", ondelete="CASCADE"), primary_key=True),
)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    arxiv_category: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    def __repr__(self) -> str:
        return f"Category({self.arxiv_category!r})"


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)

    def __repr__(self) -> str:
        return f"Author({self.name!r})"


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (
        Index("ix_papers_published_at", "published_at"),
        Index("ix_papers_primary_category_id", "primary_category_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    doi: Mapped[str | None] = mapped_column(String(256))
    journal_ref: Mapped[str | None] = mapped_column(Text)
    primary_category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"), nullable=False
    )

    def __repr__(self) -> str:
        return f"Paper({self.arxiv_id!r})"


class PaperEmbedding(Base):
    __tablename__ = "paper_embeddings"

    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
