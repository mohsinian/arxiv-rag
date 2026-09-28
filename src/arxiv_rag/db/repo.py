"""Persistence operations: idempotent upserts, wipes, and aggregations."""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import String, delete, func, or_, select
from sqlalchemy.orm import Session

from arxiv_rag.db import models
from arxiv_rag.ingest.parser import PaperRecord

logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    new: int = 0
    updated: int = 0
    unchanged: int = 0

    def __str__(self) -> str:
        return f"{self.new} new, {self.updated} updated, {self.unchanged} unchanged"

    def __add__(self, other: "IngestSummary") -> "IngestSummary":
        return IngestSummary(
            new=self.new + other.new,
            updated=self.updated + other.updated,
            unchanged=self.unchanged + other.unchanged,
        )


@dataclass
class _LookupCaches:
    categories: dict[str, models.Category] = field(default_factory=dict)
    authors: dict[str, models.Author] = field(default_factory=dict)


class Repository:
    """All database reads/writes for the dataset live here."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_records(self, records: Sequence[PaperRecord]) -> IngestSummary:
        """Insert or update papers; one transaction for the whole batch.

        Re-running ingestion is safe: identical records are skipped, changed
        ones have their fields and author/category links replaced (never
        duplicated). A failing record rolls back the entire batch so a
        partial page can never leave half-written rows behind.
        """
        summary = IngestSummary()
        caches = _LookupCaches()
        try:
            for record_ in records:
                outcome = self._upsert_one(record_, caches)
                setattr(summary, outcome, getattr(summary, outcome) + 1)
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.exception("batch of %d records rolled back", len(records))
            raise
        if records:
            logger.info("upsert batch: %s", summary)
        return summary

    def wipe(self, embeddings_only: bool = False) -> None:
        """Delete everything, or just the embeddings (for a re-index)."""
        if embeddings_only:
            self._session.execute(delete(models.PaperEmbedding))
        else:
            # FK-safe order: dependents first
            self._session.execute(delete(models.PaperEmbedding))
            self._session.execute(delete(models.paper_authors))
            self._session.execute(delete(models.paper_categories))
            self._session.execute(delete(models.Paper))
            self._session.execute(delete(models.Author))
            self._session.execute(delete(models.Category))
        self._session.commit()
        logger.info("wipe complete (embeddings_only=%s)", embeddings_only)

    # -- aggregations (served by the stats API; always computed in SQL) ------

    def papers_by_month(self, category: str | None = None, months: int | None = None) -> list[dict]:
        """Paper counts grouped by publication month (YYYY-MM), ascending.

        `months` keeps only papers published within roughly the last N months
        (30-day approximation). `category` filters to papers carrying that
        arXiv category (primary or cross-listed).
        """
        month = func.substr(func.cast(models.Paper.published_at, String), 1, 7).label("month")
        stmt = select(month, func.count().label("count")).group_by(month).order_by(month)
        if months is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
            stmt = stmt.where(models.Paper.published_at >= cutoff)
        if category is not None:
            stmt = (
                stmt.join(
                    models.paper_categories,
                    models.paper_categories.c.paper_id == models.Paper.id,
                )
                .join(
                    models.Category,
                    models.Category.id == models.paper_categories.c.category_id,
                )
                .where(models.Category.arxiv_category == category)
            )
        return [{"month": month_, "count": count} for month_, count in self._session.execute(stmt)]

    def top_categories(self, limit: int = 10) -> list[dict]:
        """Categories with the most papers (ties broken alphabetically)."""
        count = func.count(models.paper_categories.c.paper_id).label("count")
        stmt = (
            select(models.Category.arxiv_category.label("category"), count)
            .join(
                models.paper_categories,
                models.paper_categories.c.category_id == models.Category.id,
            )
            .group_by(models.Category.arxiv_category)
            .order_by(
                func.count(models.paper_categories.c.paper_id).desc(),
                models.Category.arxiv_category,
            )
            .limit(limit)
        )
        return [dict(row._mapping) for row in self._session.execute(stmt)]

    def top_authors(self, limit: int = 10) -> list[dict]:
        """Authors with the most papers (ties broken alphabetically)."""
        count = func.count(models.paper_authors.c.paper_id).label("count")
        stmt = (
            select(models.Author.name.label("author"), count)
            .join(models.paper_authors, models.paper_authors.c.author_id == models.Author.id)
            .group_by(models.Author.name)
            .order_by(
                func.count(models.paper_authors.c.paper_id).desc(),
                models.Author.name,
            )
            .limit(limit)
        )
        return [dict(row._mapping) for row in self._session.execute(stmt)]

    def summary(self) -> dict:
        """Headline counts: papers, authors, categories, avg authors per paper."""
        papers = self._session.scalar(select(func.count()).select_from(models.Paper)) or 0
        authors = (
            self._session.scalar(
                select(func.count(func.distinct(models.paper_authors.c.author_id)))
            )
            or 0
        )
        categories = (
            self._session.scalar(
                select(func.count(func.distinct(models.paper_categories.c.category_id)))
            )
            or 0
        )
        links = self._session.scalar(select(func.count()).select_from(models.paper_authors)) or 0
        avg = round(links / papers, 2) if papers else 0
        return {
            "papers": papers,
            "authors": authors,
            "categories": categories,
            "avg_authors_per_paper": avg,
        }

    # -- embeddings -----------------------------------------------------------

    def papers_needing_embedding(self) -> Iterator[tuple[int, str, str]]:
        """Papers with no embedding row, or whose text changed after embedding."""
        rows = self._session.execute(
            select(models.Paper.id, models.Paper.title, models.Paper.abstract)
            .join(
                models.PaperEmbedding,
                models.PaperEmbedding.paper_id == models.Paper.id,
                isouter=True,
            )
            .where(
                or_(
                    models.PaperEmbedding.paper_id.is_(None),
                    models.Paper.updated_at > models.PaperEmbedding.embedded_at,
                )
            )
        ).all()
        return iter(rows)

    def store_embeddings(self, rows: Sequence[tuple[int, list[float], str]]) -> None:
        """Replace embedding rows in one transaction: (paper_id, vector, model)."""
        try:
            for paper_id, vector, model in rows:
                self._session.execute(
                    delete(models.PaperEmbedding).where(models.PaperEmbedding.paper_id == paper_id)
                )
                self._session.add(
                    models.PaperEmbedding(paper_id=paper_id, embedding=vector, model=model)
                )
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.exception("embedding batch of %d rows rolled back", len(rows))
            raise

    def execute_similar_papers(self, stmt, query_vector: Sequence[float]):
        """Run a similarity query with the query vector bound as a parameter."""
        return self._session.execute(stmt, {"query_vector": query_vector}).all()

    # -- upsert internals ---------------------------------------------------

    def _upsert_one(self, record_: PaperRecord, caches: _LookupCaches) -> str:
        primary = self._get_or_create_category(record_.primary_category, caches)
        paper = self._session.execute(
            select(models.Paper).where(models.Paper.arxiv_id == record_.arxiv_id)
        ).scalar_one_or_none()

        if paper is None:
            self._insert_paper(record_, primary, caches)
            return "new"

        if self._paper_is_current(paper, record_):
            return "unchanged"

        self._update_paper(paper, record_, primary, caches)
        return "updated"

    def _insert_paper(
        self, record_: PaperRecord, primary: models.Category, caches: _LookupCaches
    ) -> None:
        paper = models.Paper(
            arxiv_id=record_.arxiv_id,
            title=record_.title,
            abstract=record_.abstract,
            published_at=record_.published,
            updated_at=record_.updated,
            doi=record_.doi,
            journal_ref=record_.journal_ref,
            primary_category_id=primary.id,
        )
        self._session.add(paper)
        self._session.flush()  # need paper.id for the join rows
        self._write_join_rows(paper, record_, caches)

    def _update_paper(
        self,
        paper: models.Paper,
        record_: PaperRecord,
        primary: models.Category,
        caches: _LookupCaches,
    ) -> None:
        paper.title = record_.title
        paper.abstract = record_.abstract
        paper.published_at = record_.published
        paper.updated_at = record_.updated
        paper.doi = record_.doi
        paper.journal_ref = record_.journal_ref
        paper.primary_category_id = primary.id

        self._session.execute(
            delete(models.paper_authors).where(models.paper_authors.c.paper_id == paper.id)
        )
        self._session.execute(
            delete(models.paper_categories).where(models.paper_categories.c.paper_id == paper.id)
        )
        # the text changed, so any stored embedding is stale — drop it; the
        # next reindex pass re-embeds papers without a row
        self._session.execute(
            delete(models.PaperEmbedding).where(models.PaperEmbedding.paper_id == paper.id)
        )
        self._write_join_rows(paper, record_, caches)

    def _write_join_rows(
        self, paper: models.Paper, record_: PaperRecord, caches: _LookupCaches
    ) -> None:
        # dict.fromkeys: the same author listed twice in one feed entry must not
        # violate the (paper_id, author_id) primary key
        for position, name in enumerate(dict.fromkeys(record_.authors)):
            author = self._get_or_create_author(name, caches)
            self._session.execute(
                models.paper_authors.insert().values(
                    paper_id=paper.id, author_id=author.id, position=position
                )
            )
        for category_name in dict.fromkeys(record_.categories):
            category = self._get_or_create_category(category_name, caches)
            self._session.execute(
                models.paper_categories.insert().values(paper_id=paper.id, category_id=category.id)
            )

    def _paper_is_current(self, paper: models.Paper, record_: PaperRecord) -> bool:
        stored_authors = list(
            self._session.execute(
                select(models.Author.name)
                .join(models.paper_authors, models.paper_authors.c.author_id == models.Author.id)
                .where(models.paper_authors.c.paper_id == paper.id)
                .order_by(models.paper_authors.c.position)
            ).scalars()
        )
        stored_categories = set(
            self._session.execute(
                select(models.Category.arxiv_category)
                .join(
                    models.paper_categories,
                    models.paper_categories.c.category_id == models.Category.id,
                )
                .where(models.paper_categories.c.paper_id == paper.id)
            ).scalars()
        )
        stored_primary = self._session.get(models.Category, paper.primary_category_id)
        return (
            paper.title == record_.title
            and paper.abstract == record_.abstract
            and paper.doi == record_.doi
            and paper.journal_ref == record_.journal_ref
            and paper.published_at == record_.published
            and paper.updated_at == record_.updated
            and stored_authors == list(record_.authors)
            and stored_categories == set(record_.categories)
            and stored_primary is not None
            and stored_primary.arxiv_category == record_.primary_category
        )

    def _get_or_create_category(self, name: str, caches: _LookupCaches) -> models.Category:
        if name not in caches.categories:
            found = self._session.execute(
                select(models.Category).where(models.Category.arxiv_category == name)
            ).scalar_one_or_none()
            caches.categories[name] = found or self._persist(models.Category(arxiv_category=name))
        return caches.categories[name]

    def _get_or_create_author(self, name: str, caches: _LookupCaches) -> models.Author:
        if name not in caches.authors:
            found = self._session.execute(
                select(models.Author).where(models.Author.name == name)
            ).scalar_one_or_none()
            caches.authors[name] = found or self._persist(models.Author(name=name))
        return caches.authors[name]

    def _persist(self, obj):
        self._session.add(obj)
        self._session.flush()
        return obj
