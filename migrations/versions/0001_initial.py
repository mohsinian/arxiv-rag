"""initial schema: papers, authors, categories, join tables, embeddings

Revision ID: 0001
Revises:
Create Date: 2026-09-29

"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    # Self-contained on any fresh database (the compose init script only runs
    # on first container start); requires the pgvector extension available.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("arxiv_category", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("arxiv_category"),
    )
    op.create_table(
        "authors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "papers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("arxiv_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("doi", sa.String(length=256), nullable=True),
        sa.Column("journal_ref", sa.Text(), nullable=True),
        sa.Column("primary_category_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["primary_category_id"], ["categories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("arxiv_id"),
    )
    op.create_index("ix_papers_published_at", "papers", ["published_at"])
    op.create_index("ix_papers_primary_category_id", "papers", ["primary_category_id"])
    op.create_table(
        "paper_authors",
        sa.Column("paper_id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["author_id"], ["authors.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("paper_id", "author_id"),
    )
    op.create_index("ix_paper_authors_author_id", "paper_authors", ["author_id"])
    op.create_table(
        "paper_categories",
        sa.Column("paper_id", sa.Integer(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["category_id"], ["categories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("paper_id", "category_id"),
    )
    op.create_index("ix_paper_categories_category_id", "paper_categories", ["category_id"])
    op.create_table(
        "paper_embeddings",
        sa.Column("paper_id", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("paper_id"),
    )


def downgrade() -> None:
    op.drop_table("paper_embeddings")
    op.drop_index("ix_paper_categories_category_id", table_name="paper_categories")
    op.drop_table("paper_categories")
    op.drop_index("ix_paper_authors_author_id", table_name="paper_authors")
    op.drop_table("paper_authors")
    op.drop_index("ix_papers_primary_category_id", table_name="papers")
    op.drop_index("ix_papers_published_at", table_name="papers")
    op.drop_table("papers")
    op.drop_table("authors")
    op.drop_table("categories")
    op.execute("DROP EXTENSION IF EXISTS vector")
