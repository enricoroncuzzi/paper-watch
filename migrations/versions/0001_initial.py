"""add the items, tier history, runs, dead letters, and source cursors schema

Revision ID: 0001
Revises:
Create Date: 2026-10-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # pgvector adds the `vector` type; it must exist before the items table that uses a column.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("tier", sa.String(length=32), nullable=True),
        sa.Column("admission_basis", sa.String(length=32), nullable=True),
        sa.Column("profile", sa.String(length=64), nullable=True),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("authors", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("venue", sa.String(length=256), nullable=True),
        sa.Column("venue_aliases", sa.JSON(), nullable=False),
        sa.Column("citation_count", sa.Integer(), nullable=False),
        sa.Column("hf_upvotes", sa.Integer(), nullable=True),
        sa.Column("s2_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("licence", sa.String(length=128), nullable=False),
        # The literal 384, not the EMBEDDING_DIM constant: a migration records the schema of its
        # time, so a later dimension change is a new migration plus a re-embed, never an edit here.
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("admit_attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        # Stored, not just indexed: the generated column is what full-text search matches.
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('english', coalesce(title,'')), 'A') || "
                "setweight(to_tsvector('english', coalesce(text,'')), 'B')",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('admitted', 'candidate', 'irrelevant')", name="ck_items_status"
        ),
        sa.CheckConstraint(
            "(status = 'admitted') = (tier IS NOT NULL)", name="ck_items_tier_iff_admitted"
        ),
        sa.CheckConstraint(
            "status <> 'admitted' OR embedding IS NOT NULL",
            name="ck_items_admitted_has_embedding",
        ),
        sa.CheckConstraint(
            "(tier IS NULL) = (admission_basis IS NULL)", name="ck_items_basis_iff_tier"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "external_id", name="uq_items_source_external"),
    )
    # Partial HNSW index: search only ever returns admitted rows, so the index covers only them.
    op.create_index(
        "ix_items_embedding_hnsw",
        "items",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_where=sa.text("status = 'admitted'"),
    )
    op.create_index("ix_items_source_url", "items", ["source", "url"], unique=False)
    op.create_index("ix_items_status_published", "items", ["status", "published_at"], unique=False)
    op.create_index("ix_items_tsv", "items", ["tsv"], unique=False, postgresql_using="gin")
    op.create_table(
        "runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("counts", sa.JSON(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        # none_as_null keeps a missing health verdict as SQL NULL, not the JSON value null.
        sa.Column("health", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "source_cursors",
        sa.Column("key", sa.String(length=600), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "dead_letters",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("ref", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        # none_as_null: a source-level dead letter has no payload to retry as an item.
        sa.Column("payload", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "tier_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=False),
        sa.Column("from_tier", sa.String(length=32), nullable=True),
        sa.Column("to_tier", sa.String(length=32), nullable=False),
        sa.Column("basis", sa.String(length=32), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # The vector extension is left in place: other objects or revisions may rely on it.
    op.drop_table("tier_history")
    op.drop_table("dead_letters")
    op.drop_table("source_cursors")
    op.drop_table("runs")
    op.drop_index("ix_items_tsv", table_name="items", postgresql_using="gin")
    op.drop_index("ix_items_status_published", table_name="items")
    op.drop_index("ix_items_source_url", table_name="items")
    op.drop_index(
        "ix_items_embedding_hnsw",
        table_name="items",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_where=sa.text("status = 'admitted'"),
    )
    op.drop_table("items")
