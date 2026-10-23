"""add the Hugging Face extras to items and the append-only signal_snapshots table

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-23

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("items", sa.Column("hf_comments", sa.Integer(), nullable=True))
    # A stored link is only ever kept after the client's https GitHub check, so no scheme check
    # belongs in the column.
    op.add_column("items", sa.Column("github_repo", sa.Text(), nullable=True))
    op.add_column("items", sa.Column("github_stars", sa.Integer(), nullable=True))
    op.add_column("items", sa.Column("project_page", sa.Text(), nullable=True))
    op.add_column("items", sa.Column("hf_org", sa.String(length=256), nullable=True))
    op.add_column("items", sa.Column("hf_listed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_table(
        "signal_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=False),
        # The run's clock, not the database clock, so a test or a replay is deterministic.
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hf_upvotes", sa.Integer(), nullable=True),
        sa.Column("hf_comments", sa.Integer(), nullable=True),
        sa.Column("github_stars", sa.Integer(), nullable=True),
        sa.Column("citation_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # One item's history in time order, and the cascade from `items`, both read this index.
    op.create_index(
        "ix_signal_snapshots_item_observed",
        "signal_snapshots",
        ["item_id", "observed_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_signal_snapshots_item_observed", table_name="signal_snapshots")
    op.drop_table("signal_snapshots")
    op.drop_column("items", "hf_listed_at")
    op.drop_column("items", "hf_org")
    op.drop_column("items", "project_page")
    op.drop_column("items", "github_stars")
    op.drop_column("items", "github_repo")
    op.drop_column("items", "hf_comments")
