"""add the suggestion events and the stored summaries

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-24

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from paperwatch.store.grants import grant_api_writes

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "suggestion_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=False),
        sa.Column("event", sa.String(length=16), nullable=False),
        # The API clock, not the database clock, so the hourly "shown" throttle is testable.
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("components", sa.JSON(), nullable=True),
        sa.Column("list_version", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "event IN ('shown', 'opened', 'dismissed', 'drafted')",
            name="ck_suggestion_events_kind",
        ),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # One item's events in time order and the cascade from `items` both read this index.
    op.create_index(
        "ix_suggestion_events_item_event",
        "suggestion_events",
        ["item_id", "event"],
        unique=False,
    )
    op.create_table(
        "item_summaries",
        sa.Column("item_id", sa.UUID(), nullable=False),
        # The hash of the title and text the summary was written from: a replaced abstract
        # invalidates the summary instead of showing one about the previous text.
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("fields", sa.JSON(), nullable=False),
        sa.Column("failed_checks", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('checked', 'excerpt')", name="ck_item_summaries_status"),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("item_id"),
    )
    # The API writes both tables and owns neither: the grants come from the migration so the
    # captain has no per-table admin step. The role is absent in CI, so the block is skipped there.
    op.execute(grant_api_writes(["suggestion_events"], sequences=["suggestion_events_id_seq"]))
    op.execute(grant_api_writes(["item_summaries"], update=["item_summaries"]))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("item_summaries")
    op.drop_index("ix_suggestion_events_item_event", table_name="suggestion_events")
    op.drop_table("suggestion_events")
