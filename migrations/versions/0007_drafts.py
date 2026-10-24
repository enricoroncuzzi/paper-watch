"""add the saved LinkedIn drafts

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from paperwatch.store.grants import grant_api_writes

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "drafts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=False),
        sa.Column("regenerated_from", sa.UUID(), nullable=True),
        # The owner's point of view: stored with the draft, never logged at any level.
        sa.Column("pov", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=16), nullable=False),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("hook", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("hashtags", sa.JSON(), nullable=False),
        sa.Column("source_link", sa.Text(), nullable=False),
        sa.Column("check", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["regenerated_from"], ["drafts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    # The list is newest first, and the cascade from `items` filters by item id.
    op.create_index("ix_drafts_created_at", "drafts", ["created_at"], unique=False)
    op.create_index("ix_drafts_item_id", "drafts", ["item_id"], unique=False)
    # The API saves drafts and owns neither role's table; the grant comes from the migration, so
    # the captain has no new admin step. The role is absent in CI, so the block is skipped there.
    op.execute(grant_api_writes(["drafts"]))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_drafts_item_id", table_name="drafts")
    op.drop_index("ix_drafts_created_at", table_name="drafts")
    op.drop_table("drafts")
