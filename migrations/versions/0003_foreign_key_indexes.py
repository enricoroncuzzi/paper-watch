"""index the foreign keys on tier_history and dead_letters

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Postgres does not index foreign keys on its own; without these, reading one item's tier
    # history and the ON DELETE CASCADE from `items`/`runs` would scan the whole child table.
    op.create_index("ix_tier_history_item_id", "tier_history", ["item_id"], unique=False)
    op.create_index("ix_dead_letters_run_id", "dead_letters", ["run_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_dead_letters_run_id", table_name="dead_letters")
    op.drop_index("ix_tier_history_item_id", table_name="tier_history")
