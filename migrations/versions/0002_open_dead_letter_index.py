"""add a partial index for open dead letters

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Pipeline runs look up open letters by (source, ref); a partial index keeps it small.
    op.create_index(
        "ix_dead_letters_open",
        "dead_letters",
        ["source", "ref"],
        unique=False,
        postgresql_where=sa.text("resolved_at IS NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_dead_letters_open", table_name="dead_letters")
