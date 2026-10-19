"""add the chat_calls table that backs the daily call cap

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "chat_calls",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # The call's timestamp comes from the guard's clock, so a test can place a call on either
        # side of UTC midnight; it is not the database clock like the audit columns on `items`.
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # The daily cap counts calls since UTC midnight on every request, so this range scan must not
    # degrade into a full table scan as the table grows.
    op.create_index("ix_chat_calls_at", "chat_calls", ["at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chat_calls_at", table_name="chat_calls")
    op.drop_table("chat_calls")
