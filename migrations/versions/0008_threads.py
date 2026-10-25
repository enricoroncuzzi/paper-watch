"""add conversation threads and their finished turns

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-25

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from paperwatch.store.grants import grant_api_writes

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# What the API role may write, named once: the role test loads this module and runs
# `grant_statement()`, so a grant dropped from here fails in the test instead of on the server.
GRANTED_INSERT = ("threads", "thread_turns")
GRANTED_UPDATE = ("threads",)
GRANTED_SEQUENCES = ("thread_turns_id_seq",)


def grant_statement() -> str:
    return grant_api_writes(GRANTED_INSERT, update=GRANTED_UPDATE, sequences=GRANTED_SEQUENCES)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "threads",
        sa.Column("id", sa.UUID(), nullable=False),
        # The first question, clipped by the API; the Recent menu shows one line.
        sa.Column("title", sa.Text(), nullable=False),
        # Every alias the thread ever issued, so an earlier answer's [S#] stays citable.
        sa.Column("source_registry", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_threads_updated_at", "threads", ["updated_at"], unique=False)
    op.create_table(
        "thread_turns",
        # The id is an integer from its own sequence, so the API's USAGE grant is on a sequence
        # it can name; the unique (thread, n) is what keeps two concurrent turns from colliding.
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("thread_id", sa.UUID(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.JSON(), nullable=False),
        sa.Column("search_mode", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("thread_id", "n", name="uq_thread_turns_thread_n"),
    )
    # The API creates a thread, updates its registry and updated_at, and appends turns. The grant
    # comes from the migration, so the captain has no new admin step. The role is absent in CI.
    op.execute(grant_statement())


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("thread_turns")
    op.drop_index("ix_threads_updated_at", table_name="threads")
    op.drop_table("threads")
