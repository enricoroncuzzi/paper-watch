"""ORM tables for the single Postgres store: items, tiers, runs, dead letters, cursors."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from paperwatch.config import EMBEDDING_DIM

# The only rows search may return. The HNSW index is partial on exactly this predicate, and
# Task 10 repeats it as literal SQL so the planner can match the query to the index.
ADMITTED_SQL = "status = 'admitted'"


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_items_source_external"),
        CheckConstraint(
            "status IN ('admitted', 'candidate', 'irrelevant')", name="ck_items_status"
        ),
        # A tier exists exactly when the item is admitted, and an admitted item is searchable.
        CheckConstraint(
            "(status = 'admitted') = (tier IS NOT NULL)", name="ck_items_tier_iff_admitted"
        ),
        CheckConstraint(
            "status <> 'admitted' OR embedding IS NOT NULL", name="ck_items_admitted_has_embedding"
        ),
        # The rule behind the tier is recorded exactly when there is a tier.
        CheckConstraint(
            "(tier IS NULL) = (admission_basis IS NULL)", name="ck_items_basis_iff_tier"
        ),
        Index(
            "ix_items_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=text(ADMITTED_SQL),
        ),
        Index("ix_items_tsv", "tsv", postgresql_using="gin"),
        Index("ix_items_status_published", "status", "published_at"),
        # Lets a moved feed's posts be recognised by link instead of stored twice.
        Index("ix_items_source_url", "source", "url"),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16))
    tier: Mapped[str | None] = mapped_column(String(32))
    # Which admission rule gave the current tier (AdmissionBasis), so a lab's own report
    # stays distinguishable from community-vetted work.
    admission_basis: Mapped[str | None] = mapped_column(String(32))
    profile: Mapped[str | None] = mapped_column(String(64))
    relevance: Mapped[float] = mapped_column(Float)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # arXiv version of the stored text (1 for feed items); a higher fetched version replaces it.
    version: Mapped[int] = mapped_column(Integer, default=1)
    venue: Mapped[str | None] = mapped_column(String(256))
    # Semantic Scholar's venue name plus alternate names, so admission can run on stored evidence.
    venue_aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    hf_upvotes: Mapped[int | None] = mapped_column(Integer)
    # Last time Semantic Scholar answered for this paper; drives the age-based re-check cadence.
    s2_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    licence: Mapped[str] = mapped_column(String(128))
    # Stored only at admission: a candidate or an irrelevant item never keeps a vector.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    # Failed admissions; at MAX_ITEM_ATTEMPTS the item is dead-lettered and no longer re-checked.
    admit_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title,'')), 'A') || "
            "setweight(to_tsvector('english', coalesce(text,'')), 'B')",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TierHistory(Base):
    __tablename__ = "tier_history"
    __table_args__ = (
        # Postgres does not index foreign keys on its own; without this, reading one item's tier
        # history and the ON DELETE CASCADE from `items` would scan the whole child table.
        Index("ix_tier_history_item_id", "item_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"))
    # None on first admission (from candidate, or at first sight).
    from_tier: Mapped[str | None] = mapped_column(String(32))
    to_tier: Mapped[str] = mapped_column(String(32))
    # The rule that fired; a same-tier row (from_tier == to_tier) records a basis improvement.
    basis: Mapped[str] = mapped_column(String(32))
    # The merged evidence that caused the change: venue, aliases, citations, upvotes.
    evidence: Mapped[dict[str, object]] = mapped_column(JSON)
    # Set from the run's `now`, never the database clock, so time-based tests are deterministic.
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="running")
    counts: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Health verdict written when the run finishes: {"healthy": bool, "problems": [...]}.
    health: Mapped[dict[str, object] | None] = mapped_column(JSON(none_as_null=True))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DeadLetter(Base):
    __tablename__ = "dead_letters"
    __table_args__ = (
        # Partial index for the open-dead-letter lookup; only unresolved rows are ever queried.
        Index(
            "ix_dead_letters_open",
            "source",
            "ref",
            postgresql_where=text("resolved_at IS NULL"),
        ),
        # Postgres does not index foreign keys on its own; the ON DELETE CASCADE from `runs`
        # would otherwise scan the whole child table.
        Index("ix_dead_letters_run_id", "run_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(32))
    ref: Mapped[str] = mapped_column(Text)
    error: Mapped[str] = mapped_column(Text)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # none_as_null stores a missing payload as SQL NULL. Plain JSON stores None as the JSON
    # value null, which `payload IS NOT NULL` would match, so a source-level dead letter would
    # be handed back to the pipeline as an item to retry.
    payload: Mapped[dict[str, object] | None] = mapped_column(JSON(none_as_null=True))
    # Failed attempts for an item dead letter; retries stop at MAX_ITEM_ATTEMPTS.
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SourceCursor(Base):
    """Per-source high-water mark: only advanced when that source's fetch fully succeeded."""

    __tablename__ = "source_cursors"
    key: Mapped[str] = mapped_column(String(600), primary_key=True)  # "arxiv" or "feed:<url>"
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
