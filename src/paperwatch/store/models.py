"""ORM tables for the single Postgres store: items, tiers, runs, dead letters, cursors."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
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

from paperwatch.config import (
    CHAT_CALL_MODEL_MAX_CHARS,
    EMBEDDING_DIM,
    HF_ORG_MAX_CHARS,
    ITEM_VENUE_MAX_CHARS,
)

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
    venue: Mapped[str | None] = mapped_column(String(ITEM_VENUE_MAX_CHARS))
    # Semantic Scholar's venue name plus alternate names, so admission can run on stored evidence.
    venue_aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    hf_upvotes: Mapped[int | None] = mapped_column(Integer)
    # The rest of what the Hugging Face listing carries: kept so the page and the ranking can use
    # it without another request. Upvotes only grow (evidence); the others are the newest
    # non-null reading of the listing.
    hf_comments: Mapped[int | None] = mapped_column(Integer)
    github_repo: Mapped[str | None] = mapped_column(Text)
    github_stars: Mapped[int | None] = mapped_column(Integer)
    project_page: Mapped[str | None] = mapped_column(Text)
    hf_org: Mapped[str | None] = mapped_column(String(HF_ORG_MAX_CHARS))
    # When the paper first appeared on the daily listing; the earliest the window has seen.
    hf_listed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
    # Audit columns, on purpose the database clock (not the run's `now`): they record the real
    # wall-clock time of the write, while every business timestamp is stamped from `now`.
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


class SignalSnapshot(Base):
    """One observation of a paper's quality signals, appended when a reading changes.

    Momentum needs a history, not the running maximum on `items`: this table turns '25 upvotes
    now' into a velocity. Rows are only written when one of the recorded numbers moves, which
    keeps a long-listed paper from adding a flat row every morning.
    """

    __tablename__ = "signal_snapshots"
    __table_args__ = (
        # Reading one item's history and the ON DELETE CASCADE from `items` both need this; Postgres
        # does not index foreign keys on its own.
        Index("ix_signal_snapshots_item_observed", "item_id", "observed_at"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"))
    # The run's `now`, not the database clock, so a test or a replay is deterministic.
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    hf_upvotes: Mapped[int | None] = mapped_column(Integer)
    hf_comments: Mapped[int | None] = mapped_column(Integer)
    github_stars: Mapped[int | None] = mapped_column(Integer)
    citation_count: Mapped[int] = mapped_column(Integer)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="running")
    counts: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Health verdict written when the run finishes:
    # {"healthy": bool, "problems": [...], "warnings": [...]}.
    health: Mapped[dict[str, object] | None] = mapped_column(JSON(none_as_null=True))
    # Set from the run's `now` when the run starts, never the database clock; the server default
    # is only a fallback for a row inserted outside the repository.
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # The wall clock read when the run ended, so finished_at - started_at is its real duration.
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


class ChatCall(Base):
    """One chat-model request, with the usage the provider reported for it.

    Rows are the durable half of the free-tier guard: the daily cap is counted from them, so it
    survives a process restart, unlike an in-memory counter. Tokens are recorded, not enforced:
    the per-answer cap is charged through the in-process Ledger.
    """

    __tablename__ = "chat_calls"
    __table_args__ = (
        # The daily cap asks "how many calls since UTC midnight" on every call, so this index
        # keeps that count a range scan instead of a growing table scan.
        Index("ix_chat_calls_at", "at"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The moment of the call, from the guard's clock (not the database clock), so a test can
    # place calls before or after a midnight boundary.
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(CHAT_CALL_MODEL_MAX_CHARS))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)


class SuggestionEvent(Base):
    """One interaction with a suggestion card, so the list can learn what was already seen.

    `dismissed` and `drafted` also take the item out of the ranking (never shown again); `shown`
    throttles repeats. The score and components are stored with the event, so a ranking can be
    audited later from the data a reader actually saw.
    """

    __tablename__ = "suggestion_events"
    __table_args__ = (
        CheckConstraint(
            "event IN ('shown', 'opened', 'dismissed', 'drafted')", name="ck_suggestion_events_kind"
        ),
        # Reading one item's events and excluding from the ranking both filter by item and kind;
        # Postgres does not index foreign keys on its own.
        Index("ix_suggestion_events_item_event", "item_id", "event"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"))
    event: Mapped[str] = mapped_column(String(16))
    # The API clock, not the database clock, so the hourly "shown" throttle is testable.
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Only a `shown` event carries these: they describe the list the reader was looking at.
    rank: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    components: Mapped[dict[str, float] | None] = mapped_column(JSON(none_as_null=True))
    list_version: Mapped[str | None] = mapped_column(String(32))


class ItemSummary(Base):
    """The stored 30-second summary of one item, keyed by the item that has exactly one.

    `text_hash` is the title plus the stored text: an item whose abstract was replaced gets a new
    summary instead of showing one written about the previous text.
    """

    __tablename__ = "item_summaries"
    __table_args__ = (
        CheckConstraint("status IN ('checked', 'excerpt')", name="ck_item_summaries_status"),
    )
    item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("items.id", ondelete="CASCADE"), primary_key=True
    )
    text_hash: Mapped[str] = mapped_column(String(64))
    # The provider's model id, so a model change is visible and a summary is regenerated with it.
    model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    # The four fields as written (or the excerpt), plus the checks that failed, so a reviewer can
    # see why a card fell back without re-running the model.
    fields: Mapped[dict[str, object]] = mapped_column(JSON)
    failed_checks: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
