"""Repository: every read and write the pipeline makes, in one place.

The database's constraints, not application code, make ingestion idempotent: a duplicate
`(source, external_id)` is refused by the unique key, and an admitted item without a tier
or an embedding is refused by a check constraint.
"""

import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from paperwatch.config import MAX_ITEM_ATTEMPTS, S2_CADENCE
from paperwatch.domain import AdmissionBasis, Signals, SourceItem, Tier
from paperwatch.store.models import ChatCall, DeadLetter, Item, Run, SourceCursor, TierHistory

# Semantic Scholar reports the preprint server while a paper is only a preprint. A real venue,
# once seen, is never replaced by one of these, so a temporary answer cannot erase an acceptance.
PREPRINT_VENUES = frozenset({"arxiv", "arxiv.org"})


@dataclass(frozen=True)
class StoredItem:
    """A stored item with its classification and its merged evidence."""

    id: uuid.UUID
    item: SourceItem
    status: str
    tier: Tier | None
    basis: AdmissionBasis | None
    profile: str | None
    relevance: float
    signals: Signals


def _is_preprint(venue: str | None) -> bool:
    return venue is not None and venue.strip().lower() in PREPRINT_VENUES


def _signals(row: Item) -> Signals:
    return Signals(
        venue=row.venue,
        venue_aliases=tuple(row.venue_aliases or ()),
        citation_count=row.citation_count,
        hf_upvotes=row.hf_upvotes,
    )


def _stored(row: Item) -> StoredItem:
    # Categories are not stored; admission does not read them.
    item = SourceItem(
        row.source,
        row.external_id,
        row.title,
        tuple(row.authors),
        row.text,
        row.url,
        row.published_at,
        licence=row.licence,
        version=row.version,
    )
    return StoredItem(
        row.id,
        item,
        row.status,
        Tier(row.tier) if row.tier else None,
        AdmissionBasis(row.admission_basis) if row.admission_basis else None,
        row.profile,
        row.relevance,
        _signals(row),
    )


class Repository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def existing_ids(self, source: str, external_ids: list[str]) -> set[str]:
        if not external_ids:
            return set()
        rows = self._s.scalars(
            select(Item.external_id).where(
                Item.source == source, Item.external_id.in_(external_ids)
            )
        )
        return set(rows)

    def record_classified(
        self, item: SourceItem, status: str, profile: str | None, relevance: float
    ) -> uuid.UUID:
        values = dict(
            source=item.source,
            external_id=item.external_id,
            status=status,
            tier=None,
            profile=profile,
            relevance=relevance,
            title=item.title,
            authors=list(item.authors),
            text=item.text,
            url=item.url,
            published_at=item.published_at,
            version=item.version,
            venue_aliases=[],
            citation_count=0,
            licence=item.licence,
            admit_attempts=0,
        )
        # The unique key refuses a duplicate even if two runs race; return the existing id.
        stmt = (
            insert(Item)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_items_source_external")
            .returning(Item.id)
        )
        new_id = self._s.scalar(stmt)
        if new_id is None:
            new_id = self._s.scalar(
                select(Item.id).where(
                    Item.source == item.source, Item.external_id == item.external_id
                )
            )
        self._s.flush()
        assert new_id is not None
        return new_id

    def has_item_url(self, source: str, url: str) -> bool:
        return (
            self._s.scalar(select(Item.id).where(Item.source == source, Item.url == url).limit(1))
            is not None
        )

    def stored(self, source: str, external_id: str) -> StoredItem | None:
        row = self._s.scalar(
            select(Item).where(Item.source == source, Item.external_id == external_id)
        )
        return _stored(row) if row else None

    def merge_signals(self, item_id: uuid.UUID, fresh: Signals) -> Signals:
        row = self._s.get_one(Item, item_id)
        # Take a non-empty venue, but never let a preprint server replace a real venue.
        if fresh.venue and (
            not _is_preprint(fresh.venue) or not row.venue or _is_preprint(row.venue)
        ):
            row.venue, row.venue_aliases = fresh.venue, list(fresh.venue_aliases)
        # Counts only grow: a lower or missing answer is an outage or a lag, not lost evidence.
        row.citation_count = max(row.citation_count, fresh.citation_count)
        if fresh.hf_upvotes is not None:
            row.hf_upvotes = max(row.hf_upvotes or 0, fresh.hf_upvotes)
        self._s.flush()
        return _signals(row)

    def admit_item(
        self,
        item_id: uuid.UUID,
        tier: Tier,
        basis: AdmissionBasis,
        embedding: list[float] | None,
        evidence: Signals,
        at: datetime,
    ) -> None:
        row = self._s.get_one(Item, item_id)
        # An irrelevant item was classified off-topic on purpose; admitting it would put it back
        # into search. The only caller admits candidates and vetted preprints, so a code guard is
        # proportionate (a database trigger is deliberately not used).
        if row.status == "irrelevant":
            raise ValueError(f"{row.external_id} was classified irrelevant and cannot be admitted")
        if embedding is None and row.embedding is None:
            raise ValueError(f"first admission of {row.external_id} needs an embedding")
        self._s.add(
            TierHistory(
                item_id=row.id,
                from_tier=row.tier,
                to_tier=tier.value,
                basis=basis.value,
                evidence={
                    "venue": evidence.venue,
                    "venue_aliases": list(evidence.venue_aliases),
                    "citation_count": evidence.citation_count,
                    "hf_upvotes": evidence.hf_upvotes,
                },
                changed_at=at,
            )
        )
        row.status, row.tier, row.admission_basis = "admitted", tier.value, basis.value
        if embedding is not None:
            row.embedding = embedding
        # The cap counts consecutive failures: a success starts the count again.
        row.admit_attempts, row.last_error = 0, None
        self._s.flush()

    def upsert_item(
        self,
        item: SourceItem,
        tier: Tier,
        profile: str,
        relevance: float,
        signals: Signals,
        embedding: list[float],
        *,
        basis: AdmissionBasis = AdmissionBasis.SEED,
    ) -> uuid.UUID:
        # Seeding helper for tests and evals: the same write path the pipeline uses.
        item_id = self.record_classified(item, "candidate", profile, relevance)
        row = self._s.get_one(Item, item_id)
        if row.status != "admitted":
            merged = self.merge_signals(item_id, signals)
            self.admit_item(item_id, tier, basis, embedding, merged, item.published_at)
        return item_id

    def replace_text(
        self, item_id: uuid.UUID, item: SourceItem, embedding: list[float] | None
    ) -> None:
        # The caller embeds first, so this write happens only once the new embedding exists.
        row = self._s.get_one(Item, item_id)
        row.title, row.text, row.url, row.version = item.title, item.text, item.url, item.version
        if embedding is not None:
            row.embedding = embedding
        self._s.flush()

    def record_admit_failure(self, item_id: uuid.UUID, error: str) -> int:
        row = self._s.get_one(Item, item_id)
        row.admit_attempts += 1
        row.last_error = error[:4000]
        self._s.flush()
        return row.admit_attempts

    def recheck_targets(
        self,
        now: datetime,
        horizon: timedelta,
        also: Collection[uuid.UUID] = (),
        paper_limit: int | None = None,
    ) -> list[StoredItem]:
        live = Item.admit_attempts < MAX_ITEM_ATTEMPTS
        also_ids = list(also)
        # Papers whose evidence can still change their tier, while it plausibly still changes.
        papers = and_(
            Item.source == "arxiv",
            Item.published_at >= now - horizon,
            or_(Item.status == "candidate", Item.tier == Tier.VETTED_PREPRINT.value),
        )
        # Keyless cadence: the interval grows with the paper's age. A paper older than the last
        # band has no interval (NULL), so it is never due.
        interval = case(
            *((Item.published_at >= now - max_age, every) for max_age, every in S2_CADENCE)
        )
        # An age that matches no band has no interval, so the paper is never due, even when it
        # was never checked. A checked paper could not satisfy the comparison against NULL anyway.
        due = and_(
            interval.is_not(None),
            or_(Item.s2_checked_at.is_(None), Item.s2_checked_at + interval <= now),
        )
        # Most overdue first, so a capped run reaches them and carries the rest to the next run.
        stmt = (
            select(Item)
            .where(papers, due, live)
            .order_by(Item.s2_checked_at.asc().nulls_first(), Item.published_at, Item.id)
        )
        if also_ids:
            stmt = stmt.where(Item.id.not_in(also_ids))  # always due, and outside the cap
        if paper_limit is not None:
            stmt = stmt.limit(paper_limit)
        rows = list(self._s.scalars(stmt))
        # A relevant feed item needs no evidence: it is a candidate only after an interrupted
        # admission, so it is retried whatever its age.
        posts = and_(Item.source == "feed", Item.status == "candidate")
        extra = or_(posts, Item.id.in_(also_ids)) if also_ids else posts
        rows += self._s.scalars(select(Item).where(extra, live).order_by(Item.published_at))
        return [_stored(r) for r in rows]

    def mark_s2_checked(self, item_ids: Collection[uuid.UUID], at: datetime) -> None:
        # Only for papers whose Semantic Scholar chunk was answered; a refused one stays due.
        if item_ids:
            self._s.execute(
                update(Item).where(Item.id.in_(list(item_ids))).values(s2_checked_at=at)
            )
            self._s.flush()

    def fail_open_runs(self, *, at: datetime) -> None:
        # A single-instance daily job closes rows a crash (`kill -9`) left 'running', so the
        # run table never accumulates rows that look in-flight forever. The close is stamped
        # from the run's `now`, like every other business timestamp.
        self._s.execute(
            update(Run).where(Run.status == "running").values(status="failed", finished_at=at)
        )
        self._s.flush()

    def start_run(self, window_start: datetime, window_end: datetime, *, at: datetime) -> uuid.UUID:
        # started_at is the run's `now`, not the database clock: time enters only through `now`.
        run = Run(window_start=window_start, window_end=window_end, status="running", started_at=at)
        self._s.add(run)
        self._s.flush()
        return run.id

    def finish_run(
        self,
        run_id: uuid.UUID,
        status: str,
        counts: dict[str, int],
        tokens: int,
        health: dict[str, object] | None = None,
        *,
        at: datetime,
    ) -> None:
        run = self._s.get_one(Run, run_id)
        run.status, run.counts, run.tokens, run.health = status, counts, tokens, health
        run.finished_at = at
        self._s.flush()

    def cursor(self, key: str) -> datetime | None:
        return self._s.scalar(select(SourceCursor.window_end).where(SourceCursor.key == key))

    def advance_cursor(self, key: str, window_end: datetime) -> None:
        stmt = (
            insert(SourceCursor)
            .values(key=key, window_end=window_end)
            .on_conflict_do_update(
                index_elements=[SourceCursor.key], set_={"window_end": window_end}
            )
        )
        self._s.execute(stmt)
        self._s.flush()

    def add_dead_letter(
        self,
        run_id: uuid.UUID,
        source: str,
        ref: str,
        error: str,
        *,
        window: tuple[datetime, datetime] | None = None,
        payload: dict[str, object] | None = None,
        attempts: int = 1,
    ) -> None:
        self._s.add(
            DeadLetter(
                run_id=run_id,
                source=source,
                ref=ref,
                error=error[:4000],
                window_start=window[0] if window else None,
                window_end=window[1] if window else None,
                payload=payload,
                attempts=attempts,
            )
        )
        self._s.flush()

    def open_item_dead_letters(self) -> list[tuple[int, dict[str, object]]]:
        rows = self._s.execute(
            select(DeadLetter.id, DeadLetter.payload).where(
                DeadLetter.payload.is_not(None),
                DeadLetter.resolved_at.is_(None),
                DeadLetter.attempts < MAX_ITEM_ATTEMPTS,
            )
        ).all()
        return [(r.id, r.payload) for r in rows]

    def open_item_letter(
        self, source: str, external_id: str
    ) -> tuple[int, dict[str, object]] | None:
        # The open parked letter for one item, if any; a given-up letter (at the attempt cap)
        # is not one, so a human still owns it.
        row = self._s.execute(
            select(DeadLetter.id, DeadLetter.payload)
            .where(
                DeadLetter.source == source,
                DeadLetter.ref == external_id,
                DeadLetter.payload.is_not(None),
                DeadLetter.resolved_at.is_(None),
                DeadLetter.attempts < MAX_ITEM_ATTEMPTS,
            )
            .limit(1)
        ).first()
        return (row.id, row.payload) if row is not None else None

    def refresh_dead_letter_payload(self, dead_letter_id: int, payload: dict[str, object]) -> None:
        # A later day in the same catch-up can carry a newer version of a parked paper; keep the
        # stored payload at the newest text, so a crash before phase 2 resumes that, not the old.
        self._s.execute(
            update(DeadLetter).where(DeadLetter.id == dead_letter_id).values(payload=payload)
        )
        self._s.flush()

    def record_retry_failure(self, dead_letter_id: int, error: str) -> int:
        dl = self._s.get_one(DeadLetter, dead_letter_id)
        dl.attempts += 1
        dl.error = error[:4000]
        self._s.flush()
        return dl.attempts

    def has_open_dead_letter(self, source: str, ref: str) -> bool:
        return (
            self._s.scalar(
                select(DeadLetter.id)
                .where(
                    DeadLetter.source == source,
                    DeadLetter.ref == ref,
                    DeadLetter.resolved_at.is_(None),
                )
                .limit(1)
            )
            is not None
        )

    def resolve_dead_letter(self, dead_letter_id: int, *, at: datetime) -> None:
        dl = self._s.get_one(DeadLetter, dead_letter_id)
        dl.resolved_at = at
        # Resolving means "fixed, try again": a given-up stored item re-enters the re-check.
        item = self._s.scalar(
            select(Item).where(Item.source == dl.source, Item.external_id == dl.ref)
        )
        if item is not None:
            item.admit_attempts, item.last_error = 0, None
        self._s.flush()

    def resolve_open_dead_letters(self, source: str, ref: str, *, at: datetime) -> None:
        # A recovered source (or a healthy signal source) closes its own outage letters. Only
        # letters with a null payload for this exact (source, ref) match: an item letter carries
        # a payload, and a feed's malformed-entry letters have their own refs, so neither is
        # closed here. Unlike resolve_dead_letter, this never touches a stored item.
        self._s.execute(
            update(DeadLetter)
            .where(
                DeadLetter.source == source,
                DeadLetter.ref == ref,
                DeadLetter.resolved_at.is_(None),
                DeadLetter.payload.is_(None),
            )
            .values(resolved_at=at)
        )
        self._s.flush()

    def record_chat_call(
        self,
        *,
        at: datetime,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        # One row per request: the daily cap counts these rows, so a restart cannot forget a call
        # that was already made. `at` comes from the caller's clock, not the database's. This
        # flushes only; the production guard writes through `ChatCallLog`, which commits.
        self._s.add(
            ChatCall(
                at=at,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        self._s.flush()

    def chat_calls_since(self, at: datetime) -> int:
        # The cap is per UTC day, so the caller passes that day's midnight and this counts only
        # the calls at or after it; the ix_chat_calls_at index keeps it a range scan.
        return (
            self._s.scalar(select(func.count()).select_from(ChatCall).where(ChatCall.at >= at)) or 0
        )


class ChatCallLog:
    """The production `ChatCallStore`: every usage row is committed on its own short session.

    The request's own session cannot carry these rows: a failed answer or a client that hangs up
    rolls it back, and the calls it already spent against the free tier would vanish from the
    daily cap (ADR 0006). A row also becomes visible to other requests at once, instead of at the
    end of a long answer, which keeps the cap accurate when two questions run together.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def record_chat_call(
        self,
        *,
        at: datetime,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        # Its own transaction, committed before returning: this is the whole point of the class.
        with self._session_factory() as session:
            Repository(session).record_chat_call(
                at=at,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            session.commit()

    def chat_calls_since(self, at: datetime) -> int:
        # A read-only session is enough, and closing it releases the connection at once.
        with self._session_factory() as session:
            return Repository(session).chat_calls_since(at)
