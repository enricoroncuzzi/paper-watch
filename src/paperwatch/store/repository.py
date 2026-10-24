"""Repository: every read and write the pipeline makes, in one place.

The database's constraints, not application code, make ingestion idempotent: a duplicate
`(source, external_id)` is refused by the unique key, and an admitted item without a tier
or an embedding is refused by a check constraint.
"""

import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import Select, and_, case, func, or_, select, text, tuple_, update
from sqlalchemy.dialects.postgresql import distinct_on, insert
from sqlalchemy.orm import Session, load_only, sessionmaker

from paperwatch.config import (
    CHAT_CALLS_LOCK_KEY,
    HF_ORG_MAX_CHARS,
    INT4_MAX,
    ITEM_VENUE_MAX_CHARS,
    MAX_ITEM_ATTEMPTS,
    S2_CADENCE,
    S2_RECHECK_SLACK,
)
from paperwatch.domain import AdmissionBasis, HfPaper, Signals, SourceItem, Tier
from paperwatch.store.models import (
    ADMITTED_SQL,
    ChatCall,
    DeadLetter,
    Draft,
    Item,
    ItemSummary,
    Run,
    SignalSnapshot,
    SourceCursor,
    SuggestionEvent,
    Thread,
    ThreadTurn,
    TierHistory,
)
from paperwatch.suggestions.prompts import SUMMARY_PROMPT_VERSION
from paperwatch.suggestions.ranking import Candidate

if TYPE_CHECKING:
    # Only an annotation: the summary module imports LangChain, and the store must stay free of it
    # because the ingestion imports the store on every run.
    from paperwatch.suggestions.summary import Summary

# The four interaction kinds the API records; "dismissed" and "drafted" also hide the item.
SHOWN_EVENT = "shown"
OPENED_EVENT = "opened"
DISMISSED_EVENT = "dismissed"
DRAFTED_EVENT = "drafted"
SUGGESTION_EVENTS = (SHOWN_EVENT, OPENED_EVENT, DISMISSED_EVENT, DRAFTED_EVENT)

# Semantic Scholar reports the preprint server while a paper is only a preprint. A real venue,
# once seen, is never replaced by one of these, so a temporary answer cannot erase an acceptance.
PREPRINT_VENUES = frozenset({"arxiv", "arxiv.org"})
# A reserved call is on record before the provider is asked, because the daily cap counts rows;
# "reserved" and "failed" are the two non-answer model values, like the chat guard's "failed".
RESERVED_MODEL = "reserved"
FAILED_MODEL = "failed"
# A search over the corpus is read, not ranked into a list someone scrolls forever: past this the
# page offers a narrower query instead of deeper paging, which keeps the offset cursor bounded.
CORPUS_SEARCH_MAX = 200
# The item page draws the signal series as a table; thirty readings are a month of mornings.
CORPUS_SIGNALS_MAX = 30


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


@dataclass(frozen=True)
class SuggestionEventRow:
    """One interaction to record, as the suggestions route builds it."""

    item_id: uuid.UUID
    event: str
    at: datetime
    rank: int | None = None
    score: float | None = None
    components: dict[str, float] | None = None
    list_version: str | None = None


@dataclass(frozen=True)
class DraftItem:
    """One admitted item as a draft reads it: the stored text, its card facts and its candidate.

    The candidate is built by the same code path the ranking uses, so a Rising item and an early
    draft agree on `is_pick_grade` by construction rather than by two similar checks.
    """

    item_id: uuid.UUID
    title: str
    url: str
    source: str
    text: str
    org: str | None
    github_repo: str | None
    project_page: str | None
    candidate: Candidate


@dataclass(frozen=True)
class DraftRow:
    """One saved draft as the API and the CLI read it back."""

    id: uuid.UUID
    item_id: uuid.UUID
    item_title: str
    item_url: str
    created_at: datetime
    regenerated_from: uuid.UUID | None
    pov: str
    language: str
    model: str
    prompt_version: str
    inputs: Mapping[str, object]
    hook: str
    body: str
    hashtags: tuple[str, ...]
    source_link: str
    flags: tuple[Mapping[str, object], ...]
    counts: Mapping[str, tuple[int, int]]


class ThreadConflict(RuntimeError):
    """A turn cannot be saved because the thread changed after this answer loaded it.

    Two answers racing on one thread would each rewrite the registry from the aliases it started
    with, so the later save would make the earlier turn's citations point at other papers. The
    refusal is logged (the answer is already delivered) and the turn is not stored.
    """


@dataclass(frozen=True)
class ThreadTurnRow:
    """One stored turn as the API and the page read it back."""

    n: int
    question: str
    answer: Mapping[str, object]
    search_mode: str
    created_at: datetime


@dataclass(frozen=True)
class ThreadRow:
    """One conversation in the Recent menu, with how many finished turns it holds."""

    id: uuid.UUID
    title: str
    turns: int
    updated_at: datetime


@dataclass(frozen=True)
class ThreadDetailRow:
    """One conversation with the registry every turn may cite and its finished turns."""

    id: uuid.UUID
    title: str
    registry: tuple[Mapping[str, str], ...]
    turns: tuple[ThreadTurnRow, ...]


@dataclass(frozen=True)
class ItemDetail:
    """One admitted item as the agent's tools may show it to the model."""

    item_id: uuid.UUID
    title: str
    authors: tuple[str, ...]
    text: str
    url: str
    tier: str
    basis: str
    published_at: datetime
    venue: str | None
    citation_count: int
    hf_upvotes: int | None
    profile: str | None


@dataclass(frozen=True)
class CorpusItemRow:
    """One row of the browsable corpus table, as the API lists it."""

    id: uuid.UUID
    title: str
    source: str
    tier: str
    basis: str
    profile: str | None
    venue: str | None
    citation_count: int
    hf_upvotes: int | None
    github_stars: int | None
    published_at: datetime


@dataclass(frozen=True)
class CorpusPage:
    """One page of the table, plus whether another page follows it."""

    rows: tuple[CorpusItemRow, ...]
    total: int
    has_more: bool


@dataclass(frozen=True)
class CorpusItemDetail:
    """A row plus the text, the extras and the candidate the ranking would see."""

    row: CorpusItemRow
    authors: tuple[str, ...]
    url: str
    text: str
    github_repo: str | None
    project_page: str | None
    org: str | None
    # The page's `early` flag comes from the same candidate the ranking and a draft use, so the
    # dialog's warning cannot drift from what the draft itself will say.
    candidate: Candidate


@dataclass(frozen=True)
class TierChangeRow:
    """One move in an item's tier history, oldest first on the page."""

    from_tier: str | None
    to_tier: str
    basis: str
    changed_at: datetime


@dataclass(frozen=True)
class SignalPointRow:
    """One observed reading of an item's quality signals."""

    observed_at: datetime
    hf_upvotes: int | None
    hf_comments: int | None
    github_stars: int | None
    citation_count: int


def _int4(value: int | None) -> int | None:
    # The columns are int4, so a number outside that range is a bad extra that would fail the
    # whole listing write; drop it instead of costing every paper its extras.
    if value is None or not 0 <= value <= INT4_MAX:
        return None
    return value


def _is_preprint(venue: str | None) -> bool:
    return venue is not None and venue.strip().lower() in PREPRINT_VENUES


def _candidate(row: Item) -> Candidate:
    """One stored item as a ranking candidate, without its momentum history.

    The single place the item-to-candidate mapping lives, so the ranking and a draft's `early`
    flag cannot drift apart. Momentum fields are filled in by `suggestion_candidates`.
    """
    return Candidate(
        item_id=row.id,
        source=row.source,
        external_id=row.external_id,
        title=row.title,
        url=row.url,
        # The database's own constraint says an admitted item always has both.
        tier=Tier(cast(str, row.tier)),
        basis=AdmissionBasis(cast(str, row.admission_basis)),
        profile=row.profile,
        relevance=row.relevance,
        published_at=row.published_at,
        authors=tuple(row.authors or ()),
        # A lab post's only attribution is its author, which the feed client sets to the lab's own
        # name; Hugging Face has no organisation for it.
        org=row.hf_org or (row.authors[0] if row.source == "feed" and row.authors else None),
        github_repo=row.github_repo,
        github_stars=row.github_stars,
        project_page=row.project_page,
        text_chars=len(row.text or ""),
        hf_upvotes=row.hf_upvotes,
        hf_listed_at=row.hf_listed_at,
        upvote_series=(),
        upvote_baseline=None,
        promoted_to_peer_at=None,
        repo_first_seen_at=None,
        embedding=tuple(row.embedding or ()),
        venue=row.venue,
        citation_count=row.citation_count,
    )


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


def _detail(row: Item) -> ItemDetail:
    # Only admitted rows are passed in; the check constraints give them a tier and a basis, and
    # the assert narrows the columns for mypy instead of writing a fallback nobody should see.
    assert row.tier is not None and row.admission_basis is not None
    return ItemDetail(
        item_id=row.id,
        title=row.title,
        authors=tuple(row.authors),
        text=row.text,
        url=row.url,
        tier=row.tier,
        basis=row.admission_basis,
        published_at=row.published_at,
        venue=row.venue,
        citation_count=row.citation_count,
        hf_upvotes=row.hf_upvotes,
        profile=row.profile,
    )


def _draft_row(row: Draft) -> DraftRow:
    # The check is JSON, so a tuple round-trips as a list; the API's shape is (checked, passed).
    raw_counts = row.check.get("counts")
    counts = {
        str(kind): (int(values[0]), int(values[1]))
        for kind, values in (raw_counts if isinstance(raw_counts, Mapping) else {}).items()
        if isinstance(values, list | tuple) and len(values) == 2
    }
    raw_flags = row.check.get("flags")
    flags = tuple(
        flag
        for flag in (raw_flags if isinstance(raw_flags, list | tuple) else ())
        if isinstance(flag, Mapping)
    )
    return DraftRow(
        id=row.id,
        item_id=row.item_id,
        item_title=str(row.inputs.get("title") or ""),
        item_url=str(row.inputs.get("url") or ""),
        created_at=row.created_at,
        regenerated_from=row.regenerated_from,
        pov=row.pov,
        language=row.language,
        model=row.model,
        prompt_version=row.prompt_version,
        inputs=row.inputs,
        hook=row.hook,
        body=row.body,
        hashtags=tuple(row.hashtags or ()),
        source_link=row.source_link,
        flags=flags,
        counts=counts,
    )


def _thread_turn_row(row: ThreadTurn) -> ThreadTurnRow:
    return ThreadTurnRow(
        n=row.n,
        question=row.question,
        answer=row.answer,
        search_mode=row.search_mode,
        created_at=row.created_at,
    )


def _corpus_select() -> Any:
    # Returns Any: SQLAlchemy types a Select over its exact column tuple, which no single
    # annotation here can name. The columns are the table's and the item page's, and no vector:
    # a 384-dimension embedding per row would dwarf a fifty-row page.
    return select(
        Item.id,
        Item.title,
        Item.source,
        Item.tier,
        Item.admission_basis,
        Item.profile,
        Item.venue,
        Item.citation_count,
        Item.hf_upvotes,
        Item.github_stars,
        Item.published_at,
    )


def _corpus_filters(
    statement: Select[Any],
    *,
    tiers: frozenset[str] | None,
    source: str | None,
    profile: str | None,
    since: datetime | None,
) -> Select[Any]:
    # The literal predicate, not a bound parameter, so the planner can match the partial search
    # index and the (status, published_at) keyset index under a cached plan.
    statement = statement.where(text(f"items.{ADMITTED_SQL}"))
    if tiers:
        statement = statement.where(Item.tier.in_(tiers))
    if source is not None:
        statement = statement.where(Item.source == source)
    if profile is not None:
        statement = statement.where(Item.profile == profile)
    if since is not None:
        statement = statement.where(Item.published_at >= since)
    return statement


def _corpus_row(row: Any) -> CorpusItemRow:
    # An admitted row always has a tier and a basis (the check constraints say so); the assert
    # narrows the columns for mypy instead of writing a fallback nobody should ever see.
    assert row.tier is not None and row.admission_basis is not None
    return CorpusItemRow(
        id=row.id,
        title=row.title,
        source=row.source,
        tier=row.tier,
        basis=row.admission_basis,
        profile=row.profile,
        venue=row.venue,
        citation_count=row.citation_count,
        hf_upvotes=row.hf_upvotes,
        github_stars=row.github_stars,
        published_at=row.published_at,
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

    def admitted_details(self, ids: Collection[uuid.UUID]) -> dict[uuid.UUID, ItemDetail]:
        # Admitted rows only: an id from anywhere else (a stale alias, a candidate item) is
        # simply absent, so a caller cannot show text the corpus has not admitted. One query for
        # every id the caller asks about, so a search does not run one query per hit.
        wanted = list(ids)
        if not wanted:
            return {}
        rows = self._s.scalars(select(Item).where(Item.id.in_(wanted), Item.status == "admitted"))
        return {row.id: _detail(row) for row in rows}

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
            # Clipped to the column's width: an over-long venue is a malformed answer, and failing
            # the write would cost the paper its admission over a name no one reads.
            row.venue, row.venue_aliases = (
                fresh.venue[:ITEM_VENUE_MAX_CHARS],
                list(fresh.venue_aliases),
            )
        # Counts only grow: a lower or missing answer is an outage or a lag, not lost evidence.
        row.citation_count = max(row.citation_count, fresh.citation_count)
        if fresh.hf_upvotes is not None:
            row.hf_upvotes = max(row.hf_upvotes or 0, fresh.hf_upvotes)
        self._s.flush()
        return _signals(row)

    def apply_hf_listing(self, papers: Mapping[str, HfPaper], *, at: datetime) -> int:
        """Store the listing extras on stored arXiv items and append changed signal snapshots.

        Returns the number of snapshots written. One read for the items and one for their latest
        snapshots keeps this a fixed cost per run however many papers the window lists.
        """
        if not papers:
            return 0
        # An irrelevant item was classified off-topic on purpose, so it is filtered in SQL: it
        # must not be loaded at all, and the rest loads only the columns this step writes, while
        # the model and the run's fetched pages are still in memory.
        rows = self._s.scalars(
            select(Item)
            .where(
                Item.source == "arxiv",
                Item.status != "irrelevant",
                Item.external_id.in_(list(papers)),
            )
            .options(
                load_only(
                    Item.id,
                    Item.external_id,
                    Item.hf_upvotes,
                    Item.hf_comments,
                    Item.github_repo,
                    Item.github_stars,
                    Item.project_page,
                    Item.hf_org,
                    Item.hf_listed_at,
                    Item.citation_count,
                )
            )
        ).all()
        if not rows:
            return 0
        latest: dict[uuid.UUID, SignalSnapshot] = {}
        for snapshot in self._s.scalars(
            select(SignalSnapshot)
            .where(SignalSnapshot.item_id.in_([row.id for row in rows]))
            # DISTINCT ON keeps the newest row per item, so the comparison costs one query.
            .ext(distinct_on(SignalSnapshot.item_id))
            .order_by(
                SignalSnapshot.item_id, SignalSnapshot.observed_at.desc(), SignalSnapshot.id.desc()
            )
        ):
            latest[snapshot.item_id] = snapshot
        written = 0
        for row in rows:
            paper = papers[row.external_id]
            # Upvotes are evidence, so a lower reading is a lag; the extras describe the paper
            # right now, so the newest non-null reading wins and a missing one keeps the stored
            # value. The names and numbers are clipped to the columns so one hostile value cannot
            # fail the write for every paper.
            upvotes = _int4(paper.upvotes)
            if upvotes is not None:
                row.hf_upvotes = max(row.hf_upvotes or 0, upvotes)
            comments = _int4(paper.comments)
            if comments is not None:
                row.hf_comments = comments
            if paper.github_repo is not None:
                row.github_repo = paper.github_repo
            stars = _int4(paper.github_stars)
            if stars is not None:
                row.github_stars = stars
            if paper.project_page is not None:
                row.project_page = paper.project_page
            if paper.organization is not None:
                row.hf_org = paper.organization[:HF_ORG_MAX_CHARS]
            if paper.listed_at is not None and (
                row.hf_listed_at is None or paper.listed_at < row.hf_listed_at
            ):
                row.hf_listed_at = paper.listed_at
            current = (row.hf_upvotes, row.hf_comments, row.github_stars, row.citation_count)
            previous = latest.get(row.id)
            if (
                previous is not None
                and (
                    previous.hf_upvotes,
                    previous.hf_comments,
                    previous.github_stars,
                    previous.citation_count,
                )
                == current
            ):
                continue
            self._s.add(
                SignalSnapshot(
                    item_id=row.id,
                    observed_at=at,
                    hf_upvotes=row.hf_upvotes,
                    hf_comments=row.hf_comments,
                    github_stars=row.github_stars,
                    citation_count=row.citation_count,
                )
            )
            written += 1
        self._s.flush()
        return written

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
        # The slack is applied on the `now` side, so a NULL interval still needs no special case:
        # `s2_checked_at + interval - slack <= now` and `... + interval <= now + slack` agree.
        due = and_(
            interval.is_not(None),
            or_(
                Item.s2_checked_at.is_(None),
                Item.s2_checked_at + interval <= now + S2_RECHECK_SLACK,
            ),
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
        # from the run's `now`, like every other business timestamp: a row closed this way has
        # the next run's start as its finished_at, not a measured end.
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
        # `at` is the wall clock the caller read when the run ended, not the run's start `now`,
        # so the row's duration is the real one.
        run.finished_at = at
        self._s.flush()

    def run(self, run_id: uuid.UUID) -> Run | None:
        return self._s.get(Run, run_id)

    def recent_runs(self, limit: int) -> list[Run]:
        """The newest runs by start time, for `/runs` and the today strip."""
        return list(self._s.scalars(select(Run).order_by(Run.started_at.desc()).limit(limit)).all())

    def run_dead_letters(self, run_id: uuid.UUID) -> list[DeadLetter]:
        """This run's letters, resolved ones included: a resolved letter still happened."""
        return list(
            self._s.scalars(
                select(DeadLetter).where(DeadLetter.run_id == run_id).order_by(DeadLetter.id)
            ).all()
        )

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

    def has_open_outage_from_earlier_run(self, source: str, ref: str, run_id: uuid.UUID) -> bool:
        # Whether this (source, ref) outage was already open before this run: the first day of an
        # outage is a warning and the second is a failure, and a letter this run opened minutes
        # ago must not be mistaken for yesterday's. Keyed on the letter's run_id, not on its
        # existence, so a follow-up pass that fails again stays the first day (Task 22 review S3).
        return (
            self._s.scalar(
                select(DeadLetter.id)
                .where(
                    DeadLetter.source == source,
                    DeadLetter.ref == ref,
                    DeadLetter.resolved_at.is_(None),
                    DeadLetter.run_id != run_id,
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

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        """Reserve one call against the daily cap, or return no id when the cap is spent.

        The advisory lock is transaction-scoped: it serialises the count and the insert across
        processes (the API's answer loop and the CLI), so two answers at 799 cannot both pass.
        The row is the reservation, with model "reserved" and zero tokens; `fill_chat_call` and
        `fail_chat_call` complete it. It counts against the cap from this moment on, so a provider
        call that hangs still cannot let another one through.
        """
        self._s.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": CHAT_CALLS_LOCK_KEY})
        midnight = at.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        calls_today = self.chat_calls_since(midnight)
        if calls_today >= cap:
            return None, calls_today
        row = ChatCall(
            at=at,
            provider=provider,
            model=RESERVED_MODEL,
            input_tokens=0,
            output_tokens=0,
        )
        self._s.add(row)
        self._s.flush()
        return row.id, calls_today + 1

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        # The provider answered: complete the reserved row. A row that is already gone is not an
        # error here (a test store or a pruned table); the tokens were still spent.
        self._s.execute(
            update(ChatCall)
            .where(ChatCall.id == call_id)
            .values(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        )
        self._s.flush()

    def fail_chat_call(self, call_id: int) -> None:
        # An attempt that reached the provider counts against the cap even when it raised, and
        # "failed" is what tells the operator it produced no answer.
        self._s.execute(update(ChatCall).where(ChatCall.id == call_id).values(model=FAILED_MODEL))
        self._s.flush()

    def suggestion_candidates(
        self, *, now: datetime, max_age: timedelta, momentum_days: int
    ) -> list[Candidate]:
        """Admitted items inside the window, with their upvote history and momentum baseline.

        Snapshots are written only when a recorded number changes, so a flat paper has no recent
        rows at all. The baseline is therefore paired with a run: the upvote count the last run
        that actually read the Hugging Face listing saw at or before the window start. A run from
        before `hf_listing_stored` existed lacks the key and counts as not observing, which is the
        safe reading: no observation is credited rather than a listing nobody read.
        """
        items = list(
            self._s.scalars(
                select(Item).where(Item.status == "admitted", Item.published_at >= now - max_age)
            ).all()
        )
        if not items:
            return []
        ids = [item.id for item in items]
        series: dict[uuid.UUID, list[tuple[datetime, int]]] = {}
        first_repo: dict[uuid.UUID, datetime] = {}
        for snapshot in self._s.scalars(
            select(SignalSnapshot)
            .where(
                SignalSnapshot.item_id.in_(ids),
                SignalSnapshot.observed_at >= now - max_age,
            )
            .order_by(SignalSnapshot.item_id, SignalSnapshot.observed_at, SignalSnapshot.id)
        ):
            if snapshot.hf_upvotes is not None:
                series.setdefault(snapshot.item_id, []).append(
                    (snapshot.observed_at, snapshot.hf_upvotes)
                )
            if snapshot.github_stars is not None:
                first_repo.setdefault(snapshot.item_id, snapshot.observed_at)
        promotion: dict[uuid.UUID, datetime] = {}
        for item_id, changed_at in self._s.execute(
            select(TierHistory.item_id, func.max(TierHistory.changed_at))
            .where(
                TierHistory.item_id.in_(ids),
                TierHistory.to_tier == Tier.PEER_REVIEWED.value,
            )
            .group_by(TierHistory.item_id)
        ):
            promotion[item_id] = changed_at
        baselines = self._upvote_baselines(ids, observed_at=now - timedelta(days=momentum_days))
        return [
            replace(
                _candidate(item),
                upvote_series=tuple(series.get(item.id, ())),
                upvote_baseline=baselines.get(item.id),
                promoted_to_peer_at=promotion.get(item.id),
                repo_first_seen_at=first_repo.get(item.id),
            )
            for item in items
        ]

    def _upvote_baselines(
        self, ids: Collection[uuid.UUID], *, observed_at: datetime
    ) -> dict[uuid.UUID, tuple[datetime, int]]:
        """Each item's upvotes at the last observing run at or before `observed_at`.

        One query finds that run, one `DISTINCT ON` read pairs every item with its newest snapshot
        at it. The run's own `started_at` is the time its snapshots were stamped with, so the two
        are comparable.
        """
        run_at = self._s.scalar(
            select(func.max(Run.started_at)).where(
                Run.started_at <= observed_at,
                Run.counts["hf_listing_stored"].as_integer() == 1,
            )
        )
        if run_at is None:
            return {}
        baselines: dict[uuid.UUID, tuple[datetime, int]] = {}
        for snapshot in self._s.scalars(
            select(SignalSnapshot)
            .where(
                SignalSnapshot.item_id.in_(list(ids)),
                SignalSnapshot.observed_at <= run_at,
                SignalSnapshot.hf_upvotes.is_not(None),
            )
            .ext(distinct_on(SignalSnapshot.item_id))
            .order_by(
                SignalSnapshot.item_id,
                SignalSnapshot.observed_at.desc(),
                SignalSnapshot.id.desc(),
            )
        ):
            # The run's own time, not the snapshot's: snapshots are written only on change, so a
            # paper's newest one can be days older than the run that read the listing, and the
            # gap it would add is time nobody looked (ADR 0014).
            baselines[snapshot.item_id] = (run_at, snapshot.hf_upvotes or 0)
        return baselines

    def excluded_items(self) -> frozenset[uuid.UUID]:
        """Items the reader dismissed or drafted: the ranking never offers them again."""
        return frozenset(
            self._s.scalars(
                select(SuggestionEvent.item_id).where(
                    SuggestionEvent.event.in_((DISMISSED_EVENT, DRAFTED_EVENT))
                )
            ).all()
        )

    def record_suggestion_events(self, rows: Sequence[SuggestionEventRow]) -> None:
        if not rows:
            return
        self._s.execute(
            insert(SuggestionEvent),
            [
                {
                    "item_id": row.item_id,
                    "event": row.event,
                    "at": row.at,
                    "rank": row.rank,
                    "score": row.score,
                    "components": row.components,
                    "list_version": row.list_version,
                }
                for row in rows
            ],
        )
        self._s.flush()

    def last_shown(self, item_ids: Collection[uuid.UUID], *, since: datetime) -> set[uuid.UUID]:
        """Which of these items were already shown since `since`, so a card is not re-recorded."""
        if not item_ids:
            return set()
        return set(
            self._s.scalars(
                select(SuggestionEvent.item_id)
                .where(
                    SuggestionEvent.item_id.in_(list(item_ids)),
                    SuggestionEvent.event == SHOWN_EVENT,
                    SuggestionEvent.at >= since,
                )
                .distinct()
            ).all()
        )

    def summaries(self, item_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, ItemSummary]:
        if not item_ids:
            return {}
        return {
            row.item_id: row
            for row in self._s.scalars(
                select(ItemSummary).where(ItemSummary.item_id.in_(list(item_ids)))
            )
        }

    def upsert_summary(
        self,
        item_id: uuid.UUID,
        summary: "Summary",
        *,
        text_hash: str,
        model: str,
        at: datetime,
    ) -> None:
        row = self._s.get(ItemSummary, item_id)
        if row is None:
            row = ItemSummary(item_id=item_id)
            self._s.add(row)
        row.text_hash = text_hash
        row.model = model
        row.prompt_version = SUMMARY_PROMPT_VERSION
        row.status = summary.status
        # Imported here, not at module level: `suggestions.summary` imports LangChain, and the
        # ingestion imports this module on every run.
        from paperwatch.suggestions.summary import summary_fields

        row.fields = summary_fields(summary)
        row.failed_checks = list(summary.failed_checks)
        row.created_at = at
        self._s.flush()

    def summary_sources(self, item_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, tuple[str, str]]:
        """The title and text a summary is written from, for the queue's own background session."""
        if not item_ids:
            return {}
        return {
            item_id: (title, text)
            for item_id, title, text in self._s.execute(
                select(Item.id, Item.title, Item.text).where(Item.id.in_(list(item_ids)))
            )
        }

    def admitted_item(self, item_id: uuid.UUID) -> bool:
        """Whether this id is an admitted item, so "Ask about it" can pre-register it as S1."""
        return (
            self._s.scalar(select(Item.id).where(Item.id == item_id, Item.status == "admitted"))
            is not None
        )

    def draft_item(self, item_id: uuid.UUID) -> DraftItem | None:
        """One admitted item for a draft, or None when it is not admitted or not stored."""
        row = self._s.scalar(select(Item).where(Item.id == item_id, Item.status == "admitted"))
        if row is None:
            return None
        candidate = _candidate(row)
        return DraftItem(
            item_id=row.id,
            title=row.title,
            url=row.url,
            source=row.source,
            text=row.text or "",
            org=candidate.org,
            github_repo=row.github_repo,
            project_page=row.project_page,
            candidate=candidate,
        )

    def add_draft(
        self,
        *,
        item_id: uuid.UUID,
        inputs: Mapping[str, object],
        pov: str,
        language: str,
        model: str,
        prompt_version: str,
        hook: str,
        body: str,
        hashtags: Sequence[str],
        source_link: str,
        check: Mapping[str, object],
        regenerated_from: uuid.UUID | None,
        at: datetime,
    ) -> DraftRow:
        """Save one draft exactly as the model wrote it, with its inputs and its check."""
        row = Draft(
            item_id=item_id,
            regenerated_from=regenerated_from,
            pov=pov,
            language=language,
            model=model,
            prompt_version=prompt_version,
            inputs=dict(inputs),
            hook=hook,
            body=body,
            hashtags=list(hashtags),
            source_link=source_link,
            check=dict(check),
            created_at=at,
        )
        self._s.add(row)
        self._s.flush()
        return _draft_row(row)

    def drafts(self, limit: int) -> list[DraftRow]:
        """The saved drafts, newest first, for the Drafts section."""
        return [
            _draft_row(row)
            for row in self._s.scalars(select(Draft).order_by(Draft.created_at.desc()).limit(limit))
        ]

    def draft(self, draft_id: uuid.UUID) -> DraftRow | None:
        row = self._s.get(Draft, draft_id)
        return _draft_row(row) if row is not None else None

    def threads(self, limit: int) -> list[ThreadRow]:
        """The Recent menu: newest conversation first, with its finished turn count."""
        counts = (
            select(ThreadTurn.thread_id, func.count().label("turns"))
            .group_by(ThreadTurn.thread_id)
            .subquery()
        )
        rows = self._s.execute(
            select(Thread, func.coalesce(counts.c.turns, 0))
            .outerjoin(counts, counts.c.thread_id == Thread.id)
            .order_by(Thread.updated_at.desc())
            .limit(limit)
        )
        return [
            ThreadRow(
                id=thread.id, title=thread.title, turns=int(turns), updated_at=thread.updated_at
            )
            for thread, turns in rows
        ]

    def thread(self, thread_id: uuid.UUID) -> ThreadDetailRow | None:
        thread = self._s.get(Thread, thread_id)
        if thread is None:
            return None
        turns = self._s.scalars(
            select(ThreadTurn).where(ThreadTurn.thread_id == thread_id).order_by(ThreadTurn.n)
        )
        return ThreadDetailRow(
            id=thread.id,
            title=thread.title,
            registry=tuple(thread.source_registry),
            turns=tuple(_thread_turn_row(turn) for turn in turns),
        )

    def save_thread_turn(
        self,
        *,
        thread_id: uuid.UUID | None,
        title: str,
        registry: Sequence[Mapping[str, str]],
        question: str,
        answer: Mapping[str, object],
        search_mode: str,
        at: datetime,
        expected_turns: int,
    ) -> tuple[uuid.UUID, int]:
        """Append one finished turn, creating the thread when there is none.

        The registry is rewritten with every alias the thread has now, so a follow-up that added
        sources makes them citable in the turn after it too. The thread row is locked for the
        duration, and `expected_turns` is the number of turns the answer saw when it loaded the
        thread: a thread that grew since then belongs to another answer's registry, so this save is
        refused rather than overwriting it (the unique `(thread_id, n)` cannot catch that, because
        the two turns receive different numbers).
        """
        stored = [dict(ref) for ref in registry]
        if thread_id is None:
            created = Thread(
                id=uuid.uuid4(),
                title=title,
                source_registry=stored,
                created_at=at,
                updated_at=at,
            )
            self._s.add(created)
            self._s.flush()
            thread_id = created.id
        else:
            existing = self._s.get(Thread, thread_id, with_for_update=True)
            if existing is None:
                raise LookupError("thread not found")
            held = (
                self._s.scalar(
                    select(func.count())
                    .select_from(ThreadTurn)
                    .where(ThreadTurn.thread_id == thread_id)
                )
                or 0
            )
            if held > expected_turns:
                raise ThreadConflict(
                    f"the thread holds {held} turns, the answer saw {expected_turns}"
                )
            existing.source_registry = stored
            existing.updated_at = at
        n = (
            self._s.scalar(select(func.max(ThreadTurn.n)).where(ThreadTurn.thread_id == thread_id))
            or 0
        ) + 1
        self._s.add(
            ThreadTurn(
                thread_id=thread_id,
                n=n,
                question=question,
                answer=dict(answer),
                search_mode=search_mode,
                created_at=at,
            )
        )
        self._s.flush()
        return thread_id, n

    def browse_items(
        self,
        *,
        text_search: str | None = None,
        tiers: frozenset[str] | None = None,
        source: str | None = None,
        profile: str | None = None,
        since: datetime | None = None,
        after: tuple[datetime, uuid.UUID] | None = None,
        offset: int = 0,
        limit: int,
    ) -> CorpusPage:
        """One page of admitted items: the newest published first, or the full-text matches.

        Without a query the page walks the keyset `(published_at, id)` descending, which the
        `(status, published_at)` index serves; with one, Postgres full-text ranks the matches and
        the cursor is an offset into that ranking, capped at `CORPUS_SEARCH_MAX`.
        """
        narrowed = _corpus_filters(
            select(func.count()).select_from(Item),
            tiers=tiers,
            source=source,
            profile=profile,
            since=since,
        )
        if text_search is None:
            total = int(self._s.scalar(narrowed) or 0)
            statement = _corpus_filters(
                _corpus_select(),
                tiers=tiers,
                source=source,
                profile=profile,
                since=since,
            )
            if after is not None:
                # Row comparison: one predicate, so the keyset cannot skip or repeat a row whose
                # published date it shares with another.
                statement = statement.where(
                    tuple_(Item.published_at, Item.id) < tuple_(after[0], after[1])
                )
            # One row past the page: a page that exactly fills the limit must not offer a next
            # one that turns out empty.
            statement = statement.order_by(Item.published_at.desc(), Item.id.desc()).limit(
                limit + 1
            )
            found = [_corpus_row(row) for row in self._s.execute(statement)]
            rows = tuple(found[:limit])
            return CorpusPage(rows, total, len(found) > limit)

        # Imported here, not at module level: `retrieval.hybrid` imports the embedder, and the
        # ingestion imports this module on every run.
        from paperwatch.retrieval.hybrid import TS_RANK_NORMALIZATION, any_term_tsquery

        tsq = any_term_tsquery(text_search)
        matching = narrowed.where(Item.tsv.op("@@")(tsq))
        # The count is capped like the paging: past the cap the table says "many" rather than
        # counting a match set nobody can page through.
        total = min(int(self._s.scalar(matching) or 0), CORPUS_SEARCH_MAX)
        rank = func.ts_rank_cd(Item.tsv, tsq, TS_RANK_NORMALIZATION)
        statement = (
            _corpus_filters(
                _corpus_select(),
                tiers=tiers,
                source=source,
                profile=profile,
                since=since,
            )
            .where(Item.tsv.op("@@")(tsq))
            # The stable key breaks a rank tie, so paging never repeats or hides a row; the
            # statement sorts anyway, so the GIN index still serves the match.
            .order_by(rank.desc(), Item.source.collate("C"), Item.external_id.collate("C"))
            .offset(offset)
            # The cap is on the list, not on the page size: a limit that does not divide 200 must
            # not let the table walk past it.
            .limit(min(limit, CORPUS_SEARCH_MAX - offset))
        )
        rows = tuple(_corpus_row(row) for row in self._s.execute(statement))
        return CorpusPage(rows, total, offset + len(rows) < total)

    def corpus_item(self, item_id: uuid.UUID) -> CorpusItemDetail | None:
        """One admitted item with its text, or None when it is not admitted or not stored."""
        # The whole row except its vector: `_candidate` is the one place the item-to-candidate
        # mapping lives, and the page's `early` flag has to agree with the ranking's.
        row = self._s.scalar(
            select(Item)
            .options(
                load_only(
                    Item.id,
                    Item.source,
                    Item.external_id,
                    Item.tier,
                    Item.admission_basis,
                    Item.profile,
                    Item.relevance,
                    Item.title,
                    Item.authors,
                    Item.text,
                    Item.url,
                    Item.published_at,
                    Item.venue,
                    Item.citation_count,
                    Item.hf_upvotes,
                    Item.github_stars,
                    Item.project_page,
                    Item.github_repo,
                    Item.hf_org,
                    Item.hf_listed_at,
                    # `_candidate` carries the vector; one row of it is about 3 KB, and leaving it
                    # out would only buy a second query on the item page.
                    Item.embedding,
                )
            )
            .where(Item.id == item_id, text(f"items.{ADMITTED_SQL}"))
        )
        if row is None:
            return None
        return CorpusItemDetail(
            row=_corpus_row(row),
            authors=tuple(str(author) for author in row.authors or ()),
            url=row.url,
            text=row.text or "",
            github_repo=row.github_repo,
            project_page=row.project_page,
            org=row.hf_org,
            candidate=_candidate(row),
        )

    def corpus_tier_history(self, item_id: uuid.UUID) -> tuple[TierChangeRow, ...]:
        """An item's tier moves, oldest first, so the page draws a timeline without sorting."""
        rows = self._s.execute(
            select(
                TierHistory.from_tier,
                TierHistory.to_tier,
                TierHistory.basis,
                TierHistory.changed_at,
            )
            .where(TierHistory.item_id == item_id)
            .order_by(TierHistory.changed_at, TierHistory.id)
        )
        return tuple(
            TierChangeRow(row.from_tier, row.to_tier, row.basis, row.changed_at) for row in rows
        )

    def corpus_signals(self, item_id: uuid.UUID) -> tuple[SignalPointRow, ...]:
        """The newest readings of an item's signals, oldest first for the page's table."""
        rows = self._s.execute(
            select(
                SignalSnapshot.observed_at,
                SignalSnapshot.hf_upvotes,
                SignalSnapshot.hf_comments,
                SignalSnapshot.github_stars,
                SignalSnapshot.citation_count,
            )
            .where(SignalSnapshot.item_id == item_id)
            .order_by(SignalSnapshot.observed_at.desc(), SignalSnapshot.id.desc())
            .limit(CORPUS_SIGNALS_MAX)
        )
        return tuple(
            SignalPointRow(
                row.observed_at,
                row.hf_upvotes,
                row.hf_comments,
                row.github_stars,
                row.citation_count,
            )
            for row in reversed(rows.all())
        )

    def corpus_drafts(self, item_id: uuid.UUID) -> tuple[tuple[uuid.UUID, datetime], ...]:
        """The drafts an item already has, newest first, so the page need not load them all."""
        rows = self._s.execute(
            select(Draft.id, Draft.created_at)
            .where(Draft.item_id == item_id)
            .order_by(Draft.created_at.desc(), Draft.id)
        )
        return tuple((row.id, row.created_at) for row in rows)


class ChatCallLog:
    """The production `ChatCallStore`: every usage row is committed on its own short session.

    The request's own session cannot carry these rows: a failed answer or a client that hangs up
    rolls it back, and the calls it already spent against the free tier would vanish from the
    daily cap (ADR 0006). A row also becomes visible to other requests at once, instead of at the
    end of a long answer, which keeps the cap accurate when two questions run together.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        # Its own transaction, committed before returning: the lock, the count and the insert are
        # one unit, and the reservation is visible to every other process immediately.
        with self._session_factory() as session:
            call_id, calls_today = Repository(session).reserve_chat_call(
                at=at, provider=provider, cap=cap
            )
            session.commit()
            return call_id, calls_today

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        with self._session_factory() as session:
            Repository(session).fill_chat_call(
                call_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            session.commit()

    def fail_chat_call(self, call_id: int) -> None:
        with self._session_factory() as session:
            Repository(session).fail_chat_call(call_id)
            session.commit()
