"""The corpus routes: the browsable table and one item with its history, signals and drafts.

The table pages by keyset on `(published_at, id)`, so a tie never repeats or hides a row, and a
search walks the full-text ranking by offset because the ranking is a score, not a key. A search
never embeds: the corpus is readable in every search mode, including while the ingestion holds the
model's memory.

The shapes here are what `web/src/lib/api-types.ts` is generated from, so a field that changes must
be added to the committed contract in the same commit.
"""

import base64
import binascii
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel
from sqlalchemy.orm import Session

from paperwatch.api.responses import BEARER, GATE_RESPONSES, Refusal
from paperwatch.api.routes.suggestions import SummaryData
from paperwatch.domain import AdmissionBasis, Tier
from paperwatch.store.models import ItemSummary
from paperwatch.store.repository import CorpusItemRow, Repository
from paperwatch.suggestions.cards import basis_label, summary_data

# The table shows 25 rows and the API takes at most 50: a page is read, not crawled.
CORPUS_DEFAULT_LIMIT = 25
CORPUS_MAX_LIMIT = 50
# A query longer than this is a paste, not a search: the full-text parse would only spend time on
# words no abstract holds.
QUERY_MAX_CHARS = 200
PROFILE_MAX_CHARS = 64
# The three tiers a reader can filter to, in the words the tabs print.
CorpusTier = Literal["peer_reviewed", "vetted_preprint", "lab_blog"]
# The two stored sources: an arXiv record, or a lab's own feed post.
CorpusSource = Literal["arxiv", "feed"]


class CorpusItemData(BaseModel):
    """One row of the table: the facts the columns show, and the API's own basis label."""

    id: UUID
    title: str
    source: str
    tier: str
    basis: str
    basis_label: str
    profile: str | None
    venue: str | None
    citation_count: int
    hf_upvotes: int | None
    github_stars: int | None
    published_at: datetime


class CorpusData(BaseModel):
    """One page: the rows, the cursor for the next page (null at the end) and the match count."""

    items: list[CorpusItemData]
    next_cursor: str | None
    total: int


class TierChangeData(BaseModel):
    """One move in the item's tier history."""

    from_tier: str | None
    to_tier: str
    basis: str
    changed_at: datetime


class SignalPointData(BaseModel):
    """One observed reading of the item's signals, oldest first in the list."""

    observed_at: datetime
    hf_upvotes: int | None
    hf_comments: int | None
    github_stars: int | None
    citation_count: int


class ItemDraftData(BaseModel):
    """A draft the item already has, so the page can offer to open it instead of making another."""

    id: UUID
    created_at: datetime


class CorpusItemDetail(CorpusItemData):
    """The row plus the text, the extras, the history and what was already written about it."""

    authors: list[str]
    url: str
    text: str
    github_repo: str | None
    project_page: str | None
    org: str | None
    tier_history: list[TierChangeData]
    signals: list[SignalPointData]
    # None when no 30-second summary is cached yet: the page shows the stored text and says so
    # rather than serving an unchecked sentence.
    summary: SummaryData | None
    drafts: list[ItemDraftData]


@dataclass(frozen=True)
class _Cursor:
    """A decoded cursor: a keyset position, or an offset into the full-text ranking."""

    published_at: datetime | None = None
    item_id: UUID | None = None
    offset: int | None = None


def _encode_newest(row: CorpusItemData) -> str:
    return _encode({"k": "newest", "p": row.published_at.isoformat(), "i": str(row.id)})


def _encode_search(offset: int) -> str:
    return _encode({"k": "search", "o": offset})


def _encode(values: dict[str, object]) -> str:
    # Opaque on purpose: the page must hand the cursor back untouched, so it can change shape
    # without a page release.
    raw = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str) -> _Cursor:
    try:
        padded = value + "=" * (-len(value) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
        if not isinstance(data, dict):
            raise ValueError("cursor is not an object")
        if data.get("k") == "newest":
            return _Cursor(
                published_at=datetime.fromisoformat(str(data["p"])), item_id=UUID(str(data["i"]))
            )
        if data.get("k") == "search":
            offset = int(str(data["o"]))
            if offset > 0:
                return _Cursor(offset=offset)
    except (ValueError, KeyError, TypeError, binascii.Error):
        pass
    # A cursor is the API's own token; anything else is a client bug, and answering 422 keeps a
    # malformed one from silently browsing the corpus from the start.
    raise HTTPException(status_code=422, detail="cursor is not a corpus cursor")


def _item_data(row: CorpusItemRow) -> CorpusItemData:
    return CorpusItemData(
        id=row.id,
        title=row.title,
        source=row.source,
        tier=row.tier,
        basis=row.basis,
        # The API's own sentence, so a changed admission rule never leaves the page describing it
        # wrong; the enums narrow the stored strings the check constraints already guarantee.
        basis_label=basis_label(
            Tier(row.tier),
            AdmissionBasis(row.basis),
            row.venue,
            row.hf_upvotes,
            row.citation_count,
        ),
        profile=row.profile,
        venue=row.venue,
        citation_count=row.citation_count,
        hf_upvotes=row.hf_upvotes,
        github_stars=row.github_stars,
        published_at=row.published_at,
    )


def _since_midnight(day: date | None) -> datetime | None:
    # A published date is a whole day, so "since" includes that day from its UTC midnight.
    return datetime(day.year, day.month, day.day, tzinfo=UTC) if day is not None else None


def make_items_router(
    session_factory: Callable[[], Session], profile_names: frozenset[str]
) -> APIRouter:
    """Build the router with the read factory and the known profile names."""
    router = APIRouter()

    @router.get(
        "/items",
        response_model=CorpusData,
        responses={
            **GATE_RESPONSES,
            422: {"model": Refusal, "description": "An unknown profile, tier or cursor."},
        },
        dependencies=[Security(BEARER)],
    )
    def items(
        q: Annotated[str | None, Query(max_length=QUERY_MAX_CHARS)] = None,
        tier: CorpusTier | None = None,
        source: CorpusSource | None = None,
        profile: Annotated[str | None, Query(max_length=PROFILE_MAX_CHARS)] = None,
        since: date | None = None,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=CORPUS_MAX_LIMIT)] = CORPUS_DEFAULT_LIMIT,
    ) -> CorpusData:
        # A blank search box is no search: the table shows the newest items instead of nothing.
        search = q.strip() if q and q.strip() else None
        if profile is not None and profile not in profile_names:
            # The names are the process's own, so a typo is refused here rather than answered with
            # an empty table that looks like a corpus without that topic.
            raise HTTPException(status_code=422, detail=f"unknown profile: {profile}")
        decoded = _decode_cursor(cursor) if cursor else _Cursor()
        # A cursor belongs to the branch that issued it: a keyset position and a search offset are
        # not interchangeable, and guessing one would page from the wrong place.
        if cursor and (decoded.offset is not None) != (search is not None):
            raise HTTPException(status_code=422, detail="cursor does not match the request")

        with session_factory() as session:
            page = Repository(session).browse_items(
                text_search=search,
                tiers=frozenset({tier}) if tier else None,
                source=source,
                profile=profile,
                since=_since_midnight(since),
                after=(decoded.published_at, decoded.item_id)
                if decoded.published_at is not None and decoded.item_id is not None
                else None,
                offset=decoded.offset or 0,
                limit=limit,
            )
        rows = [_item_data(row) for row in page.rows]
        if not page.has_more or not rows:
            next_cursor = None
        elif search is not None:
            next_cursor = _encode_search((decoded.offset or 0) + len(rows))
        else:
            next_cursor = _encode_newest(rows[-1])
        return CorpusData(items=rows, next_cursor=next_cursor, total=page.total)

    @router.get(
        "/items/{item_id}",
        response_model=CorpusItemDetail,
        responses={
            **GATE_RESPONSES,
            404: {"model": Refusal, "description": "No admitted item with that id."},
        },
        dependencies=[Security(BEARER)],
    )
    def item_detail(item_id: UUID) -> CorpusItemDetail:
        with session_factory() as session:
            repo = Repository(session)
            item = repo.corpus_item(item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="item not found")
            stored: ItemSummary | None = repo.summaries([item_id]).get(item_id)
            # The cards' own summary builder, so the item page and a pick can never describe the
            # same stored summary differently; the model shape is the one the contract declares.
            built = summary_data(stored, item.text) if stored is not None else None
            summary = SummaryData.model_validate(asdict(built)) if built is not None else None
            return CorpusItemDetail(
                **_item_data(item.row).model_dump(),
                authors=list(item.authors),
                url=item.url,
                text=item.text,
                github_repo=item.github_repo,
                project_page=item.project_page,
                org=item.org,
                tier_history=[
                    TierChangeData(
                        from_tier=change.from_tier,
                        to_tier=change.to_tier,
                        basis=change.basis,
                        changed_at=change.changed_at,
                    )
                    for change in repo.corpus_tier_history(item_id)
                ],
                signals=[
                    SignalPointData(
                        observed_at=point.observed_at,
                        hf_upvotes=point.hf_upvotes,
                        hf_comments=point.hf_comments,
                        github_stars=point.github_stars,
                        citation_count=point.citation_count,
                    )
                    for point in repo.corpus_signals(item_id)
                ],
                summary=summary,
                drafts=[
                    ItemDraftData(id=draft_id, created_at=created_at)
                    for draft_id, created_at in repo.corpus_drafts(item_id)
                ],
            )

    return router
