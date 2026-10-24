"""The suggestions route: the ranked list, its summaries and the interactions it records.

The list is computed on request from stored features (never stored as a ranking) and cached in the
process for five minutes, so a page refresh is free but a dismissal is visible at once: the events
route drops the cache. Cards are recorded as shown at most once an hour each, so "shown" measures
what a reader actually saw rather than how often their tab refreshed.

The shapes here are what `web/src/lib/api-types.ts` is generated from, so a field that changes must
be added to the committed contract in the same commit.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from sqlalchemy.orm import Session

from paperwatch.api.responses import BEARER, GATE_RESPONSES, Refusal
from paperwatch.api.summary_queue import SummaryQueue
from paperwatch.domain import Profile
from paperwatch.store.models import ItemSummary
from paperwatch.store.repository import (
    DISMISSED_EVENT,
    Repository,
    SuggestionEventRow,
)
from paperwatch.suggestions.cards import SuggestionCard, card
from paperwatch.suggestions.config import MAX_CANDIDATE_DAYS, SuggestionsConfig
from paperwatch.suggestions.ranking import Ranking, Scored, rank

log = structlog.get_logger()

# Five minutes: long enough that a page refresh costs nothing, short enough that a list promised to
# be recomputed on request does not lag behind the corpus by a working session.
RANK_CACHE_S = 300
# "Shown" is recorded at most once an hour per item: one reader scrolling back and forth is not
# eight impressions.
SHOWN_THROTTLE_S = 3600
# The page prints this under the Rising shelf, and it is part of the response so the caveat cannot
# drift away from the cards it is about.
RISING_NOTE = "Rising, not yet independently vetted."


class SummaryFieldData(BaseModel):
    text: str
    quote: str | None
    interpretation: bool


class SummaryData(BaseModel):
    status: Literal["pending", "checked", "excerpt"]
    what: SummaryFieldData | None
    result: SummaryFieldData | None
    why: SummaryFieldData | None
    catch: SummaryFieldData | None
    excerpt: str


class AlsoData(BaseModel):
    title: str
    url: str
    source: str


class SuggestionCardData(BaseModel):
    item_id: UUID
    title: str
    url: str
    source: str
    tier: str
    basis: str
    basis_label: str
    venue: str | None
    citation_count: int
    hf_upvotes: int | None
    github_repo: str | None
    github_stars: int | None
    project_page: str | None
    org: str | None
    published_at: datetime
    profile: str | None
    relevance: float
    momentum_label: str | None
    also: list[AlsoData]
    summary: SummaryData


class SuggestionsData(BaseModel):
    generated_at: datetime
    list_version: str
    picks: list[SuggestionCardData]
    rising: list[SuggestionCardData]
    rising_note: str


class EventRequest(BaseModel):
    # "shown" is recorded by the API itself: a page could otherwise claim an impression it never
    # made, and the throttle's whole point is that the API counts what it served.
    event: Literal["opened", "dismissed"]


@dataclass(frozen=True)
class _Cached:
    ranking: Ranking
    # The "now" the ranking was computed with, so the cache age and every label on its cards agree.
    generated_at: datetime
    list_version: str


class SuggestionService:
    """The ranking cache, the card building and the summary queue, in one place per process."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        profiles: Mapping[str, Profile],
        config: SuggestionsConfig,
        queue: SummaryQueue,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_factory = session_factory
        self._profiles = profiles
        self._config = config
        self._queue = queue
        self._clock = clock
        self._cached: _Cached | None = None

    def invalidate(self) -> None:
        # Called by the events route on a dismissal: the item must leave the list at once.
        self._cached = None

    def list(self) -> SuggestionsData:
        now = self._clock()
        cached = self._cached
        if cached is None or (now - cached.generated_at).total_seconds() >= RANK_CACHE_S:
            cached = self._compute(now)
            self._cached = cached
        with self._session_factory() as session:
            repo = Repository(session)
            scored = [*cached.ranking.picks, *cached.ranking.rising]
            ids = [item.candidate.item_id for item in scored]
            summaries = repo.summaries(ids)
            # The excerpt of a card with no summary yet is the abstract's own sentences, so the
            # route needs the text of exactly the cards it is about to show.
            sources = repo.summary_sources(ids)
            shown = repo.last_shown(ids, since=now - timedelta(seconds=SHOWN_THROTTLE_S))
            picks = [
                _card(item, summaries, sources, cached.generated_at)
                for item in cached.ranking.picks
                if item.candidate.item_id in sources
            ]
            rising = [
                _card(item, summaries, sources, cached.generated_at)
                for item in cached.ranking.rising
                if item.candidate.item_id in sources
            ]
            self._record_shown(repo, cached, scored, shown, now)
            session.commit()
        self._queue.submit([entry.item_id for entry in (*picks, *rising)])
        return SuggestionsData(
            generated_at=cached.generated_at,
            list_version=cached.list_version,
            picks=picks,
            rising=rising,
            rising_note=RISING_NOTE,
        )

    def record(self, item_id: UUID, event: Literal["opened", "dismissed"]) -> None:
        now = self._clock()
        with self._session_factory() as session:
            repo = Repository(session)
            if not repo.admitted_item(item_id):
                raise HTTPException(status_code=404, detail="item not found")
            repo.record_suggestion_events(
                [SuggestionEventRow(item_id=item_id, event=event, at=now)]
            )
            session.commit()
        if event == DISMISSED_EVENT:
            self.invalidate()

    def _compute(self, now: datetime) -> _Cached:
        max_age = timedelta(
            days=max(self._config.window_days.paper, self._config.window_days.lab_post)
        )
        max_age = min(max_age, timedelta(days=MAX_CANDIDATE_DAYS))
        with self._session_factory() as session:
            repo = Repository(session)
            candidates = repo.suggestion_candidates(
                now=now, max_age=max_age, momentum_days=self._config.momentum_days
            )
            excluded = repo.excluded_items()
        ranking = rank(
            candidates,
            now=now,
            config=self._config,
            profiles=self._profiles,
            excluded=excluded,
        )
        ordered = "|".join(
            f"{item.candidate.source}:{item.candidate.external_id}"
            for item in (*ranking.picks, *ranking.rising)
        )
        return _Cached(
            ranking=ranking,
            generated_at=now,
            list_version=hashlib.sha256(f"{now.isoformat()}|{ordered}".encode()).hexdigest()[:32],
        )

    def _record_shown(
        self,
        repo: Repository,
        cached: _Cached,
        scored: Sequence[Scored],
        shown: set[UUID],
        now: datetime,
    ) -> None:
        rows = []
        for position, entry in enumerate(scored, start=1):
            item_id = entry.candidate.item_id
            if item_id in shown:
                continue
            rows.append(
                SuggestionEventRow(
                    item_id=item_id,
                    event="shown",
                    at=now,
                    rank=position,
                    score=entry.score,
                    components=dict(entry.components),
                    list_version=cached.list_version,
                )
            )
        repo.record_suggestion_events(rows)


def _card(
    scored: Scored,
    summaries: Mapping[UUID, ItemSummary],
    sources: Mapping[UUID, tuple[str, str]],
    now: datetime,
) -> SuggestionCardData:
    built: SuggestionCard = card(
        scored,
        summaries.get(scored.candidate.item_id),
        now,
        text=sources[scored.candidate.item_id][1],
    )
    return SuggestionCardData.model_validate(asdict(built))


def make_suggestions_router(service: SuggestionService) -> APIRouter:
    """Build the router from the service, so the app owns the cache and the queue."""
    router = APIRouter()

    @router.get(
        "/suggestions",
        response_model=SuggestionsData,
        responses=GATE_RESPONSES,
        dependencies=[Security(BEARER)],
    )
    def suggestions() -> SuggestionsData:
        return service.list()

    @router.post(
        "/suggestions/{item_id}/events",
        status_code=204,
        responses={
            **GATE_RESPONSES,
            404: {"model": Refusal, "description": "No admitted item with that id."},
        },
        dependencies=[Security(BEARER)],
    )
    def record_event(item_id: UUID, body: EventRequest) -> None:
        service.record(item_id, body.event)

    return router


__all__ = [
    "RISING_NOTE",
    "SuggestionCardData",
    "SuggestionService",
    "SuggestionsData",
    "make_suggestions_router",
]
