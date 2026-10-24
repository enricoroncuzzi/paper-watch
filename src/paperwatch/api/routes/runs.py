"""The read routes the page's today strip and Runs section need.

These are plain sync routes: FastAPI runs them in a thread, so a slow read never blocks the event
loop an answer's stream is running on. Each request opens its own short session from the read
factory, so a page load holds no connection while the browser renders.

The shapes here are what `web/src/lib/api-types.ts` is generated from, so a field that changes must
be added to the committed contract in the same commit.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal, Protocol
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from paperwatch.api.search_mode import ModeDecision
from paperwatch.store.models import DeadLetter, Run
from paperwatch.store.repository import Repository

# A dead letter's `error` is a provider message that can run to thousands of characters (a whole
# HTML error page); the page shows one line, and the cap keeps the response small.
DEAD_LETTER_TEXT_MAX = 500
# The strip shows the last run and the Runs section the last seven; ten leaves room for a reader
# who wants a couple more without making the response heavy.
RUNS_DEFAULT_LIMIT = 10
RUNS_MAX_LIMIT = 50


class ModeGate(Protocol):
    """The one method a route needs from the search-mode gate."""

    def current(self) -> ModeDecision: ...


class RunCounts(BaseModel):
    """The counts the page reads as six cells; a key the run never counted is 0."""

    fetched: int = 0
    fetched_arxiv: int = 0
    stored: int = 0
    new: int = 0
    irrelevant: int = 0
    failed: int = 0
    admitted_late: int = 0
    followup_passes: int = 0
    s2_refused: int = 0


class FollowupState(BaseModel):
    """A run waiting between follow-up passes, as `health.followup` records it."""

    pass_: int = Field(alias="pass")
    of: int
    next_at: datetime | None = None
    pending_s2: int | None = None

    # The record key and the JSON key are both `pass`, which cannot be a Python field name.
    model_config = {"populate_by_name": True}


class RunSummary(BaseModel):
    id: UUID
    started_at: datetime
    finished_at: datetime | None
    status: str
    # None when the run closed without a verdict (a crash closed by the next run), which the page
    # shows as unknown rather than healthy or unhealthy.
    healthy: bool | None
    problems: list[str]
    warnings: list[str]
    counts: RunCounts
    tokens: int
    followup: FollowupState | None


class DeadLetterData(BaseModel):
    source: str
    ref: str
    error: str
    attempts: int
    created_at: datetime
    resolved_at: datetime | None


class RunDetail(RunSummary):
    window_start: datetime
    window_end: datetime
    dead_letters: list[DeadLetterData]


class RunsData(BaseModel):
    runs: list[RunSummary]


class TodaySearch(BaseModel):
    mode: Literal["hybrid", "text_only"]
    reason: Literal["ok", "ingestion", "memory", "database"]
    since: datetime | None = None


class TodayData(BaseModel):
    run: RunSummary | None
    search: TodaySearch


def _strings(value: object) -> list[str]:
    return [str(entry) for entry in value] if isinstance(value, list) else []


def _followup(health: dict[str, object] | None) -> FollowupState | None:
    if not health:
        return None
    record = health.get("followup")
    if not isinstance(record, dict):
        return None
    # The record also describes how the loop ended (`{"passes", "stopped"}`); only a waiting pass
    # has the shape the strip prints, and an ended loop has no next pass to show.
    if "pass" not in record or "of" not in record:
        return None
    pending = record.get("pending")
    pending_s2 = pending.get("s2") if isinstance(pending, dict) else None
    return FollowupState.model_validate(
        {
            "pass": record.get("pass"),
            "of": record.get("of"),
            "next_at": record.get("next_at"),
            "pending_s2": pending_s2,
        }
    )


def _summary(run: Run) -> RunSummary:
    health = run.health or {}
    healthy = health.get("healthy")
    return RunSummary(
        id=run.id,
        started_at=run.started_at,
        finished_at=run.finished_at,
        status=run.status,
        healthy=healthy if isinstance(healthy, bool) else None,
        problems=_strings(health.get("problems")),
        warnings=_strings(health.get("warnings")),
        counts=RunCounts.model_validate(run.counts or {}),
        tokens=run.tokens,
        followup=_followup(run.health),
    )


def _letter(letter: DeadLetter) -> DeadLetterData:
    return DeadLetterData(
        source=letter.source,
        ref=letter.ref[:DEAD_LETTER_TEXT_MAX],
        error=letter.error[:DEAD_LETTER_TEXT_MAX],
        attempts=letter.attempts,
        created_at=letter.created_at,
        resolved_at=letter.resolved_at,
    )


def make_runs_router(session_factory: Callable[[], Session], search_mode: ModeGate) -> APIRouter:
    """Build the router with the read factory and the mode gate, so the app owns both."""
    router = APIRouter()

    @router.get("/runs", response_model=RunsData)
    def runs(
        limit: Annotated[int, Query(ge=1, le=RUNS_MAX_LIMIT)] = RUNS_DEFAULT_LIMIT,
    ) -> RunsData:
        with session_factory() as session:
            return RunsData(runs=[_summary(run) for run in Repository(session).recent_runs(limit)])

    @router.get("/runs/{run_id}", response_model=RunDetail)
    def run_detail(run_id: UUID) -> RunDetail:
        with session_factory() as session:
            repo = Repository(session)
            run = repo.run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            summary = _summary(run)
            return RunDetail(
                **summary.model_dump(),
                window_start=run.window_start,
                window_end=run.window_end,
                dead_letters=[_letter(letter) for letter in repo.run_dead_letters(run_id)],
            )

    @router.get("/today", response_model=TodayData)
    def today() -> TodayData:
        # The mode gate reads /proc and the run table with its own cache, so a page load pays for
        # it once and not per section.
        decision = search_mode.current()
        with session_factory() as session:
            latest = Repository(session).recent_runs(1)
        return TodayData(
            run=_summary(latest[0]) if latest else None,
            search=TodaySearch(mode=decision.mode, reason=decision.reason, since=decision.since),
        )

    return router
