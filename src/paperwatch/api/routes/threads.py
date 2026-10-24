"""The threads routes: the Recent menu's list and one conversation's finished turns.

A follow-up re-fetches a thread instead of re-running anything: only finished turns are stored, so
what the page shows after a dropped stream is exactly what the server checked. These routes cost
no model call, so they are not behind the request limiter.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel
from sqlalchemy.orm import Session

from paperwatch.api.payloads import AnswerData
from paperwatch.api.responses import BEARER, GATE_RESPONSES, Refusal
from paperwatch.store.repository import Repository

# The Recent menu shows a short list; a larger page is a client bug, not a reader's need.
THREAD_LIST_LIMIT = 20


class ThreadSummaryData(BaseModel):
    """One conversation in the Recent menu: its name and how many turns it holds."""

    id: UUID
    title: str
    turns: int
    updated_at: datetime


class ThreadsData(BaseModel):
    threads: list[ThreadSummaryData]


class ThreadTurnData(BaseModel):
    """One finished turn as the page re-renders it after a dropped stream."""

    n: int
    question: str
    answer: AnswerData
    search_mode: str
    created_at: datetime


class ThreadDetailData(BaseModel):
    """One conversation with every finished turn, oldest first."""

    id: UUID
    title: str
    turns: list[ThreadTurnData]


def make_threads_router(session_factory: Callable[[], Session]) -> APIRouter:
    """Build the router from the read factory, so one connection serves one short request."""
    router = APIRouter()

    @router.get(
        "/threads",
        response_model=ThreadsData,
        dependencies=[Security(BEARER)],
        responses={**GATE_RESPONSES},
    )
    def list_threads(
        limit: int = Query(THREAD_LIST_LIMIT, ge=1, le=THREAD_LIST_LIMIT),
    ) -> ThreadsData:
        """The Recent menu: newest conversation first."""
        with session_factory() as session:
            rows = Repository(session).threads(limit)
        return ThreadsData(
            threads=[
                ThreadSummaryData(
                    id=row.id, title=row.title, turns=row.turns, updated_at=row.updated_at
                )
                for row in rows
            ]
        )

    @router.get(
        "/threads/{thread_id}",
        response_model=ThreadDetailData,
        dependencies=[Security(BEARER)],
        responses={
            **GATE_RESPONSES,
            404: {"model": Refusal, "description": "No thread with that id."},
        },
    )
    def get_thread(thread_id: UUID) -> ThreadDetailData:
        with session_factory() as session:
            row = Repository(session).thread(thread_id)
        if row is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return ThreadDetailData(
            id=row.id,
            title=row.title,
            turns=[
                ThreadTurnData(
                    n=turn.n,
                    question=turn.question,
                    answer=_answer(turn.answer),
                    search_mode=turn.search_mode,
                    created_at=turn.created_at,
                )
                for turn in row.turns
            ],
        )

    return router


def _answer(payload: Any) -> AnswerData:
    """The stored `AnswerData`, validated: a row written by an older contract stays readable."""
    return AnswerData.model_validate(payload)


__all__ = [
    "ThreadDetailData",
    "ThreadSummaryData",
    "ThreadTurnData",
    "ThreadsData",
    "make_threads_router",
]
