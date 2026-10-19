"""The chat_calls table and the daily call cap it backs."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperwatch.llm.chat import CallGuard, DailyCapReached
from paperwatch.llm.ledger import Ledger
from paperwatch.store.models import ChatCall
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration
MIDNIGHT = datetime(2026, 10, 16, tzinfo=UTC)


def record(repo: Repository, at: datetime) -> None:
    repo.record_chat_call(at=at, provider="openrouter", model="m", input_tokens=1, output_tokens=1)


def test_chat_calls_since_counts_from_the_given_time(db_session: Session) -> None:
    repo = Repository(db_session)
    record(repo, MIDNIGHT - timedelta(minutes=1))
    record(repo, MIDNIGHT + timedelta(minutes=1))
    assert repo.chat_calls_since(MIDNIGHT) == 1


def test_daily_cap_blocks_before_the_call(db_session: Session) -> None:
    repo = Repository(db_session)
    record(repo, MIDNIGHT + timedelta(hours=1))
    record(repo, MIDNIGHT + timedelta(hours=2))
    guard = CallGuard(repo, Ledger(100), 2, lambda: MIDNIGHT + timedelta(hours=3))
    with pytest.raises(DailyCapReached):
        guard.before_call()
    assert db_session.scalar(select(func.count()).select_from(ChatCall)) == 2


def test_cap_resets_at_utc_midnight(db_session: Session) -> None:
    repo = Repository(db_session)
    record(repo, MIDNIGHT + timedelta(hours=1))
    record(repo, MIDNIGHT + timedelta(hours=2))
    tomorrow = MIDNIGHT + timedelta(days=1, seconds=1)
    CallGuard(repo, Ledger(100), 2, lambda: tomorrow).before_call()
