"""The chat_calls table and the daily call cap it backs."""

import threading
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from paperwatch.llm.chat import CallGuard, DailyCapReached
from paperwatch.llm.ledger import Ledger
from paperwatch.store.db import make_session_factory
from paperwatch.store.models import ChatCall
from paperwatch.store.repository import ChatCallLog, Repository

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
        guard.before_call("openrouter")
    assert db_session.scalar(select(func.count()).select_from(ChatCall)) == 2


def test_cap_resets_at_utc_midnight(db_session: Session) -> None:
    repo = Repository(db_session)
    record(repo, MIDNIGHT + timedelta(hours=1))
    record(repo, MIDNIGHT + timedelta(hours=2))
    tomorrow = MIDNIGHT + timedelta(days=1, seconds=1)
    guard = CallGuard(repo, Ledger(100), 2, lambda: tomorrow)
    call_id, calls_today = repo.reserve_chat_call(at=tomorrow, provider="openrouter", cap=2)
    assert call_id is not None and calls_today == 1
    # The same guard and clock are what production uses.
    guard.before_call("openrouter")


def test_chat_call_log_survives_a_rolled_back_request_session(engine: Engine) -> None:
    # The production guard writes through ChatCallLog, not through the request's own session: a
    # request that fails or whose client hangs up rolls that session back, and the calls it
    # already spent must stay counted (review S2). A plain `db_session` cannot show this because
    # the fixture itself rolls everything back.
    factory = make_session_factory(engine)
    log = ChatCallLog(factory)
    at = datetime(2026, 10, 16, 3, tzinfo=UTC)
    try:
        log.record_chat_call(
            at=at, provider="openrouter", model="m", input_tokens=1, output_tokens=2
        )
        # A separate request session that rolls back must not take the usage row with it.
        with factory() as doomed:
            doomed.add(
                ChatCall(at=at, provider="groq", model="gone", input_tokens=0, output_tokens=0)
            )
            doomed.flush()
            doomed.rollback()
        # A fresh session sees the committed row, and only that one.
        with factory() as fresh:
            assert log.chat_calls_since(MIDNIGHT) == 1
            assert fresh.scalar(select(func.count()).select_from(ChatCall)) == 1
    finally:
        # The row is committed, so it outlives the test: remove it or later runs count it.
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()


def test_chat_call_log_returns_every_connection_to_the_pool(engine: Engine) -> None:
    # Both methods open a short session, so a long answer cannot hold a pool slot between calls
    # (review N2). checkedout() is the pool's own count, so it catches a session that is only
    # released when garbage collected.
    factory = make_session_factory(engine)
    log = ChatCallLog(factory)
    at = datetime(2026, 10, 16, 4, tzinfo=UTC)
    # A delta, not zero: the engine is session-scoped, so an unrelated test's open connection must
    # not be able to fail this one (review N2).
    before = engine.pool.checkedout()
    try:
        log.chat_calls_since(MIDNIGHT)
        assert engine.pool.checkedout() == before
        log.record_chat_call(
            at=at, provider="openrouter", model="m", input_tokens=1, output_tokens=1
        )
        assert engine.pool.checkedout() == before
    finally:
        # The row is committed, so it outlives the test.
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()


def test_two_concurrent_reservations_at_799_give_exactly_one_success(engine: Engine) -> None:
    # Captain's requirement (6): the count and the insert share one transaction lock, so two
    # answers that both see 799 cannot both reach the provider.
    factory = make_session_factory(engine)
    log = ChatCallLog(factory)
    at = datetime(2026, 10, 24, 12, tzinfo=UTC)
    with factory() as session:
        for _ in range(799):
            session.add(
                ChatCall(at=at, provider="openrouter", model="m", input_tokens=0, output_tokens=0)
            )
        session.commit()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            CallGuard(log, Ledger(10_000), 800, lambda: at).before_call("openrouter")
        except DailyCapReached:
            with lock:
                outcomes.append("refused")
            return
        with lock:
            outcomes.append("reserved")

    try:
        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert sorted(outcomes) == ["refused", "reserved"]
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(ChatCall)) == 800
    finally:
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()


def test_a_failed_call_keeps_its_reserved_row(engine: Engine) -> None:
    factory = make_session_factory(engine)
    log = ChatCallLog(factory)
    at = datetime(2026, 10, 24, 13, tzinfo=UTC)
    call_id, _ = log.reserve_chat_call(at=at, provider="openrouter", cap=10)
    assert call_id is not None
    try:
        CallGuard(log, Ledger(100), 10, lambda: at).after_failure("openrouter", call_id)
        with factory() as session:
            row = session.get_one(ChatCall, call_id)
            assert (row.model, row.input_tokens, row.output_tokens) == ("failed", 0, 0)
    finally:
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()


def test_after_call_fills_the_reserved_row(engine: Engine) -> None:
    factory = make_session_factory(engine)
    log = ChatCallLog(factory)
    at = datetime(2026, 10, 24, 14, tzinfo=UTC)
    call_id, _ = log.reserve_chat_call(at=at, provider="openrouter", cap=10)
    assert call_id is not None
    message = AIMessage(
        "x",
        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        response_metadata={"model_name": "vendor/model:free"},
    )
    try:
        CallGuard(log, Ledger(1000), 10, lambda: at).after_call("openrouter", message, call_id)
        with factory() as session:
            row = session.get_one(ChatCall, call_id)
            assert (row.model, row.input_tokens, row.output_tokens) == (
                "vendor/model:free",
                5,
                3,
            )
            assert session.scalar(select(func.count()).select_from(ChatCall)) == 1
    finally:
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()
