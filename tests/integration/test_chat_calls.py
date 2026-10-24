"""The chat_calls table and the daily call cap it backs."""

import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import text
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from paperwatch.config import CHAT_CALLS_LOCK_KEY
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
    # Eight threads, not two: with two, a scheduler that happens to serialise the count and the
    # insert hides a missing advisory lock. Eight overlap reliably.
    barrier = threading.Barrier(8)
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
        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert outcomes.count("reserved") == 1 and outcomes.count("refused") == 7
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
        with factory() as session:
            row = session.get_one(ChatCall, call_id)
            assert (row.model, row.input_tokens, row.output_tokens) == ("reserved", 0, 0)
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


def test_a_held_advisory_lock_holds_the_reservation(engine: Engine) -> None:
    # Deterministic: another connection holds the transaction lock, so the reservation must wait.
    # Without the lock line it finishes at once and this fails, whatever the scheduler does.
    factory = make_session_factory(engine)
    log = ChatCallLog(factory)
    at = datetime(2026, 10, 24, 15, tzinfo=UTC)
    holder = factory()
    holder.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": CHAT_CALLS_LOCK_KEY})
    started, done = threading.Event(), threading.Event()
    result: dict[str, tuple[int | None, int]] = {}

    def reserve() -> None:
        started.set()
        result["value"] = log.reserve_chat_call(at=at, provider="openrouter", cap=10)
        done.set()

    thread = threading.Thread(target=reserve)
    try:
        thread.start()
        assert started.wait(1)
        assert not done.wait(1.0), "the reservation ignored the advisory lock"
        holder.rollback()
        holder.close()
        assert done.wait(5), "the reservation did not finish after the lock was released"
        assert result["value"][0] is not None
    finally:
        holder.close()
        thread.join(timeout=5)
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()


def test_the_reservation_day_is_utc_not_the_clocks_zone(db_session: Session) -> None:
    # 00:30 in Berlin is still 22:30 UTC on the previous day, so a cap spent on the 16th must
    # still refuse; only UTC midnight starts the new day (mutant B1f drops astimezone(UTC)).
    repo = Repository(db_session)
    record(repo, datetime(2026, 10, 16, 1, tzinfo=UTC))
    record(repo, datetime(2026, 10, 16, 2, tzinfo=UTC))
    berlin = timezone(timedelta(hours=2))
    late = datetime(2026, 10, 17, 0, 30, tzinfo=berlin)  # 2026-10-16T22:30Z
    call_id, calls_today = repo.reserve_chat_call(at=late, provider="openrouter", cap=2)
    assert (call_id, calls_today) == (None, 2)
    # The same wall clock on the next UTC day is a fresh cap.
    tomorrow = datetime(2026, 10, 17, 0, 30, tzinfo=UTC)
    call_id, calls_today = repo.reserve_chat_call(at=tomorrow, provider="openrouter", cap=2)
    assert call_id is not None and calls_today == 1
