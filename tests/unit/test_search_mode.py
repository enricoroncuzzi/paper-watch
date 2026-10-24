"""When the API answers by keyword alone: a running main pass, or memory running out.

The gate reads the run record and two `/proc` files; every reader and the clock are injected, so a
test covers each rule and the two second cache without a database or a real server.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from paperwatch.api.search_mode import ModeDecision, SearchModeGate
from paperwatch.config import MAIN_PASS_MAX_AGE, SEARCH_MODE_CACHE_S

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeSession:
    """A session whose one query answers the run read, or raises like a broken database."""

    def __init__(self, started_at: Any) -> None:
        self.started_at = started_at

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def scalar(self, statement: Any) -> Any:
        if isinstance(self.started_at, Exception):
            raise self.started_at
        return self.started_at


def _gate(
    started_at: Any = None,
    *,
    clock: Clock | None = None,
    meminfo: str | None = None,
    psi: str | None = None,
    cache_s: float = SEARCH_MODE_CACHE_S,
    factory_calls: list[int] | None = None,
) -> SearchModeGate:
    clock = clock or Clock()

    def factory() -> FakeSession:
        if factory_calls is not None:
            factory_calls.append(1)
        return FakeSession(started_at)

    return SearchModeGate(
        factory,
        clock=clock,
        read_meminfo=lambda: meminfo,
        read_psi=lambda: psi,
        cache_s=cache_s,
    )


def _meminfo(available_mb: int) -> str:
    return f"MemTotal: 1900000 kB\nMemAvailable: {available_mb * 1024} kB\n"


def _psi(avg10: float) -> str:
    return f"some avg10={avg10:.2f} avg60=1.00 avg300=0.50 total=12345\nfull avg10=0.00\n"


def test_a_running_main_pass_switches_to_keyword_only() -> None:
    since = NOW - timedelta(minutes=30)
    decision = _gate(since).current()
    assert decision == ModeDecision("text_only", "ingestion", since)


def test_a_running_row_older_than_three_hours_does_not_count() -> None:
    # A row left 'running' by a killed run must not pin the API to keyword-only for ever.
    stale = NOW - MAIN_PASS_MAX_AGE - timedelta(minutes=1)
    assert _gate(stale).current() == ModeDecision("hybrid", "ok")


def test_low_memavailable_switches_to_keyword_only() -> None:
    assert _gate(meminfo=_meminfo(700), psi=_psi(0.0)).current() == ModeDecision(
        "text_only", "memory"
    )


def test_high_psi_switches_to_keyword_only() -> None:
    assert _gate(meminfo=_meminfo(1_500), psi=_psi(9.0)).current() == ModeDecision(
        "text_only", "memory"
    )


def test_missing_proc_files_are_ignored() -> None:
    # On the Mac there is no /proc; the API must still answer, in hybrid mode.
    assert _gate(None, meminfo=None, psi=None).current() == ModeDecision("hybrid", "ok")


def test_a_database_error_means_keyword_only() -> None:
    decision = _gate(RuntimeError("database down")).current()
    assert decision == ModeDecision("text_only", "database")


def test_the_decision_is_cached_for_two_seconds() -> None:
    clock = Clock()
    calls: list[int] = []
    gate = _gate(None, clock=clock, factory_calls=calls)
    assert gate.current().mode == "hybrid"
    # A second call inside the window does not touch the database.
    assert gate.current().mode == "hybrid"
    assert len(calls) == 1
    clock.now += timedelta(seconds=SEARCH_MODE_CACHE_S + 0.1)
    assert gate.current().mode == "hybrid"
    assert len(calls) == 2


@pytest.mark.parametrize("available", [799, 800])
def test_the_memory_threshold_is_inclusive_at_eight_hundred_mb(available: int) -> None:
    decision = _gate(meminfo=_meminfo(available), psi=_psi(0.0)).current()
    assert decision.mode == ("text_only" if available < 800 else "hybrid")
