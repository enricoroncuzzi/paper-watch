"""When the API answers by keyword alone: a running main pass, or memory running out.

The gate reads the run record and two `/proc` files; every reader and the clock are injected, so a
test covers each rule and the two second cache without a database or a real server.
"""

from collections.abc import Callable
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
    read_child_rss: Callable[[], float | None] = lambda: None,
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
        read_child_rss=read_child_rss,
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


def test_a_live_model_child_does_not_trip_the_memory_gate() -> None:
    # Measured on the server on 2026-10-24: idle MemAvailable 1104 MB, the API 130 MB and its
    # model child 213 to 219 MB RssAnon. The child's own memory is what killing it frees, so the
    # gate must add it back and stay hybrid instead of turning keyword-only on its own child.
    decision = _gate(meminfo=_meminfo(700), psi=_psi(0.0), read_child_rss=lambda: 220.0).current()
    assert decision == ModeDecision("hybrid", "ok")


def test_the_child_memory_is_not_enough_when_the_machine_is_short() -> None:
    # 500 + 220 is still under the threshold: killing the child would not buy the machine its
    # memory back, so the search stays keyword-only and the child exits.
    decision = _gate(meminfo=_meminfo(500), psi=_psi(0.0), read_child_rss=lambda: 220.0).current()
    assert decision == ModeDecision("text_only", "memory")


def test_an_unreadable_child_status_counts_as_no_child() -> None:
    # A /proc read that fails must not make the gate trust memory it cannot see; the reader
    # answers None and the child counts as nothing.
    decision = _gate(meminfo=_meminfo(700), psi=_psi(0.0), read_child_rss=lambda: None).current()
    assert decision == ModeDecision("text_only", "memory")


def test_no_child_counts_as_no_child() -> None:
    # The default reader: an idle API with no model child gets the plain MemAvailable number.
    assert _gate(meminfo=_meminfo(700), psi=_psi(0.0)).current() == ModeDecision(
        "text_only", "memory"
    )


def test_psi_still_switches_to_keyword_only_with_a_child_present() -> None:
    # The add-back only lifts the memory figure. A machine whose processes are stalling on memory
    # is short whatever one child holds, so the pressure check runs for a resident child too.
    decision = _gate(meminfo=_meminfo(2_000), psi=_psi(9.0), read_child_rss=lambda: 220.0).current()
    assert decision == ModeDecision("text_only", "memory")


def test_a_running_ingestion_wins_over_a_large_child_figure() -> None:
    # During a main pass the answer is keyword-only even when the child's own memory would cover
    # the shortage: that memory is what the ingestion needs, so the child exits.
    since = NOW - timedelta(minutes=30)
    decision = _gate(
        since, meminfo=_meminfo(5_000), psi=_psi(0.0), read_child_rss=lambda: 5_000.0
    ).current()
    assert decision == ModeDecision("text_only", "ingestion", since)


@pytest.mark.parametrize("available", [799, 800])
def test_the_memory_threshold_is_inclusive_at_eight_hundred_mb(available: int) -> None:
    decision = _gate(meminfo=_meminfo(available), psi=_psi(0.0)).current()
    assert decision.mode == ("text_only" if available < 800 else "hybrid")
