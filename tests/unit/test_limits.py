"""The per-client limiter behind `/ask`: a minute window, a UTC-day count and a concurrency slot.

Every test drives the limiter with an injected clock, so a window boundary or a UTC midnight is
exact and the suite never sleeps.
"""

from datetime import UTC, datetime, timedelta

from paperwatch.api.limits import Admitted, Refused, RequestLimiter

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _limiter(
    clock: Clock,
    *,
    per_minute: int = 6,
    per_day: int = 100,
    concurrent: int = 2,
) -> RequestLimiter:
    return RequestLimiter(
        per_minute=per_minute, per_day=per_day, concurrent=concurrent, clock=clock
    )


def test_six_a_minute_then_429_with_retry_after() -> None:
    clock = Clock()
    limiter = _limiter(clock, concurrent=10)
    admitted = [limiter.admit() for _ in range(6)]
    assert all(isinstance(a, Admitted) for a in admitted)
    refused = limiter.admit()
    assert isinstance(refused, Refused)
    assert refused.reason == "rate_limited"
    # The oldest admitted call leaves the window in 60 seconds.
    assert refused.retry_after_s == 60
    # The window slides: a call made 30 seconds ago still counts at 45 s, and leaves at 90 s.
    clock.now += timedelta(seconds=45)
    still_refused = limiter.admit()
    assert isinstance(still_refused, Refused)
    assert still_refused.retry_after_s == 15
    clock.now += timedelta(seconds=45)
    assert isinstance(limiter.admit(), Admitted)


def test_a_hundred_a_utc_day_resets_at_midnight() -> None:
    clock = Clock(datetime(2026, 10, 16, 23, 30, tzinfo=UTC))
    limiter = _limiter(clock, per_minute=1000, per_day=100, concurrent=1000)
    for _ in range(100):
        assert isinstance(limiter.admit(), Admitted)
    refused = limiter.admit()
    assert isinstance(refused, Refused)
    assert refused.reason == "rate_limited"
    # The cap is a UTC day, so the reset is midnight UTC: 30 minutes from now.
    assert refused.retry_after_s == 1800
    clock.now = datetime(2026, 10, 17, 0, 0, tzinfo=UTC)
    assert isinstance(limiter.admit(), Admitted)


def test_two_concurrent_then_busy_until_one_releases() -> None:
    clock = Clock()
    limiter = _limiter(clock, concurrent=2)
    first = limiter.admit()
    second = limiter.admit()
    assert isinstance(first, Admitted) and isinstance(second, Admitted)
    refused = limiter.admit()
    assert isinstance(refused, Refused)
    assert refused.reason == "busy"
    assert refused.retry_after_s > 0
    first.release()
    assert isinstance(limiter.admit(), Admitted)


def test_release_is_idempotent() -> None:
    # A double release must not free a second slot: the concurrency limit is the scarce thing.
    clock = Clock()
    limiter = _limiter(clock, concurrent=2)
    first = limiter.admit()
    second = limiter.admit()
    third = limiter.admit()
    assert isinstance(third, Refused)
    assert isinstance(first, Admitted)
    first.release()
    first.release()
    assert isinstance(limiter.admit(), Admitted)
    assert isinstance(limiter.admit(), Refused)
    second.release()


def test_a_naive_clock_is_refused() -> None:
    # A naive datetime read as local time would shift both the window and the day boundary.
    import pytest

    limiter = _limiter(Clock(datetime(2026, 10, 16, 12)))
    with pytest.raises(ValueError, match="aware"):
        limiter.admit()


def test_the_limit_can_be_reached_by_an_unknown_but_well_formed_caller() -> None:
    # The limiter keys on nothing but the window and the day: it is per process, and the caller's
    # subject is what the route already checked. This test pins that one process counts every
    # admitted request, never one counter per token.
    clock = Clock()
    limiter = _limiter(clock, concurrent=10)
    assert [type(limiter.admit()).__name__ for _ in range(7)] == ["Admitted"] * 6 + ["Refused"]
