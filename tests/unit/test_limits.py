"""The per-client limiter behind `/ask`: a minute window, a UTC-day count and a concurrency slot.

Every test drives the limiter with an injected clock, so a window boundary or a UTC midnight is
exact and the suite never sleeps.
"""

from datetime import UTC, datetime, timedelta

import pytest

from paperwatch.api.limits import Admitted, Refused, RequestLimiter, SlidingThrottle

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
    admitted = [limiter.admit("a") for _ in range(6)]
    assert all(isinstance(a, Admitted) for a in admitted)
    refused = limiter.admit("a")
    assert isinstance(refused, Refused)
    assert refused.reason == "rate_limited"
    # The oldest admitted call leaves the window in 60 seconds.
    assert refused.retry_after_s == 60
    # The window slides: a call made 30 seconds ago still counts at 45 s, and leaves at 90 s.
    clock.now += timedelta(seconds=45)
    still_refused = limiter.admit("a")
    assert isinstance(still_refused, Refused)
    assert still_refused.retry_after_s == 15
    clock.now += timedelta(seconds=45)
    assert isinstance(limiter.admit("a"), Admitted)


def test_a_hundred_a_utc_day_resets_at_midnight() -> None:
    clock = Clock(datetime(2026, 10, 16, 23, 30, tzinfo=UTC))
    limiter = _limiter(clock, per_minute=1000, per_day=100, concurrent=1000)
    for _ in range(100):
        assert isinstance(limiter.admit("a"), Admitted)
    refused = limiter.admit("a")
    assert isinstance(refused, Refused)
    assert refused.reason == "rate_limited"
    # The cap is a UTC day, so the reset is midnight UTC: 30 minutes from now.
    assert refused.retry_after_s == 1800
    clock.now = datetime(2026, 10, 17, 0, 0, tzinfo=UTC)
    assert isinstance(limiter.admit("a"), Admitted)


def test_two_concurrent_then_busy_until_one_releases() -> None:
    clock = Clock()
    limiter = _limiter(clock, concurrent=2)
    first = limiter.admit("a")
    second = limiter.admit("a")
    assert isinstance(first, Admitted) and isinstance(second, Admitted)
    refused = limiter.admit("a")
    assert isinstance(refused, Refused)
    assert refused.reason == "busy"
    assert refused.retry_after_s > 0
    first.release()
    assert isinstance(limiter.admit("a"), Admitted)


def test_release_is_idempotent() -> None:
    # A double release must not free a second slot: the concurrency limit is the scarce thing.
    clock = Clock()
    limiter = _limiter(clock, concurrent=2)
    first = limiter.admit("a")
    second = limiter.admit("a")
    third = limiter.admit("a")
    assert isinstance(third, Refused)
    assert isinstance(first, Admitted)
    first.release()
    first.release()
    assert isinstance(limiter.admit("a"), Admitted)
    assert isinstance(limiter.admit("a"), Refused)
    second.release()


def test_a_naive_clock_is_refused() -> None:
    # A naive datetime read as local time would shift both the window and the day boundary.
    import pytest

    limiter = _limiter(Clock(datetime(2026, 10, 16, 12)))
    with pytest.raises(ValueError, match="aware"):
        limiter.admit("a")


def test_each_subject_has_its_own_minute_but_shares_the_answer_slots() -> None:
    # One caller's loop must not spend another's budget, while the concurrency slots protect the
    # server's memory and so are shared by every caller.
    clock = Clock()
    limiter = _limiter(clock, concurrent=10)
    held = [limiter.admit("a") for _ in range(6)]
    assert all(isinstance(grant, Admitted) for grant in held)
    assert isinstance(limiter.admit("a"), Refused)
    second = limiter.admit("b")
    assert isinstance(second, Admitted)
    for grant in held:
        assert isinstance(grant, Admitted)
        grant.release()
    second.release()

    shared = _limiter(clock, concurrent=2)
    first = shared.admit("a")
    other = shared.admit("b")
    assert isinstance(first, Admitted) and isinstance(other, Admitted)
    busy = shared.admit("c")
    assert isinstance(busy, Refused) and busy.reason == "busy"


def test_a_subject_whose_windows_empty_is_forgotten() -> None:
    # The per-subject map holds only callers with a live minute or day, so it cannot grow without
    # bound; a forgotten subject starts again with a full budget.
    clock = Clock(datetime(2026, 10, 16, 23, 59, tzinfo=UTC))
    limiter = _limiter(clock, per_day=1, concurrent=10)
    assert isinstance(limiter.admit("a"), Admitted)
    assert isinstance(limiter.admit("a"), Refused)
    clock.now = datetime(2026, 10, 17, 0, 1, tzinfo=UTC)
    assert isinstance(limiter.admit("b"), Admitted)
    assert set(limiter._budgets) == {"b"}
    assert isinstance(limiter.admit("a"), Admitted)


def _throttle(clock: Clock, *, per_minute: int = 2) -> SlidingThrottle:
    return SlidingThrottle(per_minute=per_minute, clock=clock)


def test_a_subject_past_the_minute_bound_is_told_when_to_try_again() -> None:
    clock = Clock()
    throttle = _throttle(clock)
    assert [throttle.admit("a"), throttle.admit("a")] == [None, None]
    # The oldest call leaves the window in 60 seconds, the same arithmetic as the answer limiter.
    assert throttle.admit("a") == 60
    clock.now += timedelta(seconds=20)
    assert throttle.admit("a") == 40
    clock.now += timedelta(seconds=40)
    assert throttle.admit("a") is None


def test_a_throttled_subject_does_not_spend_another_subject_s_minute() -> None:
    # The events route is per credential, so one reader's page posting many events must not stop
    # another caller's event.
    throttle = _throttle(Clock())
    throttle.admit("a")
    throttle.admit("a")
    assert throttle.admit("a") is not None
    assert throttle.admit("b") is None


def test_the_throttle_forgets_a_subject_whose_minute_is_empty() -> None:
    # The map holds the callers seen in the current minute only, so a rotated credential cannot
    # grow it without bound.
    clock = Clock()
    throttle = _throttle(clock)
    throttle.admit("a")
    clock.now += timedelta(minutes=2)
    throttle.admit("b")
    # The map holds the callers of the current minute, so the memory bound is the claim.
    assert len(throttle._windows) == 1


def test_the_throttle_refuses_a_naive_clock() -> None:
    throttle = SlidingThrottle(per_minute=2, clock=lambda: datetime(2026, 10, 16, 12))
    with pytest.raises(ValueError):
        throttle.admit("a")
