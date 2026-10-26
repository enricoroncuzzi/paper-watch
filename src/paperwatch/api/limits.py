"""The per-client request limit in front of the model: a minute window, a day, two answers.

The daily cap in `llm/chat.py` counts provider calls against the free tier. This is the layer above
it: how many answers one caller may start at all, so an authenticated loop cannot spend the cap
and leave nothing for a real question. The two scarce things are the 800 provider calls a day and
the memory one open answer holds, so both are bounded here.

The minute and the day are counted per credential subject, so one caller's loop cannot spend
another's budget. The answer slots are shared by the whole process: they protect the server's
memory, not a client's budget. In-memory on purpose: a restart is rare, and a limiter that
outlived the process would need a table the API role can write.
"""

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal

# A busy slot usually frees in seconds; this is a rough wait, not a promise.
BUSY_RETRY_AFTER_S = 5
WINDOW = timedelta(minutes=1)


@dataclass(frozen=True)
class Admitted:
    """One granted request. `release` must be called when its stream ends, once at most."""

    _release: Callable[[], None] = field(repr=False)

    def release(self) -> None:
        self._release()


@dataclass(frozen=True)
class Refused:
    """A refused request: why, and how long a client should wait before trying again."""

    reason: Literal["rate_limited", "busy"]
    retry_after_s: int


@dataclass
class _Budget:
    """One subject's sliding minute and UTC-day count."""

    minute: deque[datetime] = field(default_factory=deque)
    day: date | None = None
    day_count: int = 0


class SlidingThrottle:
    """At most `per_minute` calls per subject, for a route whose cost is one small write.

    The events route records a card's opening or dismissal, and the row is stored before anything
    else happens: a loop on one credential must not fill the table, but the route owns no model
    call, so it needs a bound and not a budget. No day count, no concurrency slot, and the same
    minute arithmetic as `RequestLimiter`, so "wait 40 s" means the same thing in both places.
    """

    def __init__(self, *, per_minute: int, clock: Callable[[], datetime]) -> None:
        self._per_minute = per_minute
        self._clock = clock
        self._lock = threading.Lock()
        # Only subjects seen in the current minute, dropped when their window empties, so a
        # rotated credential cannot grow the map without bound.
        self._windows: dict[str, deque[datetime]] = {}

    def admit(self, subject: str) -> int | None:
        """None when the call may proceed, else the seconds to wait before trying again."""
        now = self._clock()
        # A naive clock read as local time would move the window by the machine's offset.
        if now.tzinfo is None:
            raise ValueError("the clock must return an aware datetime")
        now = now.astimezone(UTC)
        with self._lock:
            for key in list(self._windows):
                window = self._windows[key]
                while window and now - window[0] >= WINDOW:
                    window.popleft()
                if not window:
                    del self._windows[key]
            window = self._windows.setdefault(subject, deque())
            if len(window) >= self._per_minute:
                return _ceil_s(window[0] + WINDOW - now)
            window.append(now)
            return None


class RequestLimiter:
    """A sliding minute and a UTC-day count per subject, and shared concurrency slots, one lock."""

    def __init__(
        self,
        *,
        per_minute: int,
        per_day: int,
        concurrent: int,
        clock: Callable[[], datetime],
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._concurrent = concurrent
        self._clock = clock
        self._lock = threading.Lock()
        # Only subjects the verifier accepted reach this map, and a subject whose minute and day
        # are both empty is dropped, so it holds the few callers seen today and no more.
        self._budgets: dict[str, _Budget] = {}
        self._active = 0

    def admit(self, subject: str) -> Admitted | Refused:
        now = self._clock()
        # A naive clock read as local time would shift both the window and the day boundary.
        if now.tzinfo is None:
            raise ValueError("the clock must return an aware datetime")
        now = now.astimezone(UTC)
        today = now.date()
        with self._lock:
            for budget in self._budgets.values():
                while budget.minute and now - budget.minute[0] >= WINDOW:
                    budget.minute.popleft()
                if budget.day != today:
                    budget.day = today
                    budget.day_count = 0
            self._budgets = {
                key: budget
                for key, budget in self._budgets.items()
                if budget.minute or budget.day_count
            }
            budget = self._budgets.get(subject, _Budget(day=today))
            if len(budget.minute) >= self._per_minute:
                oldest = budget.minute[0]
                return Refused("rate_limited", _ceil_s(oldest + WINDOW - now))
            if budget.day_count >= self._per_day:
                midnight = datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(
                    days=1
                )
                return Refused("rate_limited", _ceil_s(midnight - now))
            if self._active >= self._concurrent:
                return Refused("busy", BUSY_RETRY_AFTER_S)
            budget.minute.append(now)
            budget.day_count += 1
            self._budgets[subject] = budget
            self._active += 1
        released = threading.Lock()

        def release() -> None:
            # Idempotent per grant: a double release must not free a slot the request held once.
            if not released.acquire(blocking=False):
                return
            with self._lock:
                self._active -= 1

        return Admitted(release)


def _ceil_s(delta: timedelta) -> int:
    # At least one second: a Retry-After of 0 invites an immediate retry that would be refused.
    return max(1, math.ceil(delta.total_seconds()))
