"""The per-client request limit in front of the model: a minute window, a day, two answers.

The daily cap in `llm/chat.py` counts provider calls against the free tier. This is the layer above
it: how many answers one caller may start at all, so an authenticated loop cannot spend the cap
and leave nothing for a real question. The two scarce things are the 800 provider calls a day and
the memory one open answer holds, so both are bounded here.

Keyed on the credential's subject by the caller (`/ask` uses the verified `Principal`); the limiter
itself only counts what the process admitted, which is enough for one API process with one
operator. In-memory on purpose: a restart is rare, and a limiter that outlived the process would
need a table the API role can write.
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


class RequestLimiter:
    """A sliding minute, a UTC-day count and a concurrency slot, under one lock."""

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
        self._minute: deque[datetime] = deque()
        self._day: date | None = None
        self._day_count = 0
        self._active = 0

    def admit(self) -> Admitted | Refused:
        now = self._clock()
        # A naive clock read as local time would shift both the window and the day boundary.
        if now.tzinfo is None:
            raise ValueError("the clock must return an aware datetime")
        now = now.astimezone(UTC)
        with self._lock:
            while self._minute and now - self._minute[0] >= WINDOW:
                self._minute.popleft()
            today = now.date()
            if self._day != today:
                self._day = today
                self._day_count = 0
            if len(self._minute) >= self._per_minute:
                oldest = self._minute[0]
                return Refused("rate_limited", _ceil_s(oldest + WINDOW - now))
            if self._day_count >= self._per_day:
                midnight = datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(
                    days=1
                )
                return Refused("rate_limited", _ceil_s(midnight - now))
            if self._active >= self._concurrent:
                return Refused("busy", BUSY_RETRY_AFTER_S)
            self._minute.append(now)
            self._day_count += 1
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
