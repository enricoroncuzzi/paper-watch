"""Whether the API may use the vector branch, or must answer by keyword alone.

The ingestion peaks near 707 MB on a machine with about 1.3 GB available, and the API's child
embedder needs the same memory. So the API reads the run record and the kernel's own memory
signals: while a main pass is running, or memory is short, a search uses full-text only and says
so. The database is the source of truth for the run; `/proc` is the source for memory, and a Mac
(no `/proc`) simply does not report pressure there.

The decision is cached for two seconds: a tool call is a few database queries and checking the run
table before every one of them would be noise, while two seconds is far shorter than a main pass.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperwatch.config import (
    MAIN_PASS_MAX_AGE,
    SEARCH_MODE_CACHE_S,
    SEARCH_MODE_MAX_PSI_SOME_AVG10,
    SEARCH_MODE_MIN_AVAILABLE_MB,
)
from paperwatch.store.models import Run

log = structlog.get_logger()

MEMINFO_PATH = Path("/proc/meminfo")
PSI_MEMORY_PATH = Path("/proc/pressure/memory")


@dataclass(frozen=True)
class ModeDecision:
    """The mode to answer in, why, and (for an ingestion) when the main pass started."""

    mode: Literal["hybrid", "text_only"]
    reason: Literal["ok", "ingestion", "memory", "database"]
    since: datetime | None = None


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def available_mb(meminfo: str | None) -> float | None:
    if meminfo is None:
        return None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            try:
                return int(fields[1]) / 1024
            except (IndexError, ValueError):
                return None
    return None


def psi_some_avg10(psi: str | None) -> float | None:
    if psi is None:
        return None
    for line in psi.splitlines():
        if not line.startswith("some "):
            continue
        for field in line.split()[1:]:
            if field.startswith("avg10="):
                try:
                    return float(field.removeprefix("avg10="))
                except ValueError:
                    return None
    return None


class SearchModeGate:
    """Choose the search mode from the run record and the kernel's memory signals."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        clock: Callable[[], datetime],
        read_meminfo: Callable[[], str | None] = lambda: _read(MEMINFO_PATH),
        read_psi: Callable[[], str | None] = lambda: _read(PSI_MEMORY_PATH),
        cache_s: float = SEARCH_MODE_CACHE_S,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock
        self._read_meminfo = read_meminfo
        self._read_psi = read_psi
        self._cache_s = cache_s
        self._lock = threading.Lock()
        self._cached: ModeDecision | None = None
        self._cached_at: datetime | None = None

    def current(self) -> ModeDecision:
        now = self._clock()
        with self._lock:
            if (
                self._cached is not None
                and self._cached_at is not None
                and now - self._cached_at < timedelta(seconds=self._cache_s)
            ):
                return self._cached
        decision = self._decide(now)
        with self._lock:
            self._cached = decision
            self._cached_at = now
        return decision

    def _decide(self, now: datetime) -> ModeDecision:
        try:
            with self._session_factory() as session:
                started_at = session.scalar(
                    select(Run.started_at)
                    .where(Run.status == "running")
                    .order_by(Run.started_at.desc())
                    .limit(1)
                )
        except Exception as exc:  # noqa: BLE001 - no database means no vector search
            # Only the type: a driver error's text can hold a host detail.
            log.warning("search_mode_read_failed", error_type=type(exc).__name__)
            return ModeDecision("text_only", "database")
        # The age is checked here as well as in the query so a stale row fetched by a caller's
        # fake session, or by a database with a skewed clock, cannot pin the mode forever.
        if started_at is not None and started_at > now.astimezone(UTC) - MAIN_PASS_MAX_AGE:
            return ModeDecision("text_only", "ingestion", started_at)

        available = available_mb(self._read_meminfo())
        if available is not None and available < SEARCH_MODE_MIN_AVAILABLE_MB:
            return ModeDecision("text_only", "memory")
        pressure = psi_some_avg10(self._read_psi())
        if pressure is not None and pressure > SEARCH_MODE_MAX_PSI_SOME_AVG10:
            return ModeDecision("text_only", "memory")
        return ModeDecision("hybrid", "ok")
