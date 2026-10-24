"""The background queue that writes missing 30-second summaries for the visible cards.

The suggestions response returns at once; a card without a summary shows the abstract's excerpt and
says one is on its way, and this queue fills it in on a daemon thread while the reader looks at the
list. One thread, serially, on the server: a chat call is the most expensive thing this process
does, and a burst of cards must not become a burst of requests.

Two failures are not the item's fault and are not retried the same way. A provider that is down
(`ModelUnavailable`) pauses the whole queue for thirty minutes, because every card would fail the
same way and one doomed call per card is how an outage eats a day's cap; the daily free-tier cap
(`DailyCapReached`) holds everything until UTC midnight, because retrying before then cannot work.
Any other error is logged and dropped: a summary is presentation data, never a reason to crash the
API process or to fail the request that queued it.
"""

import threading
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from sqlalchemy.orm import Session

from paperwatch.llm.chat import DailyCapReached, ModelUnavailable
from paperwatch.store.models import ItemSummary
from paperwatch.store.repository import Repository
from paperwatch.suggestions.prompts import SUMMARY_PROMPT_VERSION
from paperwatch.suggestions.summary import SummaryWriter, text_hash

log = structlog.get_logger()

# A provider outage is usually measured in minutes; thirty is short enough that a card gets a real
# summary the same day and long enough that a dead provider is not asked once per page refresh.
PROVIDER_RETRY_S = 1800


class SummaryQueue:
    """One serial worker over the summary misses of the cards that were just served."""

    def __init__(
        self,
        writer_factory: Callable[[], SummaryWriter],
        session_factory: Callable[[], Session],
        clock: Callable[[], datetime],
        model: Callable[[], str],
    ) -> None:
        self._writer_factory = writer_factory
        self._session_factory = session_factory
        self._clock = clock
        # The configured chat model id: a summary written by another model is regenerated, so a
        # model change reaches the cards instead of leaving a page written by two models.
        self._model = model
        self._lock = threading.Lock()
        self._queue: deque[UUID] = deque()
        # Queued or in flight: submitting the same card twice must not queue two calls.
        self._pending: set[UUID] = set()
        # Set after the daily cap is reached: nothing is attempted before this moment.
        self._held_until: datetime | None = None
        # Set by one provider failure: every card would fail the same way, so the queue rests too.
        self._paused_until: datetime | None = None
        self._thread: threading.Thread | None = None

    def submit(self, item_ids: Sequence[UUID]) -> None:
        """Queue these cards for a summary; ones queued, in flight or paused are skipped."""
        now = self._clock()
        with self._lock:
            if self._held_until is not None and now < self._held_until:
                return
            if self._paused_until is not None and now < self._paused_until:
                return
            for item_id in item_ids:
                if item_id in self._pending:
                    continue
                self._queue.append(item_id)
                self._pending.add(item_id)
            if not self._queue:
                return
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._work, name="summary-queue", daemon=True
                )
                self._thread.start()

    @property
    def model_id(self) -> str:
        """The model a summary is written with, which the service uses to spot a stale one."""
        return self._model()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker, for tests and for a clean shutdown."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _work(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._thread = None
                    return
                item_id = self._queue.popleft()
            try:
                self._one(item_id)
            except DailyCapReached as exc:
                # The in-flight id is still pending; discarding it lets tomorrow's request queue it
                # again, which is the whole point of holding until midnight.
                with self._lock:
                    self._pending.discard(item_id)
                self._hold_until_midnight()
                log.info("summary_queue_capped", reason=str(exc))
                return
            except ModelUnavailable as exc:
                # An outage is not about this card: pause the queue once instead of spending one
                # doomed call per visible card every half hour.
                with self._lock:
                    self._paused_until = self._clock() + timedelta(seconds=PROVIDER_RETRY_S)
                    self._pending.discard(item_id)
                    for queued in self._queue:
                        self._pending.discard(queued)
                    self._queue.clear()
                    self._thread = None
                log.warning("summary_queue_unavailable", error_type=type(exc).__name__)
                return
            except Exception as exc:  # noqa: BLE001 - a summary must never take the API down
                with self._lock:
                    self._pending.discard(item_id)
                log.warning("summary_queue_failed", error_type=type(exc).__name__)
            else:
                with self._lock:
                    self._pending.discard(item_id)

    def _hold_until_midnight(self) -> None:
        with self._lock:
            now = self._clock().astimezone(UTC)
            self._held_until = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )
            # Nothing queued now is worth a call tomorrow: the page asks again on its next refresh.
            for queued in self._queue:
                self._pending.discard(queued)
            self._queue.clear()
            self._thread = None

    def _one(self, item_id: UUID) -> None:
        with self._session_factory() as session:
            repo = Repository(session)
            source = repo.summary_sources([item_id]).get(item_id)
            if source is None:
                return  # the item was deleted or is no longer stored
            title, text = source
            digest = text_hash(title, text)
            existing = repo.summaries([item_id]).get(item_id)
        expected = self._model()
        if not _needs_refresh(existing, digest=digest, model=expected):
            return
        writer = self._writer_factory()
        summary = writer.write(title=title, text=text)
        with self._session_factory() as session:
            Repository(session).upsert_summary(
                item_id,
                summary,
                text_hash=digest,
                model=expected,
                at=self._clock(),
            )
            session.commit()


def needs_summary(existing: ItemSummary | None, *, title: str, text: str, model: str) -> bool:
    """Whether a card needs a summary written: missing, or written from other inputs."""
    return _needs_refresh(existing, digest=text_hash(title, text), model=model)


def _needs_refresh(existing: ItemSummary | None, *, digest: str, model: str) -> bool:
    # A summary written from another abstract, by another model, or under an older prompt is not
    # the card this page should show: the text it describes, or the rules it was checked by, moved.
    if existing is None:
        return True
    return (
        existing.text_hash != digest
        or existing.prompt_version != SUMMARY_PROMPT_VERSION
        or existing.model != model
    )
