import dataclasses
import gc
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from paperwatch.admission import AdmissionConfig, admit, basis_improves, promote
from paperwatch.config import (
    ARXIV_OAI_OVERLAP,
    FEED_OVERLAP,
    FEED_STALE_AFTER,
    FOLLOWUP_DEADLINE_HOUR,
    FOLLOWUP_DELAYS_S,
    FOLLOWUP_INTERVAL_S,
    FOLLOWUP_MAX_PASSES,
    FOLLOWUP_MAX_WINDOW,
    HF_RECHECK_HORIZON,
    INGEST_LOCK_KEY,
    MAX_ITEM_ATTEMPTS,
    S2_MAX_PAPERS_PER_RUN,
    S2_RECHECK_HORIZON,
)
from paperwatch.domain import Classification, HfPaper, Signals, SourceItem
from paperwatch.ingest.health import Health, assess_run
from paperwatch.llm.classifier import Classifier
from paperwatch.llm.embedder import Embedder, embedding_text
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.sources.arxiv import ArxivOaiSource
from paperwatch.sources.feeds import Feed, FeedSource
from paperwatch.sources.hf_daily import HfDailySource
from paperwatch.sources.semantic_scholar import SemanticScholarSource
from paperwatch.store.repository import Repository, StoredItem

log = structlog.get_logger()
# The follow-up loop's deadline is local time on the server, so it is named rather than inferred
# from the machine's zone: a move or a DST change must not shift when the window closes.
ROME = ZoneInfo("Europe/Rome")
# Builds the classifier and embedder a run uses. It is called once at the start of the run, so a
# model that cannot be built fails the run before any item is touched, and again after the
# follow-up loop released them between passes (ADR 0008).
ModelFactory = Callable[[], tuple[Classifier, Embedder]]


class ModelUnavailable(RuntimeError):
    """The run's classifier or embedder could not be built: a missing model or a wrong provider."""


@dataclass(frozen=True)
class Followup:
    """Bounds of the in-run follow-up loop (ADR 0008): how long a run may keep retrying."""

    max_passes: int = FOLLOWUP_MAX_PASSES
    interval_s: int = FOLLOWUP_INTERVAL_S
    deadline_hour: int = FOLLOWUP_DEADLINE_HOUR
    # A second bound on the window, relative to the start, so a late catch-up (00:30 Rome) cannot
    # sleep past the unit's TimeoutStartSec; the clock face alone would allow 19 hours.
    max_window: timedelta = FOLLOWUP_MAX_WINDOW

    def delay_s(self, pass_number: int) -> int:
        # One-based: the wait before the pass about to run. The first waits come from the fixed
        # ramp (a burst of minutes is common), every later one from the interval.
        if pass_number <= len(FOLLOWUP_DELAYS_S):
            return FOLLOWUP_DELAYS_S[pass_number - 1]
        return self.interval_s


class _Models:
    """The classifier and embedder a run holds, and a way to let them go between passes.

    The local model is about 520 MB resident on a 1.9 GB server shared with another service, so a
    run that waits hours between passes must not keep it: `release` drops the only references (no
    caller may hold its own) and the next use builds the pair again. Only a pass whose admission
    needs a first embedding does that, so an idle day costs the interpreter alone (ADR 0008).
    """

    def __init__(self, build: ModelFactory) -> None:
        self._build = build
        self._classifier: Classifier | None = None
        self._embedder: Embedder | None = None

    def _ensure(self) -> tuple[Classifier, Embedder]:
        if self._classifier is None or self._embedder is None:
            try:
                self._classifier, self._embedder = self._build()
            except Exception as exc:
                # A model that cannot be built is a run-level failure, never an item's problem: the
                # per-item handlers would spend every new item's attempt cap on it, and a paper
                # would be dead-lettered after three days of a broken model file (review S3).
                raise ModelUnavailable(f"could not build the run's models: {exc}") from exc
        return self._classifier, self._embedder

    def classifier(self) -> Classifier:
        return self._ensure()[0]

    def embedder(self) -> Embedder:
        return self._ensure()[1]

    def release(self) -> None:
        self._classifier = self._embedder = None
        # fastembed frees its ONNX session in a finalizer; collecting now means the memory is back
        # with the OS for the wait, not whenever CPython next decides to run one.
        gc.collect()


def _rss_mb() -> float | None:
    # What the process really holds while it waits, for the journal: the server's memory is shared
    # with another service, so the idle footprint is a number worth reading, not an assumption.
    # /proc exists on the Linux server only; elsewhere the line simply carries no figure.
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def _deadline(now: datetime, hour: int) -> datetime:
    # The window closes at a fixed local hour on the run's own day: a late catch-up that starts at
    # 23:00 must not be given a wait that only ends tomorrow evening.
    return now.astimezone(ROME).replace(hour=hour, minute=0, second=0, microsecond=0)


def _iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat(timespec="seconds")


def _effective_status(status: str, causes: set[str]) -> str:
    # `partial` is a finished run that lost some work: a degraded cause makes a succeeded run
    # partial, while a budget stop keeps the status that says why it stopped.
    return "partial" if status == "succeeded" and causes else status


def _record_signal_source(
    repo: Repository,
    run_id: uuid.UUID,
    name: str,
    source: HfDailySource | SemanticScholarSource,
    *,
    now: datetime,
    causes: set[str],
    degraded_signals: set[str],
    counts: dict[str, int],
    anomaly: str | None = None,
) -> None:
    """Apply one signal source's health to the run, in the main pass and in a follow-up pass.

    One dead letter per continuing outage, and a source that answers again closes its own: a
    follow-up pass that recovers S2 must leave nothing of the morning's refusal behind. A shape
    anomaly the run saw earlier is not a throttle, though, so a later clean call must not erase
    that finding (review S4).
    """
    if source.degraded:
        causes.add(name)
        degraded_signals.add(name)
        if not repo.has_open_dead_letter(name, "signals"):
            repo.add_dead_letter(run_id, name, "signals", source.failure or "degraded")
    elif anomaly:
        # The last call answered, but an earlier call in this run was not a refusal: the source is
        # answering again, so its outage letter closes, and the run still carries the finding.
        causes.add(name)
        degraded_signals.add(name)
        repo.resolve_open_dead_letters(name, "signals", at=now)
    else:
        causes.discard(name)
        degraded_signals.discard(name)
        repo.resolve_open_dead_letters(name, "signals", at=now)
    counts["signal_failures"] = len(degraded_signals)


def _admit_targets(
    repo: Repository,
    session: Session,
    targets: list[StoredItem],
    *,
    run_id: uuid.UUID,
    now: datetime,
    admission: AdmissionConfig,
    signals: dict[str, Signals],
    upvotes: dict[str, int],
    checked: set[str],
    new_ids: set[uuid.UUID],
    models: _Models,
    counts: dict[str, int],
    causes: set[str],
    failed: set[uuid.UUID],
) -> str:
    """Re-check evidence and admit the given targets; returns the run status to carry on with.

    Extracted so a follow-up pass can admit exactly the papers it just re-checked, without
    re-running the fetch, the classification or the dead-letter retries (ADR 0008).
    """
    status = "succeeded"
    for t in targets:
        try:
            evidence = t.signals
            if t.item.source == "arxiv":
                fresh = dataclasses.replace(
                    signals.get(t.item.external_id, Signals()),
                    hf_upvotes=upvotes.get(t.item.external_id),
                )
                evidence = repo.merge_signals(t.id, fresh)
                if t.item.external_id in checked:
                    # Committed with the admission below: a target that fails or hits
                    # the budget is rolled back to its old check time and stays due.
                    repo.mark_s2_checked([t.id], now)
            proposed = admit(
                t.item,
                Classification(t.profile, t.relevance, "stored"),
                evidence,
                admission,
            )
            tier = promote(t.tier, proposed.tier if proposed else None)
            if (
                proposed is not None
                and tier is not None
                and tier == t.tier == proposed.tier
                and basis_improves(t.basis, proposed.basis)
            ):
                # Same tier, but a lab's own report now also has community evidence.
                repo.admit_item(t.id, tier, proposed.basis, None, evidence, now)
                counts["basis_improved"] += 1
            elif proposed is not None and tier is not None and tier != t.tier:
                # promote only ever returns a changed tier when it is the proposed one.
                # Only a first admission needs an embedding; a promotion keeps the
                # vector.
                vec = (
                    models.embedder().embed([embedding_text(t.item)])[0] if t.tier is None else None
                )
                repo.admit_item(t.id, tier, proposed.basis, vec, evidence, now)
                if t.tier is not None:
                    counts["promoted"] += 1
                else:
                    counts["stored"] += 1
                    if t.id not in new_ids:
                        counts["admitted_late"] += 1
            session.commit()
        except BudgetExceeded:
            # Remaining targets keep their state; the next run re-checks them anyway.
            session.rollback()
            return "budget_exhausted"
        except ModelUnavailable:
            # A model that cannot be rebuilt is the run's problem, not this item's: charging it to
            # every item's attempt cap would dead-letter them (review S3).
            raise
        except Exception as exc:  # noqa: BLE001 - one item must not abort the run
            session.rollback()
            causes.add("admission")
            counts["failed"] += 1
            # A pass must not retry what the main pass already failed to admit; the item's own
            # attempt count is the retry policy, and it is not fed more than once a run.
            failed.add(t.id)
            if repo.record_admit_failure(t.id, repr(exc)) >= MAX_ITEM_ATTEMPTS:
                # A deterministic failure would otherwise be retried, and fail, every
                # day.
                repo.add_dead_letter(
                    run_id,
                    t.item.source,
                    t.item.external_id,
                    f"gave up after {MAX_ITEM_ATTEMPTS} admission attempts: {exc!r}",
                )
                counts["given_up"] += 1
            session.commit()
    return status


@dataclass
class Sources:
    arxiv: ArxivOaiSource
    hf: HfDailySource
    s2: SemanticScholarSource
    feeds: FeedSource
    feed_list: list[Feed]


@dataclass(frozen=True)
class RunResult:
    # None only when the run was skipped: another run held the single-run advisory lock.
    run_id: uuid.UUID | None
    status: str
    counts: dict[str, int]
    tokens: int
    # The verdict the run judged itself by, carried out so the CLI can exit 3 for a finished but
    # unhealthy run. None only when skipped, like run_id.
    health: Health | None = None


def item_to_payload(item: SourceItem) -> dict[str, object]:
    d = dataclasses.asdict(item)
    d["published_at"] = item.published_at.isoformat()
    return d


def payload_to_item(p: dict[str, object]) -> SourceItem:
    return SourceItem(
        source=str(p["source"]),
        external_id=str(p["external_id"]),
        title=str(p["title"]),
        authors=tuple(p["authors"]),  # type: ignore[arg-type]
        text=str(p["text"]),
        url=str(p["url"]),
        published_at=datetime.fromisoformat(str(p["published_at"])),
        categories=tuple(p.get("categories") or ()),  # type: ignore[arg-type]
        licence=str(p["licence"]),
        version=int(str(p.get("version") or 1)),
    )


def _midnight(day: date) -> datetime:
    # The arXiv cursor stores midnight UTC of the first datestamp day not yet complete.
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _park_items(
    repo: Repository, run_id: uuid.UUID, items: list[SourceItem], parked: set[tuple[str, str]]
) -> None:
    # Park every not-yet-stored fetched item before the cursor commits, so a process that dies
    # before phase 2 stores it does not lose it when it falls outside the next overlap. The
    # letters carry attempts=0: phase 2 loads them as pending and resolves each on success.
    for item in items:
        if item.source == "feed" and repo.has_item_url("feed", item.url):
            # A moved feed gives old posts new ids; its link shows they are already stored,
            # so do not park (and resolve) a letter for it on every run.
            continue
        item_key = (item.source, item.external_id)
        open_letter = repo.open_item_letter(item.source, item.external_id)
        if open_letter is not None:
            # A later day in the same catch-up can carry a newer version of a parked paper; keep
            # the payload current, so a crash before phase 2 resumes the newest text, not v1.
            letter_id, payload = open_letter
            try:
                parked_version = payload_to_item(payload).version
            except (KeyError, TypeError, ValueError):
                parked_version = 0  # a corrupt payload is replaced by the newest seen
            if item.version > parked_version:
                repo.refresh_dead_letter_payload(letter_id, item_to_payload(item))
            continue
        if repo.stored(item.source, item.external_id) is None and not repo.has_open_dead_letter(
            item.source, item.external_id
        ):
            repo.add_dead_letter(
                run_id,
                item.source,
                item.external_id,
                "fetched, not yet stored",
                payload=item_to_payload(item),
                attempts=0,
            )
            parked.add(item_key)


def _wall_clock() -> datetime:
    # Every other timestamp in a run comes from its `now`, so a test or a replay is
    # deterministic; finished_at is the one exception, because a duration read from two equal
    # stamps is zero and the morning check printed "took 0h00m" every morning.
    return datetime.now(UTC)


@contextmanager
def _single_run_lock(session: Session) -> Iterator[bool]:
    """Hold the single-run advisory lock for the lifetime of the `with` block.

    A session-level lock lives on one dedicated connection, so it survives the run's many
    commits; an autocommit connection keeps the holder out of an idle-in-transaction state that
    a managed Postgres could kill, silently dropping the lock.
    """
    lock_conn = session.get_bind().engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        held = bool(
            lock_conn.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": INGEST_LOCK_KEY})
        )
        try:
            yield held
        finally:
            if held:
                # A dead connection already dropped the lock; suppress the unlock error so it
                # never replaces the run's own exception.
                with suppress(Exception):
                    lock_conn.scalar(
                        text("SELECT pg_advisory_unlock(:key)"), {"key": INGEST_LOCK_KEY}
                    )
    finally:
        lock_conn.close()


def run_ingestion(
    session: Session,
    sources: Sources,
    build_models: ModelFactory,
    ledger: Ledger,
    admission: AdmissionConfig,
    now: datetime,
    default_lookback: timedelta = timedelta(days=7),
    since_override: datetime | None = None,
    clock: Callable[[], datetime] = _wall_clock,
    *,
    followup: bool = False,
    bounds: Followup | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    repo = Repository(session)
    bounds = bounds or Followup()
    with _single_run_lock(session) as held:
        if not held:
            # One ingestion at a time: leave a trace for log-based alerts, then do nothing.
            log.warning("ingestion_skipped", reason="another run holds the lock")
            return RunResult(None, "skipped", {}, 0)

        def window(key: str) -> datetime:
            return since_override or repo.cursor(key) or now - default_lookback

        keys = ["arxiv", *(f"feed:{f.url}" for f in sources.feed_list)]
        # Close rows a previous process left 'running'; the lock means none can be live now.
        repo.fail_open_runs(at=now)
        run_id = repo.start_run(min(window(k) for k in keys), now, at=now)
        session.commit()

        # A crash after this commit (a killed process, SystemExit) must not leave the run row
        # 'running'. Close it as failed and re-raise; parked letters let the next run resume.
        try:
            counts = {
                "fetched": 0,
                "fetched_arxiv": 0,
                "new": 0,
                "irrelevant": 0,
                "stored": 0,
                "admitted_late": 0,
                "promoted": 0,
                "basis_improved": 0,
                "rechecked": 0,
                "s2_unchecked": 0,
                # A subset of s2_unchecked: the ids Semantic Scholar refused rather than did
                # not know, so a congested run is told apart from a corpus S2 has no record of.
                "s2_refused": 0,
                "text_updated": 0,
                "malformed": 0,
                "signal_failures": 0,
                "failed": 0,
                "retried": 0,
                "given_up": 0,
                "duplicate_url": 0,
                # Follow-up passes used, 0 unless the run was given --followup and S2 refused.
                "followup_passes": 0,
                # Signal snapshots appended by the listing step; 0 when it had nothing to write.
                "snapshots": 0,
                # 1 only when this run stored the whole Hugging Face listing.
                "hf_listing_stored": 0,
            }
            # Why this run is degraded, one cause per problem the sources caused ("arxiv",
            # "feeds", "hf", "s2", "classification", "admission", "malformed", "payload").
            # A set rather than a flag because one cause can be undone later in the same run: a
            # follow-up pass that gets S2 answering again removes its own cause (ADR 0008).
            causes: set[str] = set()
            # The signal sources that were degraded at their last attempt, so signal_failures can
            # be recomputed instead of accumulated.
            degraded_signals: set[str] = set()
            # Papers whose S2 chunk no pass answered; refused is the subset S2 turned away.
            unchecked: set[str] = set()
            # Ids S2 refused in the main pass, and the follow-up loop's own record, if it ran.
            s2_pending: set[str] = set()
            followed: dict[str, object] | None = None
            s2_problems: list[str] = []
            # A non-refusal S2 problem (a short or unpaired answer) seen in this run. Sticky on
            # purpose: a later pass that answers cleanly must not hide it (review S4).
            s2_anomaly: str | None = None
            # Targets whose admission failed this run: a follow-up pass never retries them.
            failed_admissions: set[uuid.UUID] = set()
            stuck: list[str] = []  # cursors that could not advance this run, for the health verdict
            warnings: list[str] = []  # findings that do not make the run bad, for the verdict
            fetched: list[SourceItem] = []
            # Keys phase 1 parked this run; resolving one is not a retry.
            parked: set[tuple[str, str]] = set()

            # The run's models, releasable while a follow-up pass waits. The first build happens
            # here, before any source is asked and outside the per-item handlers: a model that
            # cannot load must fail the run (exit 1, row failed) instead of being charged to every
            # new item's attempt cap (review S3).
            models = _Models(build_models)
            models.classifier()

            # Phase 1: fetch. Each source keeps its own cursor and fails alone.
            # Days to harvest, oldest first: a cursor re-reads one overlap day back, a first run
            # starts at the lookback floor, and --since overrides both. Today is fetched but
            # never completed, because its datestamp can still grow during the day.
            arxiv_cursor = repo.cursor("arxiv")
            if since_override is not None:
                first_day = since_override.date()
            elif arxiv_cursor is not None:
                first_day = arxiv_cursor.date() - ARXIV_OAI_OVERLAP
            else:
                first_day = (now - default_lookback).date()
            days = [
                first_day + timedelta(days=offset)
                for offset in range((now.date() - first_day).days + 1)
            ]
            # arXiv announces Sunday-Thursday evenings US Eastern, so weekend datestamps are
            # normally empty; a quiet Saturday or Sunday is not a broken source.
            arxiv_expected = any(day.weekday() in (1, 2, 3, 4) for day in days)
            arxiv_stopped_early = False
            # The cursor reads above opened a transaction; a network block must never run inside
            # one, so it is ended before the first page is asked for (each later page commits its
            # own parked rows first, for the same reason).
            session.commit()
            try:
                for day in days:
                    for page in sources.arxiv.harvest(day):
                        fetched += page
                        counts["fetched_arxiv"] += len(page)
                        # Park the page before the resumption request goes out, so a killed
                        # process loses nothing it has already read.
                        _park_items(repo, run_id, page, parked)
                        session.commit()
                    for bad in sources.arxiv.rejected:
                        # One dead letter per open ref; a later day re-reads the same record and
                        # the open letter keeps it from being recorded twice.
                        if not repo.has_open_dead_letter(bad.source, bad.ref):
                            repo.add_dead_letter(run_id, bad.source, bad.ref, bad.error)
                            counts["malformed"] += 1
                            causes.add("malformed")
                    if day < now.date():
                        # A completed past day advances the cursor to the next day's midnight;
                        # today is never marked complete.
                        repo.advance_cursor("arxiv", _midnight(day) + timedelta(days=1))
                    session.commit()
                    # A completed day is a source that answered: any open outage letter is over.
                    repo.resolve_open_dead_letters("arxiv", "window", at=now)
                    session.commit()
            except Exception as exc:  # noqa: BLE001 - one source must not abort the run
                causes.add("arxiv")
                if repo.cursor("arxiv") != arxiv_cursor:
                    # A completed day moved the cursor, so the source stopped early with progress;
                    # a day that only rewrote the same cursor value is not progress.
                    arxiv_stopped_early = True
                else:
                    stuck.append("arxiv")
                # A record parsed before the failure is still recorded; completed days keep
                # their cursor and parked pages stay parked for phase 2.
                for bad in sources.arxiv.rejected:
                    if not repo.has_open_dead_letter(bad.source, bad.ref):
                        repo.add_dead_letter(run_id, bad.source, bad.ref, bad.error)
                        counts["malformed"] += 1
                if not repo.has_open_dead_letter("arxiv", "window"):
                    # One letter per outage, however many days it lasts, over the whole window.
                    repo.add_dead_letter(
                        run_id,
                        "arxiv",
                        "window",
                        repr(exc),
                        window=(_midnight(first_day), now),
                    )
                if repo.cursor("arxiv") is None:
                    # Pin the first day, so repeated failures do not slide the window forward and
                    # drop the days already missed.
                    repo.advance_cursor("arxiv", _midnight(first_day))
                session.commit()
            sources.feeds.rejected = []  # filled by fetch_one across every feed of this run
            sources.feeds.newest = {}  # newest entry date per feed, for the stale-feed warning
            stale: list[str] = []
            feed_fetched: list[SourceItem] = []
            for feed in sources.feed_list:
                key = f"feed:{feed.url}"
                feed_from = window(key)
                # Same reason as arXiv: a post can appear after its date. The unique key absorbs it.
                since = feed_from if since_override else feed_from - FEED_OVERLAP
                # `window` reads the cursor, which opened a transaction: end it before the fetch,
                # so no connection sits in a transaction for the length of a feed's HTTP call.
                session.commit()
                try:
                    items = sources.feeds.fetch_one(feed, since, now)
                    feed_fetched += items
                    # Park this feed's items in the same transaction as its cursor (the next
                    # iteration's commit, or the one after the loop, ends it), so a crash between
                    # the two can never leave a cursor ahead of items nothing holds. The arXiv
                    # path parks each page before its commit for the same reason.
                    _park_items(repo, run_id, items, parked)
                    repo.advance_cursor(key, now)
                    # The feed answered: any earlier outage letter for it is over.
                    repo.resolve_open_dead_letters("feed", feed.url, at=now)
                    newest = sources.feeds.newest.get(feed.url)
                    if newest is None or now - newest > FEED_STALE_AFTER:
                        # Answers, but publishes nothing: looks like a quiet day unless flagged.
                        stale.append(feed.url)
                except Exception as exc:  # noqa: BLE001
                    # An outage that starts now is a warning: the cursor stays pinned and the
                    # next run retries the same window, so one missed day loses nothing. Only a
                    # feed with a letter already open from an earlier run (a second run in a row,
                    # or longer) degrades the run and is listed as a stuck cursor. A letter this
                    # run opened (a follow-up pass failing again) is still the first day.
                    continuing_outage = repo.has_open_outage_from_earlier_run(
                        "feed", feed.url, run_id
                    )
                    log.warning(
                        "feed_fetch_failed",
                        feed=feed.url,
                        error=repr(exc),
                        outage="continuing" if continuing_outage else "started",
                    )
                    if continuing_outage:
                        causes.add("feeds")
                        stuck.append(key)
                    else:
                        warnings.append(f"feed missed once, retried next run: {feed.url}")
                        # One letter per outage, however many runs it lasts.
                        repo.add_dead_letter(
                            run_id, "feed", feed.url, repr(exc), window=(since, now)
                        )
                    if repo.cursor(key) is None:
                        repo.advance_cursor(key, feed_from)  # pin the un-lagged start
            fetched += feed_fetched
            for bad in sources.feeds.rejected:
                # Same rule as arXiv: one dead letter per bad entry while open, and the cursor
                # still moves.
                if not repo.has_open_dead_letter(bad.source, bad.ref):
                    repo.add_dead_letter(run_id, bad.source, bad.ref, bad.error)
                    counts["malformed"] += 1
                    causes.add("malformed")
            session.commit()

            # The one place the verdict is built: the provisional verdicts the loop writes and the
            # final one must judge the same facts the same way.
            def _verdict(block: dict[str, object] | None, extra: tuple[str, ...] = ()) -> Health:
                return assess_run(
                    # The effective status, so a provisional verdict and the final one judge the
                    # same field the row carries (`partial` while a cause remains).
                    _effective_status(status, causes),
                    counts,
                    ledger.tokens_used,
                    ledger.max_tokens,
                    stuck,
                    stale,
                    warnings=warnings,
                    arxiv_expected=arxiv_expected,
                    arxiv_stopped_early=arxiv_stopped_early,
                    followup=block,
                    extra_problems=extra,
                )

            # One entry per item: an item both re-fetched by the overlap and waiting in a dead
            # letter is processed once, as its newest version, and that attempt resolves or counts.
            retry_dl: dict[tuple[str, str], int] = {}
            pending: dict[tuple[str, str], SourceItem] = {}
            for open_id, payload in repo.open_item_dead_letters():
                try:
                    item = payload_to_item(payload)
                except (KeyError, TypeError, ValueError) as exc:
                    # A corrupt payload must not kill every run at the same line: count it as a
                    # failure and, at the attempt cap, give the letter up like a classification
                    # retry.
                    causes.add("payload")
                    counts["failed"] += 1
                    if repo.record_retry_failure(open_id, repr(exc)) >= MAX_ITEM_ATTEMPTS:
                        counts["given_up"] += 1
                    # Commit now: a later item's rollback must not undo this letter's attempt count.
                    session.commit()
                    continue
                retry_dl[(item.source, item.external_id)] = open_id
                pending[(item.source, item.external_id)] = item
            for item in fetched:
                k = (item.source, item.external_id)
                if k not in pending or item.version > pending[k].version:
                    pending[k] = item
            counts["fetched"] = len(pending)

            # Phase 2: classify each new item once; store a newer arXiv version's text.
            status = "succeeded"
            new_ids: list[uuid.UUID] = []  # this run's new candidates, admitted in phase 3
            items = list(pending.values())
            for item in items:
                k = (item.source, item.external_id)
                dl_id = retry_dl.get(k)
                try:
                    if dl_id is None and repo.has_open_dead_letter(item.source, item.external_id):
                        # Already given up in an earlier run (classification or admission): the
                        # overlap re-fetch must not start the failures again. A human resolves it.
                        continue
                    stored = repo.stored(item.source, item.external_id)
                    if (
                        stored is None
                        and item.source == "feed"
                        and repo.has_item_url("feed", item.url)
                    ):
                        # A moved feed gives its old posts new ids; the link shows they are stored.
                        counts["duplicate_url"] += 1
                        if dl_id is not None:
                            repo.resolve_dead_letter(dl_id, at=now)
                        session.commit()
                        continue
                    if stored is not None:
                        if item.version > stored.item.version:
                            # Embed first: if it fails, the old text, version and embedding stay
                            # intact. embedding_text is what the classifier scored, so vectors
                            # stay comparable.
                            vec = (
                                models.embedder().embed([embedding_text(item)])[0]
                                if stored.status == "admitted"
                                else None
                            )
                            repo.replace_text(stored.id, item, vec)
                            counts["text_updated"] += 1
                        if dl_id is not None:
                            repo.resolve_dead_letter(dl_id, at=now)
                            if k not in parked:
                                counts["retried"] += 1
                        session.commit()
                        continue
                    cls = models.classifier().classify(item)
                    relevant = cls.profile is not None and cls.relevance >= admission.min_relevance
                    item_id = repo.record_classified(
                        item, "candidate" if relevant else "irrelevant", cls.profile, cls.relevance
                    )
                    # Counted only once the row is written, so a charge that stopped the run before
                    # storing the item is not counted as new work.
                    counts["new"] += 1
                    if dl_id is not None:
                        repo.resolve_dead_letter(dl_id, at=now)
                        if k not in parked:
                            counts["retried"] += 1
                    session.commit()
                    if relevant:
                        new_ids.append(item_id)
                    else:
                        counts["irrelevant"] += 1
                except BudgetExceeded:
                    session.rollback()
                    status = "budget_exhausted"
                    # Unclassified items are already parked as attempts=0 letters by phase 1, so a
                    # later run resumes them; another letter here would restart a given-up item.
                    break
                except Exception as exc:  # noqa: BLE001 - one item must not abort the run
                    session.rollback()
                    causes.add("classification")
                    counts["failed"] += 1
                    if dl_id is None:
                        repo.add_dead_letter(
                            run_id,
                            item.source,
                            item.external_id,
                            repr(exc),
                            payload=item_to_payload(item),
                        )
                    elif repo.record_retry_failure(dl_id, repr(exc)) >= MAX_ITEM_ATTEMPTS:
                        counts["given_up"] += 1
                    session.commit()

            # Phase 3: re-check evidence and admit. Targets are this run's new candidates plus every
            # stored item whose tier can still rise, each once (the repository deduplicates by id).
            # Keyless: stored papers come only when due under S2_CADENCE, at most
            # S2_MAX_PAPERS_PER_RUN of them, most overdue first; the rest wait for the next run.
            targets: list[StoredItem] = []
            papers: dict[str, HfPaper] = {}
            # Captured beside the call it describes, so the step cannot read a later write.
            hf_degraded = False
            upvotes: dict[str, int] = {}
            if status != "budget_exhausted":
                targets = repo.recheck_targets(
                    now, S2_RECHECK_HORIZON, also=new_ids, paper_limit=S2_MAX_PAPERS_PER_RUN
                )
                counts["rechecked"] = len(targets)
                paper_ids = sorted(
                    {t.item.external_id for t in targets if t.item.source == "arxiv"}
                )
                s2: dict[str, Signals] = {}
                checked: set[str] = set()
                # `recheck_targets` opened a transaction; the two signal sources are asked over the
                # network next, so the read is committed before either call goes out.
                session.commit()
                if paper_ids:
                    # Re-read the last 30 days of Hugging Face listings: about 31-62 requests
                    # whatever the corpus size; max() in the store keeps a paper's upvotes current
                    # while it is listed, and the extras and snapshots are stored after admission.
                    papers = sources.hf.papers((now - HF_RECHECK_HORIZON).date(), now.date())
                    hf_degraded = sources.hf.degraded
                    upvotes = {key: paper.upvotes for key, paper in papers.items()}
                    s2 = sources.s2.lookup(paper_ids)
                    # Only ids whose chunk Semantic Scholar answered count as checked; a refused
                    # chunk leaves its papers due, so a congested day delays evidence, never loses
                    # it. The refused ones are what a follow-up pass asks for again.
                    checked = set(sources.s2.checked_ids)
                    s2_pending = set(sources.s2.refused_ids)
                    s2_anomaly = sources.s2.anomaly
                    unchecked = set(paper_ids) - checked
                    counts["s2_unchecked"] = len(unchecked)
                    counts["s2_refused"] = len(s2_pending)
                    # Signal sources degrade instead of raising (Task 4). A degraded call is
                    # recorded so the run is partial and the outage is flagged every morning it
                    # lasts; stored evidence is untouched, and the papers it missed stay due.
                    for name, signal_source in (("hf", sources.hf), ("s2", sources.s2)):
                        _record_signal_source(
                            repo,
                            run_id,
                            name,
                            signal_source,
                            now=now,
                            causes=causes,
                            degraded_signals=degraded_signals,
                            counts=counts,
                        )
                    session.commit()
                status = _admit_targets(
                    repo,
                    session,
                    targets,
                    run_id=run_id,
                    now=now,
                    admission=admission,
                    signals=s2,
                    upvotes=upvotes,
                    checked=checked,
                    new_ids=set(new_ids),
                    models=models,
                    counts=counts,
                    causes=causes,
                    failed=failed_admissions,
                )

            if papers:
                # Extras and snapshots are presentation data: a failure here must never cost an
                # item an admission attempt or make the run unhealthy, so it runs in its own
                # savepoint after admission.
                try:
                    with session.begin_nested():
                        counts["snapshots"] = repo.apply_hf_listing(papers, at=now)
                    session.commit()
                    # Set after the commit, so the marker means stored and not attempted. A
                    # degraded listing stores what it read but is not an observation; saying it
                    # was would let a momentum baseline count a failed day's upvotes as a gain.
                    if not hf_degraded:
                        counts["hf_listing_stored"] = 1
                except (SQLAlchemyError, UnicodeError) as exc:
                    # A lone surrogate raises UnicodeEncodeError before SQLAlchemy wraps it, so
                    # the encoding failure is caught here too and stays a warning.
                    log.warning("hf_listing_not_stored", error_type=type(exc).__name__)
                    warnings.append("hugging face extras not stored this run")

            # Follow-up passes (ADR 0008). A keyless S2 burst can last hours (refused at 06:03 and
            # again at 09:52 on 2026-10-23), and the papers it refused stay due: instead of losing
            # a day of evidence, a scheduled run waits and asks again for exactly those ids until
            # 22:00 Europe/Rome, at most `followup_max_passes` times. Nothing else is redone: no
            # arXiv, feed or HF fetch, no classification, no dead-letter retry, and only the
            # refused papers are re-checked and admitted. The flag is off by default so a manual
            # run never sits in this wait, and the first day S2 stays refused is a warning, not a
            # failure, like one missed lab-feed day. A run whose token budget is spent cannot admit
            # anything, so it does not wait either.
            if followup and status != "budget_exhausted" and s2_pending:
                stopped: str | None = None
                passes = 0
                # A pass starting at the 22:00 deadline itself is too late, and the window is also
                # capped relative to the start, so a catch-up that begins at 00:30 cannot sleep past
                # the unit's timeout.
                deadline = min(_deadline(now, bounds.deadline_hour), now + bounds.max_window)
                # The main pass's pages, parked letters and evidence are not needed while waiting;
                # dropping the references keeps a long window from holding a first-run window's
                # worth of items, which the loop re-reads from the store anyway.
                del fetched, items, pending, targets, s2, papers, upvotes
                while s2_pending and passes < bounds.max_passes:
                    delay = bounds.delay_s(passes + 1)
                    next_at = clock() + timedelta(seconds=delay)
                    if next_at.astimezone(ROME) >= deadline:
                        stopped = "time_of_day"
                        break
                    # The row is finished before the process sleeps, never left 'running': the
                    # morning check must read a run that is waiting, not one that hung, and the
                    # 25 h freshness test keeps working. Overwritten after every pass.
                    followed = {
                        "pass": passes,
                        "of": bounds.max_passes,
                        "next_at": _iso(next_at),
                        "pending": {"s2": len(s2_pending)},
                    }
                    repo.finish_run(
                        run_id,
                        _effective_status(status, causes),
                        counts,
                        ledger.tokens_used,
                        _verdict(followed).as_dict(),
                        at=clock(),
                    )
                    session.commit()
                    # Nothing of the main pass is held across the wait, least of all the ~520 MB
                    # model: a pass rebuilds it only if it must embed a first admission.
                    models.release()
                    sleep(delay)
                    passes += 1
                    counts["followup_passes"] = passes
                    log.info(
                        "followup_pass",
                        pass_=passes,
                        of=bounds.max_passes,
                        waited_s=delay,
                        pending=len(s2_pending),
                        rss_mb=_rss_mb(),
                    )
                    asked = set(s2_pending)
                    answers = sources.s2.lookup(sorted(asked))
                    # A shape anomaly in any call of this run is not a throttle; it stays with the
                    # run even when this pass answered cleanly.
                    s2_anomaly = s2_anomaly or sources.s2.anomaly
                    # Same rule as the main pass: the source's own health, and its outage letter.
                    _record_signal_source(
                        repo,
                        run_id,
                        "s2",
                        sources.s2,
                        now=now,
                        causes=causes,
                        degraded_signals=degraded_signals,
                        counts=counts,
                        anomaly=s2_anomaly,
                    )
                    s2_pending = set(sources.s2.refused_ids)
                    answered = set(sources.s2.checked_ids)
                    unchecked -= answered
                    counts["s2_unchecked"] = len(unchecked)
                    counts["s2_refused"] = len(s2_pending)
                    # Fresh rows: the main pass may have admitted a paper from other evidence, so
                    # its snapshot's tier is stale. An item whose admission already failed stays
                    # failed until the next run.
                    status = _admit_targets(
                        repo,
                        session,
                        [
                            fresh
                            for eid in sorted(asked)
                            if (fresh := repo.stored("arxiv", eid)) is not None
                            and fresh.id not in failed_admissions
                        ],
                        run_id=run_id,
                        now=now,
                        admission=admission,
                        # HF listings are not re-read: this morning's upvotes are stored, and a
                        # path keeps the stored value when the fresh one is None.
                        upvotes={},
                        signals=answers,
                        checked=answered,
                        new_ids=set(new_ids),
                        models=models,
                        counts=counts,
                        causes=causes,
                        failed=failed_admissions,
                    )
                    session.commit()
                    if status != "succeeded":
                        stopped = status  # the token budget ended the run, so it ends the loop
                        break
                if stopped is None:
                    # The while ended on one of its own conditions: nothing left, or the cap.
                    stopped = "attempts" if s2_pending else "resolved"
                followed = {"passes": passes, "stopped": stopped}
                if s2_pending:
                    if passes:
                        plural = "es" if passes != 1 else ""
                        message = (
                            f"s2 still refused after {passes} follow-up pass{plural}: "
                            f"{len(s2_pending)} ids"
                        )
                    else:
                        # A run that starts after the deadline (a late Persistent catch-up) gets no
                        # pass at all, and must say that rather than claim zero passes failed.
                        message = (
                            f"s2 refused and no pass fit before {bounds.deadline_hour}:00: "
                            f"{len(s2_pending)} ids"
                        )
                    if repo.has_open_outage_from_earlier_run("s2", "signals", run_id):
                        # A second run in a row with S2 refusing: a real outage, not a burst.
                        causes.add("s2")
                        degraded_signals.add("s2")
                        counts["signal_failures"] = len(degraded_signals)
                        s2_problems.append(message)
                    else:
                        # First day of the outage, the same shape as one missed lab-feed day: a
                        # warning, the letter stays open, and tomorrow counts as the second day.
                        # A non-refusal anomaly is not a refusal, so it keeps its cause.
                        warnings.append(message)
                        if s2_anomaly is None:
                            causes.discard("s2")
                            degraded_signals.discard("s2")
                            counts["signal_failures"] = len(degraded_signals)
                log.info(
                    "followup_finished",
                    passes=passes,
                    stopped=stopped,
                    pending=len(s2_pending),
                    rss_mb=_rss_mb(),
                )
        except BaseException:
            # Anything that escapes the per-item handlers (SystemExit, KeyboardInterrupt) would
            # otherwise leave the run 'running' forever; finish it as failed, then re-raise.
            session.rollback()
            repo.finish_run(run_id, "failed", counts, ledger.tokens_used, at=clock())
            session.commit()
            raise

        if status == "succeeded" and causes:
            status = "partial"
        health = _verdict(followed, tuple(s2_problems))
        repo.finish_run(run_id, status, counts, ledger.tokens_used, health.as_dict(), at=clock())
        session.commit()
        log.info(
            "ingestion_finished",
            run_id=str(run_id),
            status=status,
            healthy=health.healthy,
            problems=list(health.problems),
            warnings=list(health.warnings),
            **counts,
        )
        return RunResult(run_id, status, counts, ledger.tokens_used, health)
