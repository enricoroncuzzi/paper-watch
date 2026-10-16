import dataclasses
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from paperwatch.admission import AdmissionConfig, admit, basis_improves, promote
from paperwatch.config import (
    ARXIV_OAI_OVERLAP,
    FEED_OVERLAP,
    FEED_STALE_AFTER,
    HF_RECHECK_HORIZON,
    INGEST_LOCK_KEY,
    MAX_ITEM_ATTEMPTS,
    S2_MAX_PAPERS_PER_RUN,
    S2_RECHECK_HORIZON,
)
from paperwatch.domain import Classification, Signals, SourceItem
from paperwatch.ingest.health import assess_run
from paperwatch.llm.classifier import Classifier
from paperwatch.llm.embedder import Embedder, embedding_text
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.sources.arxiv import ArxivOaiSource
from paperwatch.sources.feeds import Feed, FeedSource
from paperwatch.sources.hf_daily import HfDailySource
from paperwatch.sources.semantic_scholar import SemanticScholarSource
from paperwatch.store.repository import Repository

log = structlog.get_logger()


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
    classifier: Classifier,
    embedder: Embedder,
    ledger: Ledger,
    admission: AdmissionConfig,
    now: datetime,
    default_lookback: timedelta = timedelta(days=7),
    since_override: datetime | None = None,
) -> RunResult:
    repo = Repository(session)
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
                "text_updated": 0,
                "malformed": 0,
                "signal_failures": 0,
                "failed": 0,
                "retried": 0,
                "given_up": 0,
                "duplicate_url": 0,
            }
            degraded = False
            stuck: list[str] = []  # cursors that could not advance this run, for the health verdict
            fetched: list[SourceItem] = []
            # Keys phase 1 parked this run; resolving one is not a retry.
            parked: set[tuple[str, str]] = set()

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
                            degraded = True
                    if day < now.date():
                        # A completed past day advances the cursor to the next day's midnight;
                        # today is never marked complete.
                        repo.advance_cursor("arxiv", _midnight(day) + timedelta(days=1))
                    session.commit()
                    # A completed day is a source that answered: any open outage letter is over.
                    repo.resolve_open_dead_letters("arxiv", "window", at=now)
                    session.commit()
            except Exception as exc:  # noqa: BLE001 - one source must not abort the run
                degraded = True
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
                try:
                    feed_fetched += sources.feeds.fetch_one(feed, since, now)
                    repo.advance_cursor(key, now)
                    # The feed answered: any earlier outage letter for it is over.
                    repo.resolve_open_dead_letters("feed", feed.url, at=now)
                    newest = sources.feeds.newest.get(feed.url)
                    if newest is None or now - newest > FEED_STALE_AFTER:
                        # Answers, but publishes nothing: looks like a quiet day unless flagged.
                        stale.append(feed.url)
                except Exception as exc:  # noqa: BLE001
                    degraded = True
                    stuck.append(key)
                    if not repo.has_open_dead_letter("feed", feed.url):
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
                    degraded = True
            _park_items(repo, run_id, feed_fetched, parked)
            session.commit()

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
                    degraded = True
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
                                embedder.embed([embedding_text(item)])[0]
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
                    cls = classifier.classify(item)
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
                    degraded = True
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
            if status != "budget_exhausted":
                targets = repo.recheck_targets(
                    now, S2_RECHECK_HORIZON, also=new_ids, paper_limit=S2_MAX_PAPERS_PER_RUN
                )
                counts["rechecked"] = len(targets)
                paper_ids = sorted(
                    {t.item.external_id for t in targets if t.item.source == "arxiv"}
                )
                upvotes: dict[str, int] = {}
                s2: dict[str, Signals] = {}
                checked: set[str] = set()
                if paper_ids:
                    # Re-read the last 30 days of Hugging Face listings: about 31-62 requests
                    # whatever the corpus size; max() in the store keeps a paper's upvotes current
                    # while it is listed.
                    upvotes = sources.hf.upvotes((now - HF_RECHECK_HORIZON).date(), now.date())
                    s2 = sources.s2.lookup(paper_ids)
                    # Only ids whose chunk Semantic Scholar answered count as checked; a refused
                    # chunk leaves its papers due, so a congested day delays evidence, never loses
                    # it.
                    checked = sources.s2.checked_ids
                    counts["s2_unchecked"] = len(set(paper_ids) - checked)
                    # Signal sources degrade instead of raising (Task 4). A degraded call is
                    # recorded so the run is partial and the outage is flagged every morning it
                    # lasts; stored evidence is untouched, and the papers it missed stay due.
                    signal_sources: tuple[
                        tuple[str, HfDailySource | SemanticScholarSource], ...
                    ] = (
                        ("hf", sources.hf),
                        ("s2", sources.s2),
                    )
                    for name, signal_source in signal_sources:
                        if signal_source.degraded:
                            degraded = True
                            counts["signal_failures"] += 1
                            if not repo.has_open_dead_letter(name, "signals"):
                                # One letter per continuing outage, not one per run.
                                repo.add_dead_letter(
                                    run_id, name, "signals", signal_source.failure or "degraded"
                                )
                        else:
                            # The source answered this run: its earlier outage letter is over.
                            repo.resolve_open_dead_letters(name, "signals", at=now)
                    session.commit()
                new_set = set(new_ids)
                for t in targets:
                    try:
                        evidence = t.signals
                        if t.item.source == "arxiv":
                            fresh = dataclasses.replace(
                                s2.get(t.item.external_id, Signals()),
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
                                embedder.embed([embedding_text(t.item)])[0]
                                if t.tier is None
                                else None
                            )
                            repo.admit_item(t.id, tier, proposed.basis, vec, evidence, now)
                            if t.tier is not None:
                                counts["promoted"] += 1
                            else:
                                counts["stored"] += 1
                                if t.id not in new_set:
                                    counts["admitted_late"] += 1
                        session.commit()
                    except BudgetExceeded:
                        # Remaining targets keep their state; the next run re-checks them anyway.
                        session.rollback()
                        status = "budget_exhausted"
                        break
                    except Exception as exc:  # noqa: BLE001 - one item must not abort the run
                        session.rollback()
                        degraded = True
                        counts["failed"] += 1
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
        except BaseException:
            # Anything that escapes the per-item handlers (SystemExit, KeyboardInterrupt) would
            # otherwise leave the run 'running' forever; finish it as failed, then re-raise.
            session.rollback()
            repo.finish_run(run_id, "failed", counts, ledger.tokens_used, at=now)
            session.commit()
            raise

        if status == "succeeded" and degraded:
            status = "partial"
        health = assess_run(
            status,
            counts,
            ledger.tokens_used,
            ledger.max_tokens,
            stuck,
            stale,
            arxiv_expected=arxiv_expected,
        )
        repo.finish_run(run_id, status, counts, ledger.tokens_used, health.as_dict(), at=now)
        session.commit()
        log.info(
            "ingestion_finished",
            run_id=str(run_id),
            status=status,
            healthy=health.healthy,
            problems=list(health.problems),
            **counts,
        )
        return RunResult(run_id, status, counts, ledger.tokens_used)
