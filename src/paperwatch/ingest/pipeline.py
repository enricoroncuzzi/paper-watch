import dataclasses
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import structlog
from sqlalchemy.orm import Session

from paperwatch.admission import AdmissionConfig, admit, basis_improves, promote
from paperwatch.config import (
    ARXIV_ANNOUNCEMENT_LAG,
    FEED_OVERLAP,
    FEED_STALE_AFTER,
    HF_RECHECK_HORIZON,
    MAX_ITEM_ATTEMPTS,
    S2_MAX_PAPERS_PER_RUN,
    S2_RECHECK_HORIZON,
)
from paperwatch.domain import Classification, Signals, SourceItem
from paperwatch.ingest.health import assess_run
from paperwatch.llm.classifier import Classifier
from paperwatch.llm.embedder import Embedder, embedding_text
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.sources.arxiv import ArxivSource
from paperwatch.sources.feeds import Feed, FeedSource
from paperwatch.sources.hf_daily import HfDailySource
from paperwatch.sources.semantic_scholar import SemanticScholarSource
from paperwatch.store.repository import Repository

log = structlog.get_logger()


@dataclass
class Sources:
    arxiv: ArxivSource
    hf: HfDailySource
    s2: SemanticScholarSource
    feeds: FeedSource
    feed_list: list[Feed]


@dataclass(frozen=True)
class RunResult:
    run_id: uuid.UUID
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

    def window(key: str) -> datetime:
        return since_override or repo.cursor(key) or now - default_lookback

    keys = ["arxiv", *(f"feed:{f.url}" for f in sources.feed_list)]
    run_id = repo.start_run(min(window(k) for k in keys), now)
    session.commit()
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

    # Phase 1: fetch. Each source keeps its own cursor and fails alone.
    arxiv_from = window("arxiv")
    # submittedDate filters on submission, but a paper is visible only after announcement
    # (hours to days later), so re-read the lag before the cursor; an explicit --since wins.
    arxiv_since = arxiv_from if since_override else arxiv_from - ARXIV_ANNOUNCEMENT_LAG
    try:
        papers = sources.arxiv.fetch(arxiv_since, now)
        fetched += papers
        counts["fetched_arxiv"] = len(papers)
        for bad in sources.arxiv.rejected:
            # Record each unparseable entry once; the lag overlap re-reads it for days.
            if not repo.has_open_dead_letter(bad.source, bad.ref):
                repo.add_dead_letter(run_id, bad.source, bad.ref, bad.error)
                counts["malformed"] += 1
                degraded = True
        repo.advance_cursor("arxiv", now)  # advances even past a bad entry: no poison pill
    except Exception as exc:  # noqa: BLE001 - one source must not abort the run
        degraded = True
        stuck.append("arxiv")
        repo.add_dead_letter(run_id, "arxiv", "window", repr(exc), window=(arxiv_since, now))
        if repo.cursor("arxiv") is None:
            # Pin the un-lagged start, so repeated failures do not push the window back.
            repo.advance_cursor("arxiv", arxiv_from)
    sources.feeds.rejected = []  # filled by fetch_one across every feed of this run
    sources.feeds.newest = {}  # newest entry date per feed, for the stale-feed warning
    stale: list[str] = []
    for feed in sources.feed_list:
        key = f"feed:{feed.url}"
        feed_from = window(key)
        # Same reason as arXiv: a post can appear after its date. The unique key absorbs it.
        since = feed_from if since_override else feed_from - FEED_OVERLAP
        try:
            fetched += sources.feeds.fetch_one(feed, since, now)
            repo.advance_cursor(key, now)
            newest = sources.feeds.newest.get(feed.url)
            if newest is None or now - newest > FEED_STALE_AFTER:
                # Answers, but publishes nothing: looks like a quiet day unless flagged.
                stale.append(feed.url)
        except Exception as exc:  # noqa: BLE001
            degraded = True
            stuck.append(key)
            repo.add_dead_letter(run_id, "feed", feed.url, repr(exc), window=(since, now))
            if repo.cursor(key) is None:
                repo.advance_cursor(key, feed_from)  # pin the un-lagged start
    for bad in sources.feeds.rejected:
        # Same rule as arXiv: one dead letter per bad entry while open, the cursor still moves.
        if not repo.has_open_dead_letter(bad.source, bad.ref):
            repo.add_dead_letter(run_id, bad.source, bad.ref, bad.error)
            counts["malformed"] += 1
            degraded = True
    session.commit()

    # One entry per item: an item both re-fetched by the overlap and waiting in a dead letter
    # is processed once, as its newest version, and that one attempt resolves or counts.
    retry_dl: dict[tuple[str, str], int] = {}
    pending: dict[tuple[str, str], SourceItem] = {}
    for open_id, payload in repo.open_item_dead_letters():
        item = payload_to_item(payload)
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
    for idx, item in enumerate(items):
        k = (item.source, item.external_id)
        dl_id = retry_dl.get(k)
        try:
            if dl_id is None and repo.has_open_dead_letter(item.source, item.external_id):
                # Already given up in an earlier run (classification or admission): the
                # overlap re-fetch must not start the failures again. A human resolves it.
                continue
            stored = repo.stored(item.source, item.external_id)
            if stored is None and item.source == "feed" and repo.has_item_url("feed", item.url):
                # A moved feed gives its old posts new ids; the link shows they are stored.
                counts["duplicate_url"] += 1
                if dl_id is not None:
                    repo.resolve_dead_letter(dl_id)
                session.commit()
                continue
            if stored is not None:
                if item.version > stored.item.version:
                    # Embed first: if it fails, the old text, version and embedding stay intact.
                    # embedding_text is the text the classifier scored, so vectors stay comparable.
                    vec = (
                        embedder.embed([embedding_text(item)])[0]
                        if stored.status == "admitted"
                        else None
                    )
                    repo.replace_text(stored.id, item, vec)
                    counts["text_updated"] += 1
                if dl_id is not None:
                    repo.resolve_dead_letter(dl_id)
                    counts["retried"] += 1
                session.commit()
                continue
            counts["new"] += 1
            cls = classifier.classify(item)
            relevant = cls.profile is not None and cls.relevance >= admission.min_relevance
            item_id = repo.record_classified(
                item, "candidate" if relevant else "irrelevant", cls.profile, cls.relevance
            )
            if dl_id is not None:
                repo.resolve_dead_letter(dl_id)
                counts["retried"] += 1
            session.commit()
            if relevant:
                new_ids.append(item_id)
            else:
                counts["irrelevant"] += 1
        except BudgetExceeded:
            session.rollback()
            status = "budget_exhausted"
            # Cursors already moved past these items: park every unclassified one, so the
            # next run classifies it. A budget stop is not the item's fault (attempts=0).
            for rest in items[idx:]:
                rk = (rest.source, rest.external_id)
                if rk not in retry_dl and repo.stored(*rk) is None:
                    repo.add_dead_letter(
                        run_id,
                        rest.source,
                        rest.external_id,
                        "budget_exhausted",
                        payload=item_to_payload(rest),
                        attempts=0,
                    )
            session.commit()
            break
        except Exception as exc:  # noqa: BLE001 - one item must not abort the run
            session.rollback()
            degraded = True
            counts["failed"] += 1
            if dl_id is None:
                repo.add_dead_letter(
                    run_id, item.source, item.external_id, repr(exc), payload=item_to_payload(item)
                )
            elif repo.record_retry_failure(dl_id, repr(exc)) >= MAX_ITEM_ATTEMPTS:
                counts["given_up"] += 1
            session.commit()

    # Phase 3: re-check evidence and admit. Targets are this run's new candidates plus every
    # stored item whose tier can still rise, each once (the repository deduplicates by id).
    # Keyless: stored papers come only when due under S2_CADENCE, at most S2_MAX_PAPERS_PER_RUN
    # of them, most overdue first; the rest wait for the next run.
    targets = repo.recheck_targets(
        now, S2_RECHECK_HORIZON, also=new_ids, paper_limit=S2_MAX_PAPERS_PER_RUN
    )
    counts["rechecked"] = len(targets)
    paper_ids = sorted({t.item.external_id for t in targets if t.item.source == "arxiv"})
    upvotes: dict[str, int] = {}
    s2: dict[str, Signals] = {}
    checked: set[str] = set()
    if paper_ids:
        # Re-read the last 30 days of Hugging Face listings: about 31-62 requests whatever the
        # corpus size; max() in the store keeps a paper's upvotes current while it is listed.
        upvotes = sources.hf.upvotes((now - HF_RECHECK_HORIZON).date(), now.date())
        s2 = sources.s2.lookup(paper_ids)
        # Only ids whose chunk Semantic Scholar answered count as checked; a refused chunk
        # leaves its papers due, so a congested day delays evidence and never loses it.
        checked = sources.s2.checked_ids
        counts["s2_unchecked"] = len(set(paper_ids) - checked)
        # Signal sources degrade instead of raising (Task 4). A degraded call is recorded so the
        # run is partial and the health verdict flags the outage every morning it lasts; the
        # stored evidence is untouched, and the papers it missed stay due.
        signal_sources: tuple[tuple[str, HfDailySource | SemanticScholarSource], ...] = (
            ("hf", sources.hf),
            ("s2", sources.s2),
        )
        for name, signal_source in signal_sources:
            if signal_source.degraded:
                degraded = True
                counts["signal_failures"] += 1
                repo.add_dead_letter(run_id, name, "signals", signal_source.failure or "degraded")
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
                    # Committed with the admission below: a target that fails or hits the
                    # budget is rolled back to its old check time and stays due.
                    repo.mark_s2_checked([t.id], now)
            proposed = admit(
                t.item, Classification(t.profile, t.relevance, "stored"), evidence, admission
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
                # Only a first admission needs an embedding; a promotion keeps the vector.
                vec = embedder.embed([embedding_text(t.item)])[0] if t.tier is None else None
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
                # A deterministic failure would otherwise be retried, and fail, every day.
                repo.add_dead_letter(
                    run_id,
                    t.item.source,
                    t.item.external_id,
                    f"gave up after {MAX_ITEM_ATTEMPTS} admission attempts: {exc!r}",
                )
                counts["given_up"] += 1
            session.commit()

    if status == "succeeded" and degraded:
        status = "partial"
    health = assess_run(status, counts, ledger.tokens_used, ledger.max_tokens, stuck, stale)
    repo.finish_run(run_id, status, counts, ledger.tokens_used, health.as_dict())
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
