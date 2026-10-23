import dataclasses
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session
from structlog.testing import capture_logs

from paperwatch.admission import load_admission
from paperwatch.config import (
    FEED_OVERLAP,
    HF_ORG_MAX_CHARS,
    HF_RECHECK_HORIZON,
    INGEST_LOCK_KEY,
    INT4_MAX,
    MAX_ITEM_ATTEMPTS,
)
from paperwatch.domain import (
    Classification,
    RejectedEntry,
    Signals,
    SourceItem,
    Tier,
    load_profiles,
)
from paperwatch.ingest.health import Health
from paperwatch.ingest.pipeline import (
    Followup,
    ModelUnavailable,
    RunResult,
    Sources,
    _park_items,
    item_to_payload,
    payload_to_item,
    run_ingestion,
)
from paperwatch.llm.classifier import Classifier, FakeClassifier
from paperwatch.llm.embedder import Embedder, FakeEmbedder
from paperwatch.llm.ledger import Ledger
from paperwatch.sources.feeds import Feed
from paperwatch.sources.hf_daily import HfPaper
from paperwatch.store.db import make_session_factory
from paperwatch.store.models import DeadLetter, Item, Run, SignalSnapshot, TierHistory
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)
PROFILES = load_profiles(Path("config/profiles.yaml"))
ADM = load_admission(Path("config/admission.yaml"))
FEED = Feed(lab="TestLab", url="https://lab.example/feed.xml", region="us-eu")
OTHER_FEED = Feed(lab="OtherLab", url="https://other.example/feed.xml", region="us-eu")


def paper(eid: str, title: str, version: int = 1) -> SourceItem:
    return SourceItem(
        "arxiv",
        eid,
        title,
        ("A",),
        f"{title}. We study tool use in LLM agent harnesses.",
        f"https://arxiv.org/abs/{eid}",
        NOW - timedelta(days=2),
        version=version,
    )


def off_topic(eid: str) -> SourceItem:
    # No profile term at all, so the fake classifier returns no profile: irrelevant.
    return SourceItem(
        "arxiv",
        eid,
        "Sourdough fermentation kinetics",
        ("B",),
        "Yeast activity and flour hydration in bread dough.",
        f"https://arxiv.org/abs/{eid}",
        NOW - timedelta(days=2),
    )


def blog(eid: str) -> SourceItem:
    return SourceItem(
        "feed",
        eid,
        "Building an agent harness",
        ("TestLab",),
        "How we built a coding agent harness with tool use.",
        f"https://lab.example/{eid}",
        NOW - timedelta(days=1),
        licence="copyright (private search only)",
    )


class CountingClassifier(FakeClassifier):
    """Counts calls, so tests can prove the AI relevance check runs once per item."""

    def __init__(self) -> None:
        super().__init__(PROFILES)
        self.calls = 0

    def classify(self, item: SourceItem) -> Classification:
        self.calls += 1
        return super().classify(item)


class FlakyEmbedder(FakeEmbedder):
    """Fails its first `failures` calls, then embeds normally; counts every call."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures, self.calls = failures, 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("embedding service 503")
        return super().embed(texts)


def _serve_papers(s: Sources) -> Callable[[date], Iterator[list[SourceItem]]]:
    # A fake harvest that serves the source's current `papers` as one page every day; the
    # pipeline dedupes by (source, external_id), so a run still stores each item once.
    def harvest(day: date) -> Iterator[list[SourceItem]]:
        return iter([s.arxiv.papers] if s.arxiv.papers else [])

    return harvest


def _pages_for(
    pages: dict[date, list[list[SourceItem]]],
) -> Callable[[date], Iterator[list[SourceItem]]]:
    # A day missing from the map has no records; a present day yields its pages in order.
    def harvest(day: date) -> Iterator[list[SourceItem]]:
        return iter(pages.get(day, []))

    return harvest


def fake_sources(papers: list[SourceItem], feed_items: list[SourceItem] | None = None) -> Sources:
    s = Sources(
        arxiv=MagicMock(), hf=MagicMock(), s2=MagicMock(), feeds=MagicMock(), feed_list=[FEED]
    )
    s.arxiv.papers = list(papers)  # mutable: a test changes it between runs
    s.arxiv.rejected = []  # filled by the real client when a record cannot be parsed
    s.arxiv.harvest.side_effect = _serve_papers(s)
    # Signal sources report outages through these flags instead of raising (Task 4).
    s.hf.degraded, s.hf.failure = False, None
    s.s2.degraded, s.s2.failure = False, None
    # A non-refusal answer problem (a shape anomaly); like the real client, it describes one call.
    s.s2.anomaly = None
    s.hf.papers.return_value = {p.external_id: HfPaper(upvotes=50) for p in papers}
    s.s2.lookup.return_value = {}
    s.s2.refused = set()  # ids whose Semantic Scholar chunk the stub refuses
    s.s2.refused_ids = set()  # like the real client, reset on every lookup

    def s2_lookup(ids: list[str]) -> object:
        # Like the real client: ids in an answered chunk are checked. A test refuses chunks
        # through `refused`, or every chunk by setting `degraded` alone (a whole outage).
        outage = s.s2.degraded and not s.s2.refused
        s.s2.checked_ids = set() if outage else set(ids) - s.s2.refused
        s.s2.refused_ids = set(ids) if outage else set(s.s2.refused) & set(ids)
        if not s.s2.degraded:
            # The real client resets its per-call anomaly on every lookup; only a test that leaves
            # the source degraded keeps the one it set.
            s.s2.anomaly = None
        # Only an answered id can carry a fresh answer; a refused one keeps the stored evidence.
        answers: dict[str, Signals] = s.s2.lookup.return_value or {}
        return {k: v for k, v in answers.items() if k in s.s2.checked_ids}

    s.s2.lookup.side_effect = s2_lookup
    s.feeds.fetch_one.return_value = feed_items or []
    s.feeds.rejected = []  # filled by the real client when a feed entry cannot be parsed
    s.feeds.newest = {}  # the pipeline resets it before its feed loop

    def live_feed(feed: Feed, since: datetime, until: datetime) -> object:
        s.feeds.newest[feed.url] = until  # a live feed: its newest post is recent
        return DEFAULT  # then the mock returns fetch_one.return_value as usual

    s.feeds.fetch_one.side_effect = live_feed
    return s


def run(
    session: Session,
    sources: Sources,
    now: datetime = NOW,
    classifier: Classifier | None = None,
    embedder: Embedder | None = None,
    ledger: Ledger | None = None,
    since_override: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
    followup: bool = False,
    bounds: Followup | None = None,
    sleep: Callable[[float], None] | None = None,
) -> RunResult:
    return run_ingestion(
        session,
        sources,
        lambda: (classifier or FakeClassifier(PROFILES), embedder or FakeEmbedder()),
        ledger or Ledger(10**9),
        ADM,
        now,
        since_override=since_override,
        # A test that does not care about the duration finishes in the same instant it starts.
        clock=clock or (lambda: now),
        followup=followup,
        bounds=bounds,
        sleep=sleep or (lambda _seconds: None),
    )


def row(session: Session, eid: str) -> Item:
    return session.scalars(select(Item).where(Item.external_id == eid)).one()


def history(session: Session) -> list[TierHistory]:
    return list(session.scalars(select(TierHistory).order_by(TierHistory.id)))


def test_stores_admitted_items(db_session: Session) -> None:
    res = run(
        db_session,
        fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")]),
    )
    assert res.status == "succeeded" and res.counts["stored"] == 2
    # A clean run resolves every parked letter it wrote.
    assert Repository(db_session).open_item_dead_letters() == []


def test_rerun_is_idempotent_and_refreshes_signals(db_session: Session) -> None:
    clf = CountingClassifier()
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src, classifier=clf)
    src.s2.lookup.return_value = {
        "2510.1": Signals(
            venue="International Conference on Learning Representations",
            venue_aliases=("ICLR",),
            citation_count=12,
        )
    }
    res = run(db_session, src, now=NOW + timedelta(days=1), classifier=clf)
    assert res.counts["new"] == 0 and res.counts["stored"] == 0 and res.counts["promoted"] == 1
    assert clf.calls == 1  # the overlap re-fetch never reaches the AI relevance check
    r = row(db_session, "2510.1")
    assert r.tier == "peer_reviewed" and r.citation_count == 12 and r.hf_upvotes == 50
    assert [(h.from_tier, h.to_tier) for h in history(db_session)] == [
        (None, "vetted_preprint"),
        ("vetted_preprint", "peer_reviewed"),
    ]
    assert len(db_session.scalars(select(Item)).all()) == 1


def test_rerun_does_not_reclassify_rejected_papers(db_session: Session) -> None:
    clf = CountingClassifier()
    src = fake_sources([paper("2510.1", "Agent harness"), off_topic("2510.2")])
    src.hf.papers.return_value = {}  # no evidence: 2510.1 stays a candidate
    for day in range(8):  # the whole announcement-lag overlap re-sends both papers daily
        run(db_session, src, now=NOW + timedelta(days=day), classifier=clf)
    assert clf.calls == 2
    statuses = dict(db_session.execute(select(Item.external_id, Item.status)).all())
    assert statuses == {"2510.1": "candidate", "2510.2": "irrelevant"}


def test_candidate_admitted_when_upvotes_arrive_late(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {"2510.1": HfPaper(upvotes=3)}
    first = run(db_session, src)
    assert first.counts["stored"] == 0 and row(db_session, "2510.1").status == "candidate"
    day20 = NOW + timedelta(days=20)
    src.arxiv.papers = []  # long out of the arXiv fetch window
    src.hf.papers.return_value = {"2510.1": HfPaper(upvotes=14)}
    res = run(db_session, src, now=day20)
    assert src.hf.papers.call_args.args == ((day20 - HF_RECHECK_HORIZON).date(), day20.date())
    assert res.counts["admitted_late"] == 1 and res.counts["stored"] == 1
    r = row(db_session, "2510.1")
    assert (r.status, r.tier, r.hf_upvotes) == ("admitted", "vetted_preprint", 14)
    [h] = history(db_session)
    assert (h.from_tier, h.to_tier, h.basis, h.changed_at) == (
        None,
        "vetted_preprint",
        "hf_upvotes",
        day20,
    )
    assert h.evidence["hf_upvotes"] == 14


def test_a_run_stores_hf_extras_for_listed_papers(db_session: Session) -> None:
    # The listing step is the one write that turns the daily page into stored extras and a
    # snapshot, so the page and the ranking read them without another request.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {
        "2510.1": HfPaper(
            upvotes=50,
            comments=4,
            github_repo="https://github.com/a/b",
            github_stars=120,
            project_page="https://example.com/p",
            organization="MirroS",
            listed_at=NOW - timedelta(days=1),
        )
    }
    res = run(db_session, src)
    assert res.counts["snapshots"] == 1
    r = row(db_session, "2510.1")
    assert (
        r.hf_upvotes,
        r.hf_comments,
        r.github_repo,
        r.github_stars,
        r.project_page,
        r.hf_org,
        r.hf_listed_at,
    ) == (
        50,
        4,
        "https://github.com/a/b",
        120,
        "https://example.com/p",
        "MirroS",
        NOW - timedelta(days=1),
    )
    [snapshot] = db_session.scalars(
        select(SignalSnapshot).where(SignalSnapshot.item_id == r.id)
    ).all()
    assert (snapshot.observed_at, snapshot.hf_upvotes, snapshot.github_stars) == (NOW, 50, 120)


def test_a_bad_extra_row_does_not_sink_the_listing_step(db_session: Session) -> None:
    # The real store, not a stub: a number and a name the columns cannot hold are sanitised, and
    # the other paper's extras and snapshot still land.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {
        "2510.1": HfPaper(
            upvotes=10,
            github_stars=INT4_MAX + 1,
            organization="O" * (HF_ORG_MAX_CHARS + 44),
        ),
        "2510.2": HfPaper(upvotes=20, github_repo="https://github.com/a/b", github_stars=42),
    }
    res = run(db_session, src)
    assert res.counts["snapshots"] == 2
    bad = row(db_session, "2510.1")
    assert (bad.github_stars, bad.hf_org) == (None, "O" * HF_ORG_MAX_CHARS)
    good = row(db_session, "2510.2")
    assert (good.hf_upvotes, good.github_repo, good.github_stars) == (
        20,
        "https://github.com/a/b",
        42,
    )
    snapshots = db_session.scalars(
        select(SignalSnapshot).where(SignalSnapshot.item_id == good.id)
    ).all()
    assert len(snapshots) == 1 and snapshots[0].github_stars == 42


def test_a_surrogate_in_the_listing_is_a_warning_not_a_failed_run(db_session: Session) -> None:
    # A lone surrogate cannot be encoded into the database at all, so psycopg raises a raw
    # UnicodeEncodeError rather than a SQLAlchemyError. A value that skipped the client must still
    # cost only the step, never the run.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {"2510.1": HfPaper(upvotes=50, organization="a\ud800b")}
    with capture_logs() as logs:
        res = run(db_session, src)
    assert res.status == "succeeded"
    assert res.health is not None and res.health.healthy
    assert "hugging face extras not stored this run" in res.health.warnings
    [failed] = [entry for entry in logs if entry.get("event") == "hf_listing_not_stored"]
    assert failed["error_type"] == "UnicodeEncodeError"


def test_a_real_database_error_in_the_listing_step_is_a_warning(db_session: Session) -> None:
    # Extras are presentation data: a write the database really refuses must not cost an item an
    # admission attempt or make the run unhealthy, and the failure must be logged. The NUL is fed
    # straight to the repository (stubbed papers, bypassing the client, which now rejects it), so
    # the error comes from Postgres and not from a stub.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {"2510.1": HfPaper(upvotes=50, organization="bad\x00org")}
    with capture_logs() as logs:
        res = run(db_session, src)
    assert res.status == "succeeded"
    assert res.health is not None and res.health.healthy
    assert "hugging face extras not stored this run" in res.health.warnings
    assert [item.admit_attempts for item in db_session.scalars(select(Item)).all()] == [0]
    [failed] = [entry for entry in logs if entry.get("event") == "hf_listing_not_stored"]
    assert failed["error_type"] == "DataError"


def test_lab_report_admitted_same_day_then_basis_improves(db_session: Session) -> None:
    # A lab team's report is admitted on day one by authorship; when community evidence
    # arrives the tier stays, but the recorded basis says it is no longer authority alone.
    lab = dataclasses.replace(paper("2510.1", "Agent harness"), authors=("DeepSeek-AI",))
    src = fake_sources([lab])
    src.hf.papers.return_value = {}
    first = run(db_session, src)
    assert first.counts["stored"] == 1
    assert (row(db_session, "2510.1").tier, row(db_session, "2510.1").admission_basis) == (
        "vetted_preprint",
        "lab_author",
    )
    day3 = NOW + timedelta(days=3)
    src.arxiv.papers = []
    src.hf.papers.return_value = {"2510.1": HfPaper(upvotes=120)}
    res = run(db_session, src, now=day3)
    assert res.counts["basis_improved"] == 1 and res.counts["promoted"] == 0
    assert row(db_session, "2510.1").admission_basis == "hf_upvotes"
    assert [(h.from_tier, h.to_tier, h.basis) for h in history(db_session)] == [
        (None, "vetted_preprint", "lab_author"),
        ("vetted_preprint", "vetted_preprint", "hf_upvotes"),
    ]
    again = run(db_session, src, now=day3 + timedelta(days=1))
    assert again.counts["basis_improved"] == 0  # recorded once, not every run


def test_vetted_preprint_promoted_when_venue_appears_months_later(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = {"2510.1": Signals(venue="arXiv.org", citation_count=6)}
    run(db_session, src)
    day90 = NOW + timedelta(days=90)
    src.arxiv.papers = []
    src.s2.lookup.return_value = {
        "2510.1": Signals(
            venue="International Conference on Learning Representations",
            venue_aliases=("ICLR",),
            citation_count=9,
        )
    }
    res = run(db_session, src, now=day90)
    assert res.counts["promoted"] == 1
    assert [(h.from_tier, h.to_tier, h.basis, h.changed_at) for h in history(db_session)] == [
        (None, "vetted_preprint", "citations", NOW),
        ("vetted_preprint", "peer_reviewed", "venue", day90),
    ]


def test_signal_outage_never_demotes(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = {
        "2510.1": Signals(venue="ICLR"),
        "2510.2": Signals(citation_count=6),
    }
    run(db_session, src)
    src.s2.lookup.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 503 after 3 attempts"
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert res.status == "partial"
    assert (row(db_session, "2510.1").tier, row(db_session, "2510.1").venue) == (
        "peer_reviewed",
        "ICLR",
    )
    assert (row(db_session, "2510.2").tier, row(db_session, "2510.2").citation_count) == (
        "vetted_preprint",
        6,
    )
    assert len(history(db_session)) == 2  # the two first admissions, nothing else


def test_recheck_respects_horizons_and_skips_peer_reviewed(db_session: Session) -> None:
    repo = Repository(db_session)

    def aged(eid: str, days: int) -> SourceItem:
        return dataclasses.replace(
            paper(eid, "Agent harness"), published_at=NOW - timedelta(days=days)
        )

    repo.record_classified(aged("2510.366", 366), "candidate", "agentic-engineering", 1.0)
    repo.record_classified(aged("2510.31", 31), "candidate", "agentic-engineering", 1.0)
    repo.upsert_item(
        aged("2510.10", 10),
        Tier.PEER_REVIEWED,
        "agentic-engineering",
        1.0,
        Signals(venue="ICLR"),
        FakeEmbedder().embed(["x"])[0],
    )
    db_session.commit()
    src = fake_sources([])
    # Featured on its third day: that listing date is still inside the 30-day re-read.
    src.hf.papers.return_value = {"2510.31": HfPaper(upvotes=12)}
    run(db_session, src)
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.31"]
    assert row(db_session, "2510.31").tier == "vetted_preprint"
    assert row(db_session, "2510.366").status == "candidate"


def test_each_target_is_looked_up_once_per_run(db_session: Session) -> None:
    # A new candidate is also a re-check target; it must reach Semantic Scholar once.
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src)
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.1"]


def test_s2_refused_chunk_stays_due(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keyless: a refused chunk or a capped run delays a paper's evidence, never loses it.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}  # no evidence: both stay candidates, checked daily
    run(db_session, src)
    day1 = NOW + timedelta(days=1)
    src.arxiv.papers = []
    src.s2.refused = {"2510.2"}
    src.s2.degraded, src.s2.failure = True, "s2 chunk at 500: HTTP 429"
    res = run(db_session, src, now=day1)
    assert res.status == "partial" and res.counts["s2_unchecked"] == 1
    # Refused is counted apart from unchecked: the ids S2 refused, not the ones it never knew.
    assert res.counts["s2_refused"] == 1
    assert (row(db_session, "2510.1").s2_checked_at, row(db_session, "2510.2").s2_checked_at) == (
        day1,
        NOW,
    )
    # Half a day later only the refused paper is due again.
    src.s2.refused, src.s2.degraded, src.s2.failure = set(), False, None
    run(db_session, src, now=day1 + timedelta(hours=12))
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.2"]
    # A capped run takes the most overdue paper; the other is carried to the next run.
    monkeypatch.setattr("paperwatch.ingest.pipeline.S2_MAX_PAPERS_PER_RUN", 1)
    day2 = NOW + timedelta(days=2, hours=12)
    run(db_session, src, now=day2)
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.1"]
    run(db_session, src, now=day2 + timedelta(hours=6))
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.2"]


def test_interrupted_feed_admission_is_retried(db_session: Session) -> None:
    clf, emb = CountingClassifier(), FlakyEmbedder(failures=1)
    src = fake_sources([], feed_items=[blog("p1")])
    first = run(db_session, src, classifier=clf, embedder=emb)
    assert first.status == "partial" and row(db_session, "p1").status == "candidate"
    src.feeds.fetch_one.return_value = []
    # Any age: a relevant feed item needs no evidence, so it is retried until admitted.
    second = run(db_session, src, now=NOW + timedelta(days=400), classifier=clf, embedder=emb)
    assert second.counts["admitted_late"] == 1
    assert row(db_session, "p1").tier == "lab_blog" and clf.calls == 1


def test_new_arxiv_version_replaces_text_without_reclassifying(db_session: Session) -> None:
    clf, emb = CountingClassifier(), FlakyEmbedder(failures=0)
    v1 = paper("2510.1", "Agent harness")
    v2 = dataclasses.replace(
        paper("2510.1", "Agent harness", version=2),
        text="Agent harness. We add a memory module for tool use.",
    )
    day1 = datetime(2026, 10, 5, 6, tzinfo=UTC)
    day3 = datetime(2026, 10, 7, 6, tzinfo=UTC)
    src = fake_sources([])
    src.hf.papers.return_value = {
        "2510.1": HfPaper(upvotes=50)
    }  # admitted, so the v2 text is re-embedded
    # v1 is announced on day 1 and v2 only on day 3, not served on every day.
    src.arxiv.harvest.side_effect = _pages_for({day1.date(): [[v1]], day3.date(): [[v2]]})
    run(db_session, src, now=day1, since_override=day1, classifier=clf, embedder=emb)
    res = run(db_session, src, now=day3, since_override=day3, classifier=clf, embedder=emb)
    r = row(db_session, "2510.1")
    assert (r.version, r.text) == (2, v2.text) and res.counts["text_updated"] == 1
    assert clf.calls == 1 and emb.calls == 2  # first admission, then the v2 re-embedding


def test_failed_reembedding_keeps_the_old_version(db_session: Session) -> None:
    emb = FlakyEmbedder(failures=0)
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src, embedder=emb)
    before = row(db_session, "2510.1")
    old_text, old_vec = before.text, list(before.embedding)
    v2 = dataclasses.replace(paper("2510.1", "Agent harness", version=2), text="Rewritten.")
    src.arxiv.papers = [v2]
    emb.failures = 1
    res = run(db_session, src, now=NOW + timedelta(days=1), embedder=emb)
    assert res.status == "partial" and res.counts["failed"] == 1
    db_session.expire_all()
    r = row(db_session, "2510.1")
    assert (r.version, r.text) == (1, old_text) and list(r.embedding) == pytest.approx(old_vec)
    # The next run (still inside the overlap) retries and succeeds.
    run(db_session, src, now=NOW + timedelta(days=2), embedder=emb)
    db_session.expire_all()
    assert row(db_session, "2510.1").version == 2


def test_one_missed_feed_day_is_a_warning(db_session: Session) -> None:
    # A 502 once is not a bad morning (2026-10-23: Microsoft Research answered 502 at 06:00).
    # The dead letter and the pinned cursor stay, so the next run retries the same window and
    # nothing is lost; the day it missed is a warning, not a degraded run.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 502")
    res = run(db_session, src)
    assert res.status == "succeeded" and res.counts["stored"] == 1
    warning = f"feed missed once, retried next run: {FEED.url}"
    assert res.health == Health(True, (), (warning,))
    assert db_session.get_one(Run, res.run_id).health == {
        "healthy": True,
        "problems": [],
        "warnings": [warning],
    }
    # The parked-letter row for the fetched paper is resolved; the feed failure is the open one.
    dl = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == FEED.url)).one()
    assert dl.window_end == NOW and dl.resolved_at is None
    # The failed feed's cursor is pinned to the window it missed, not listed as stuck.
    assert Repository(db_session).cursor(f"feed:{FEED.url}") == NOW - timedelta(days=7)


def test_a_second_missed_feed_day_is_a_failure(db_session: Session) -> None:
    # The same feed failing on the next run is a continuing outage, not a blip: now the run is
    # degraded, the cursor did not advance, and the morning check should say so.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 502")
    assert run(db_session, src).status == "succeeded"
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert res.status == "partial" and not res.health.healthy
    health = db_session.get_one(Run, res.run_id).health
    assert health["warnings"] == []
    assert f"cursor not advanced: feed:{FEED.url}" in health["problems"]
    # One letter per outage, however many runs it lasts.
    letters = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == FEED.url)).all()
    assert len(letters) == 1


def test_a_feed_that_recovers_resolves_its_warning(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 502")
    assert run(db_session, src).status == "succeeded"

    def good_feed(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        src.feeds.newest[feed.url] = until  # answered, and its newest post is recent
        return []

    src.feeds.fetch_one.side_effect = good_feed
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert res.status == "succeeded" and res.health == Health(True, ())
    dl = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == FEED.url)).one()
    assert dl.resolved_at == NOW + timedelta(days=1)
    assert Repository(db_session).cursor(f"feed:{FEED.url}") == NOW + timedelta(days=1)


def test_a_feed_that_misses_alternate_days_is_warned_each_time(db_session: Session) -> None:
    # Failing, answering, then failing again is two separate one-day misses, not one continuing
    # outage: each miss is a warning with its own letter, and the recovery closes the first.
    src = fake_sources([paper("2510.1", "Agent harness")])

    def good_feed(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        src.feeds.newest[feed.url] = until
        return []

    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 502")
    first = run(db_session, src)
    src.feeds.fetch_one.side_effect = good_feed
    second = run(db_session, src, now=NOW + timedelta(days=1))
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 502")
    third = run(db_session, src, now=NOW + timedelta(days=2))
    warning = f"feed missed once, retried next run: {FEED.url}"
    assert first.health.warnings == (warning,) and second.health.warnings == ()
    assert third.status == "succeeded" and third.health == Health(True, (), (warning,))
    letters = db_session.scalars(
        select(DeadLetter).where(DeadLetter.ref == FEED.url).order_by(DeadLetter.id)
    ).all()
    assert len(letters) == 2
    assert letters[0].resolved_at == NOW + timedelta(days=1)
    assert letters[1].resolved_at is None and letters[1].window_end == NOW + timedelta(days=2)


def test_two_feeds_fail_on_different_days_without_hiding_each_other(
    db_session: Session,
) -> None:
    # One feed answering while another misses is still a first miss for the other. Each run's
    # warning and stuck-cursor lists must name the right feed, so a recovering feed never covers
    # for a single miss of its sibling.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feed_list = [FEED, OTHER_FEED]
    failing = {FEED.url: {0, 2}, OTHER_FEED.url: {1, 2}}
    day = {"index": 0}

    def fetch_one(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        if day["index"] in failing[feed.url]:
            raise RuntimeError(f"HTTP 502 for {feed.url}")
        src.feeds.newest[feed.url] = until  # answered, and its newest post is recent
        return []

    src.feeds.fetch_one.side_effect = fetch_one

    def warning(url: str) -> str:
        return f"feed missed once, retried next run: {url}"

    # Day 0: A misses; B answers.
    day["index"] = 0
    first = run(db_session, src, now=NOW)
    assert first.health == Health(True, (), (warning(FEED.url),))
    # Day 1: A answers, which closes A's letter; B misses, and it is B's first miss.
    day["index"] = 1
    second = run(db_session, src, now=NOW + timedelta(days=1))
    assert second.health == Health(True, (), (warning(OTHER_FEED.url),))
    a_letter = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == FEED.url)).one()
    assert a_letter.resolved_at == NOW + timedelta(days=1)
    # Day 2: A misses again, a fresh one-day warning; B is on its second miss in a row, a failure.
    day["index"] = 2
    third = run(db_session, src, now=NOW + timedelta(days=2))
    assert third.status == "partial"
    assert third.health == Health(
        False,
        ("status=partial", f"cursor not advanced: feed:{OTHER_FEED.url}"),
        (warning(FEED.url),),
    )


def test_a_feed_failure_is_logged_with_whether_the_outage_is_new(db_session: Session) -> None:
    # The health verdict is for the morning check; the journal line is for log-based alerts, and it
    # must say whether this miss started an outage or continued one.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 502")
    with capture_logs() as logs:
        run(db_session, src)
    started = [event for event in logs if event["event"] == "feed_fetch_failed"]
    assert [(event["feed"], event["outage"], event["error"]) for event in started] == [
        (FEED.url, "started", "RuntimeError('HTTP 502')")
    ]
    with capture_logs() as logs:
        run(db_session, src, now=NOW + timedelta(days=1))
    continuing = [event for event in logs if event["event"] == "feed_fetch_failed"]
    assert [(event["feed"], event["outage"]) for event in continuing] == [(FEED.url, "continuing")]


def test_failed_source_window_is_replayed_next_run(db_session: Session) -> None:
    src = fake_sources([], feed_items=[blog("p1")])
    src.arxiv.harvest.side_effect = RuntimeError("arXiv down")
    first = run(db_session, src)
    assert first.status == "partial" and first.counts["stored"] == 1
    pinned = Repository(db_session).cursor("arxiv")
    assert pinned == datetime(2026, 10, 5, tzinfo=UTC)  # midnight of the first missed day
    src.arxiv.harvest.side_effect = _serve_papers(src)
    src.arxiv.papers = [paper("2510.5", "Coding agent")]
    src.hf.papers.return_value = {"2510.5": HfPaper(upvotes=50)}
    before = src.arxiv.harvest.call_count
    second = run(db_session, src, now=NOW + timedelta(days=7))
    days = [c.args[0] for c in src.arxiv.harvest.call_args_list[before:]]
    # The pinned cursor is the missed window's start; the harvest reaches one overlap day before it.
    assert days[0] == pinned.date() - timedelta(days=1)
    assert second.status == "succeeded" and second.counts["stored"] == 1


def test_admission_failure_is_retried_next_run(db_session: Session) -> None:
    emb = FlakyEmbedder(failures=2)
    src = fake_sources([paper("2510.6", "Agent harness")])
    first = run(db_session, src, embedder=emb)
    assert first.status == "partial" and first.counts["failed"] == 1
    assert row(db_session, "2510.6").status == "candidate"
    # The failed embed rolls the S2 check time back with the item, so the paper stays due.
    assert row(db_session, "2510.6").s2_checked_at is None
    src.arxiv.papers = []
    # Even one hour later (inside the daily cadence) the paper is looked up again.
    run(db_session, src, now=NOW + timedelta(hours=1), embedder=emb)
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.6"]
    second = run(db_session, src, now=NOW + timedelta(days=1), embedder=emb)
    assert second.status == "succeeded" and second.counts["admitted_late"] == 1


def test_item_that_always_fails_admission_gives_up(db_session: Session) -> None:
    emb = FlakyEmbedder(failures=10**6)  # e.g. a text the embedding model always refuses
    src = fake_sources([paper("2510.6", "Agent harness")])
    statuses = [
        run(db_session, src, now=NOW + timedelta(days=d), embedder=emb).status
        for d in range(MAX_ITEM_ATTEMPTS + 1)
    ]
    assert statuses == ["partial"] * MAX_ITEM_ATTEMPTS + ["succeeded"]
    assert emb.calls == MAX_ITEM_ATTEMPTS  # the run after giving up no longer tries
    r = row(db_session, "2510.6")
    assert r.status == "candidate" and r.admit_attempts == MAX_ITEM_ATTEMPTS
    assert "503" in (r.last_error or "")
    # The parked letter carries a payload; the give-up letter has none.
    dl = db_session.scalars(
        select(DeadLetter).where(DeadLetter.ref == "2510.6", DeadLetter.payload.is_(None))
    ).one()
    assert dl.resolved_at is None and "gave up" in dl.error


def test_classification_failure_is_retried_then_given_up(db_session: Session) -> None:
    class Broken(FakeClassifier):
        calls = 0

        def classify(self, item: SourceItem) -> Classification:
            Broken.calls += 1
            raise RuntimeError("model 500")

    src = fake_sources([paper("2510.7", "Agent harness")])
    clf = Broken(PROFILES)
    results = [
        run(db_session, src, now=NOW + timedelta(days=d), classifier=clf)
        for d in range(MAX_ITEM_ATTEMPTS + 1)
    ]
    # Fetched again by the overlap and retried from the dead letter: one attempt per run.
    assert Broken.calls == MAX_ITEM_ATTEMPTS
    assert results[MAX_ITEM_ATTEMPTS - 1].counts["given_up"] == 1
    assert results[-1].status == "succeeded"
    dl = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "2510.7")).one()
    assert dl.attempts == MAX_ITEM_ATTEMPTS and dl.resolved_at is None


def test_budget_exhaustion_stops_run_cleanly(db_session: Session) -> None:
    class Costly(FakeEmbedder):
        def __init__(self, ledger: Ledger) -> None:
            super().__init__()
            self.ledger = ledger

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.ledger.charge("fake", 1000)
            return super().embed(texts)

    ledger = Ledger(1500)
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    res = run(db_session, src, embedder=Costly(ledger), ledger=ledger)
    assert res.status == "budget_exhausted" and res.counts["stored"] == 1
    # The second paper is already classified: it waits as a candidate, not a dead letter.
    statuses = sorted(db_session.scalars(select(Item.status)))
    assert statuses == ["admitted", "candidate"]
    assert Repository(db_session).open_item_dead_letters() == []
    later = run(db_session, src, now=NOW + timedelta(days=1))
    assert later.counts["admitted_late"] == 1


def test_budget_stop_during_classification_skips_recheck(db_session: Session) -> None:
    class CostlyClassifier(FakeClassifier):
        def __init__(self, ledger: Ledger, charge: int) -> None:
            super().__init__(PROFILES)
            self.ledger, self.charge = ledger, charge

        def classify(self, item: SourceItem) -> Classification:
            self.ledger.charge("classify", self.charge)
            return super().classify(item)

    class CheapEmbedder(FakeEmbedder):
        calls = 0

        def __init__(self, ledger: Ledger, charge: int) -> None:
            super().__init__()
            self.ledger, self.charge = ledger, charge

        def embed(self, texts: list[str]) -> list[list[float]]:
            CheapEmbedder.calls += 1
            self.ledger.charge("embed", self.charge)
            return super().embed(texts)

    ledger = Ledger(1500)
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    res = run(
        db_session,
        src,
        classifier=CostlyClassifier(ledger, 1000),
        embedder=CheapEmbedder(ledger, 100),
        ledger=ledger,
    )
    assert res.status == "budget_exhausted"
    assert CheapEmbedder.calls == 0  # phase 3 must not run after a phase-2 budget stop
    assert ledger.tokens_used == 1000  # the charge that was refused added nothing
    assert res.counts["new"] == 1  # only the item actually written counts as new
    # Phase 3 never ran, so no listing was read and the snapshot count stays at its initial 0.
    assert res.counts["snapshots"] == 0
    assert sorted(db_session.scalars(select(Item.status))) == ["candidate"]
    letters = Repository(db_session).open_item_dead_letters()
    assert len(letters) == 1 and letters[0][1] is not None
    assert db_session.get(DeadLetter, letters[0][0]).attempts == 0


def test_budget_stop_does_not_resurrect_given_up_items(db_session: Session) -> None:
    class Boom(FakeClassifier):
        def classify(self, item: SourceItem) -> Classification:
            raise RuntimeError("model 500")

    # Three classification failures give the poison paper up: one open letter at the cap.
    poison = paper("2510.poison", "Agent harness")
    src = fake_sources([poison])
    for day in range(MAX_ITEM_ATTEMPTS):
        run(db_session, src, now=NOW + timedelta(days=day), classifier=Boom(PROFILES))
    [letter] = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "2510.poison")).all()
    assert letter.attempts == MAX_ITEM_ATTEMPTS

    class Stop(FakeClassifier):
        def __init__(self, ledger: Ledger) -> None:
            super().__init__(PROFILES)
            self.ledger = ledger

        def classify(self, item: SourceItem) -> Classification:
            self.ledger.charge("classify", 10)
            return super().classify(item)

    ledger = Ledger(1)
    trigger = paper("2510.trigger", "Coding agent")
    # The trigger comes first, so a budget loop over the rest would see the poison paper.
    src.arxiv.papers = [trigger, poison]
    res = run(
        db_session,
        src,
        now=NOW + timedelta(days=3),
        classifier=Stop(ledger),
        ledger=ledger,
    )
    assert res.status == "budget_exhausted"
    letters = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "2510.poison")).all()
    assert len(letters) == 1 and letters[0].attempts == MAX_ITEM_ATTEMPTS


def test_retried_counts_only_real_retries(db_session: Session) -> None:
    class Boom(FakeClassifier):
        def classify(self, item: SourceItem) -> Classification:
            raise RuntimeError("model 500")

    src = fake_sources([paper("2510.1", "Agent harness")])
    first = run(db_session, src)
    assert first.counts["new"] == 1 and first.counts["retried"] == 0
    # A failure before the row exists leaves an open letter; succeeding later is a real retry.
    poison = fake_sources([paper("2510.2", "Coding agent")])
    run(db_session, poison, classifier=Boom(PROFILES))
    second = run(db_session, poison, now=NOW + timedelta(days=1))
    assert second.counts["retried"] == 1


def test_crash_after_fetch_commit_resumes_and_finishes(db_session: Session) -> None:
    class Crash(FakeClassifier):
        def classify(self, item: SourceItem) -> Classification:
            raise SystemExit("process killed")

    # Submitted 10 days before NOW: inside the first 14-day window, outside tomorrow's overlap.
    old = dataclasses.replace(
        paper("2510.old", "Agent harness"), published_at=NOW - timedelta(days=10)
    )
    src = fake_sources([old])
    with pytest.raises(SystemExit):
        run(db_session, src, classifier=Crash(PROFILES))
    crashed = db_session.scalars(select(Run).where(Run.status == "failed")).one()
    assert crashed.finished_at is not None  # the row is closed, not left 'running'
    # The next run's arXiv window no longer covers the paper; the parked letter resumes it.
    src.arxiv.papers = []
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert res.status == "succeeded"
    assert row(db_session, "2510.old").status == "admitted"


def test_corrupt_dead_letter_payload_is_counted_not_fatal(db_session: Session) -> None:
    class Boom(FakeClassifier):
        def classify(self, item: SourceItem) -> Classification:
            raise RuntimeError("model 500")

    repo = Repository(db_session)
    rid = repo.start_run(NOW, NOW + timedelta(days=1), at=NOW)
    # A payload missing a required key, as a bug or an older writer would leave it.
    repo.add_dead_letter(rid, "arxiv", "2510.bad", "corrupt", payload={"source": "arxiv"})
    db_session.commit()
    src = fake_sources([paper("2510.1", "Agent harness")])
    # A regular item fails in the same run, so its rollback must not undo the corrupt count.
    first = run(db_session, src, now=NOW + timedelta(days=1), classifier=Boom(PROFILES))
    assert first.status == "partial" and first.counts["failed"] == 2
    assert first.counts["given_up"] == 0
    corrupt = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "2510.bad")).one()
    assert corrupt.attempts == 2  # the increment survived the failing item's rollback
    # At the attempt cap the corrupt letter is given up instead of failing every run.
    last = run(db_session, src, now=NOW + timedelta(days=2), classifier=Boom(PROFILES))
    assert last.counts["given_up"] == 1
    db_session.refresh(corrupt)
    assert corrupt.attempts == MAX_ITEM_ATTEMPTS


def test_crash_during_fetch_finishes_the_run(db_session: Session) -> None:
    src = fake_sources([])
    src.arxiv.harvest.side_effect = SystemExit("killed during fetch")
    with pytest.raises(SystemExit):
        run(db_session, src)
    crashed = db_session.scalars(select(Run).where(Run.status == "failed")).one()
    assert crashed.finished_at is not None


def test_hard_kill_left_running_run_is_closed_next_run(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Crash(FakeClassifier):
        def classify(self, item: SourceItem) -> Classification:
            raise SystemExit("killed")

    old = dataclasses.replace(
        paper("2510.old", "Agent harness"), published_at=NOW - timedelta(days=10)
    )
    src = fake_sources([old])
    # A hard kill cannot run the handler: finish_run never records the crash.
    monkeypatch.setattr(Repository, "finish_run", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        run(db_session, src, classifier=Crash(PROFILES))
    monkeypatch.undo()
    stale = db_session.scalars(select(Run).where(Run.status == "running")).one()
    assert stale.finished_at is None
    src.arxiv.papers = []
    run(db_session, src, now=NOW + timedelta(days=1))
    db_session.refresh(stale)
    assert stale.status == "failed" and stale.finished_at is not None


def test_first_run_harvests_the_lookback_window(db_session: Session) -> None:
    days: list[date] = []
    src = fake_sources([])

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        days.append(day)
        return iter([])

    src.arxiv.harvest.side_effect = harvest
    run(db_session, src)
    # Cursorless: (now - 7 days).date() through now.date(), oldest first.
    assert days == [date(2026, 10, 5) + timedelta(days=offset) for offset in range(8)]
    # Every past day completed, so the cursor is today's midnight; today never completes.
    assert Repository(db_session).cursor("arxiv") == datetime(2026, 10, 12, tzinfo=UTC)


def test_second_run_starts_at_cursor_minus_the_overlap(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src)
    days: list[date] = []
    serve = _serve_papers(src)

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        days.append(day)
        return serve(day)

    src.arxiv.harvest.side_effect = harvest
    res = run(db_session, src, now=NOW + timedelta(days=1))
    # Cursor midnight(2026-10-12) minus the one-day overlap.
    assert days[0] == date(2026, 10, 11)
    # The re-read overlap stores nothing twice.
    assert res.counts["new"] == 0 and res.counts["stored"] == 0
    assert len(db_session.scalars(select(Item)).all()) == 1


def test_failure_on_a_later_page_keeps_completed_days_and_replays(db_session: Session) -> None:
    day0, day1, day2 = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)
    first, second = paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")
    page1, page2 = paper("2510.3", "Agent planning"), paper("2510.4", "Agent memory")

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        if day == day0:
            yield [first]
        elif day == day1:
            yield [second]
        elif day == day2:
            yield [page1]
            raise RuntimeError("page 2 of the day timed out")

    src = fake_sources([])
    src.arxiv.harvest.side_effect = harvest
    first_run = run(db_session, src)
    assert first_run.status == "partial"
    # The two days before the failure completed; the third day's page 1 is parked and stored.
    repo = Repository(db_session)
    assert repo.cursor("arxiv") == datetime(2026, 10, 7, tzinfo=UTC)
    assert set(db_session.scalars(select(Item.external_id))) == {"2510.1", "2510.2", "2510.3"}
    # A run that finished some days and then stopped says so, not "cursor not advanced".
    assert "arxiv stopped early" in db_session.get_one(Run, first_run.run_id).health["problems"]
    letters = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "arxiv", DeadLetter.ref == "window")
    ).all()
    assert len(letters) == 1 and letters[0].resolved_at is None

    # The next run re-harvests from the third day minus the overlap and stores the rest once.
    def replay(day: date) -> Iterator[list[SourceItem]]:
        if day == day2:
            yield [page2]

    src.arxiv.harvest.side_effect = replay
    src.hf.papers.return_value = {"2510.4": HfPaper(upvotes=50)}
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert set(db_session.scalars(select(Item.external_id))) == {
        "2510.1",
        "2510.2",
        "2510.3",
        "2510.4",
    }
    assert res.counts["stored"] == 1


def test_repeated_failures_without_a_cursor_pin_the_first_day(db_session: Session) -> None:
    src = fake_sources([])
    src.arxiv.harvest.side_effect = RuntimeError("arXiv down")
    assert run(db_session, src).status == "partial"
    pinned = Repository(db_session).cursor("arxiv")
    assert pinned == datetime(2026, 10, 5, tzinfo=UTC)  # midnight of the first missed day
    assert run(db_session, src, now=NOW + timedelta(days=1)).status == "partial"
    # The cursor does not slide forward with the clock, so the missed window stays covered.
    assert Repository(db_session).cursor("arxiv") == pinned


def test_failure_before_any_day_completes_reports_a_stuck_cursor(db_session: Session) -> None:
    src = fake_sources([])
    src.arxiv.harvest.side_effect = RuntimeError("arXiv down")
    res = run(db_session, src)
    assert "cursor not advanced: arxiv" in db_session.get_one(Run, res.run_id).health["problems"]


def test_page_is_parked_before_the_next_request(db_session: Session) -> None:
    seen: list[list[str]] = []

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        if day == date(2026, 10, 12):
            yield [paper("2510.77", "Agent harness")]
            # The pipeline resumes this generator only to request the next page, so the first
            # page must already be parked and committed here.
            seen.append(
                [str(p["external_id"]) for _, p in Repository(db_session).open_item_dead_letters()]
            )
            yield [paper("2510.78", "Agent memory")]

    src = fake_sources([])
    src.arxiv.harvest.side_effect = harvest
    run(db_session, src)
    assert seen == [["2510.77"]]


def test_crash_between_pages_loses_nothing(db_session: Session) -> None:
    def harvest(day: date) -> Iterator[list[SourceItem]]:
        if day == date(2026, 10, 12):
            yield [paper("2510.77", "Agent harness")]
            raise SystemExit("killed between pages")

    src = fake_sources([])
    src.arxiv.harvest.side_effect = harvest
    with pytest.raises(SystemExit):
        run(db_session, src)
    # The page parked before the crash is picked up by the next run and admitted once.
    src.arxiv.harvest.side_effect = _pages_for({})
    src.hf.papers.return_value = {"2510.77": HfPaper(upvotes=50)}
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert row(db_session, "2510.77").status == "admitted" and res.status == "succeeded"


def test_parked_letter_is_refreshed_to_a_higher_version(db_session: Session) -> None:
    # A crash during a multi-day catch-up leaves v1 parked; a later day's v2 must replace the
    # parked payload, or resuming phase 2 would store the older text. A lower version must not.
    repo = Repository(db_session)
    run_id = repo.start_run(NOW, NOW, at=NOW)
    v1 = paper("2510.9", "Agent harness")
    v2 = dataclasses.replace(paper("2510.9", "Agent harness", version=2), text="Rewritten.")
    parked: set[tuple[str, str]] = set()
    _park_items(repo, run_id, [v1], parked)
    _park_items(repo, run_id, [v2], parked)
    _park_items(repo, run_id, [v1], parked)  # a lower version must not overwrite the parked v2
    db_session.commit()
    letter = db_session.scalars(
        select(DeadLetter).where(DeadLetter.ref == "2510.9", DeadLetter.resolved_at.is_(None))
    ).one()
    assert payload_to_item(letter.payload).version == 2


def test_given_up_letter_is_not_refreshed(db_session: Session) -> None:
    # A given-up letter is a human's to resolve; parking a newer version must not rewrite it.
    repo = Repository(db_session)
    run_id = repo.start_run(NOW, NOW, at=NOW)
    v1 = paper("2510.9", "Agent harness")
    repo.add_dead_letter(
        run_id,
        "arxiv",
        "2510.9",
        "gave up after repeated failures",
        payload=item_to_payload(v1),
        attempts=MAX_ITEM_ATTEMPTS,
    )
    db_session.commit()
    v2 = dataclasses.replace(paper("2510.9", "Agent harness", version=2), text="Rewritten.")
    _park_items(repo, run_id, [v2], set())
    db_session.commit()
    letter = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "2510.9")).one()
    assert payload_to_item(letter.payload).version == 1


def test_steady_state_cursor_day_failure_reports_a_stuck_cursor(db_session: Session) -> None:
    # Run 1 leaves the cursor at midnight(2026-10-12). Run 2's overlap day rewrites that same
    # cursor, then the cursor day fails: the cursor did not move, so it is not an early stop.
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src)

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        if day == date(2026, 10, 12):
            yield [paper("2510.2", "Coding agent")]
            raise RuntimeError("the cursor day failed")
        return

    src.arxiv.harvest.side_effect = harvest
    res = run(db_session, src, now=NOW + timedelta(days=1))
    problems = db_session.get_one(Run, res.run_id).health["problems"]
    assert "cursor not advanced: arxiv" in problems
    assert "arxiv stopped early" not in problems


def test_steady_state_failure_on_today_is_an_early_stop(db_session: Session) -> None:
    # The cursor day completed and moved the cursor to today's midnight before today failed.
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src)

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        if day == date(2026, 10, 13):
            raise RuntimeError("today failed")
        return iter([])

    src.arxiv.harvest.side_effect = harvest
    res = run(db_session, src, now=NOW + timedelta(days=1))
    problems = db_session.get_one(Run, res.run_id).health["problems"]
    assert "arxiv stopped early" in problems
    assert "cursor not advanced: arxiv" not in problems


def test_since_override_harvests_from_that_date(db_session: Session) -> None:
    days: list[date] = []
    src = fake_sources([])

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        days.append(day)
        return iter([])

    src.arxiv.harvest.side_effect = harvest
    res = run(db_session, src, since_override=NOW - timedelta(days=3))
    assert days[0] == date(2026, 10, 9)
    assert days[-1] == date(2026, 10, 12)
    assert Repository(db_session).cursor("arxiv") == datetime(2026, 10, 12, tzinfo=UTC)
    assert res.status == "succeeded"


def test_weekend_only_window_with_no_items_is_not_flagged(db_session: Session) -> None:
    src = fake_sources([])
    saturday = datetime(2026, 10, 10, 6, tzinfo=UTC)
    res = run(db_session, src, now=saturday + timedelta(days=1), since_override=saturday)
    problems = db_session.get_one(Run, res.run_id).health["problems"]
    assert "arxiv fetched 0 items" not in problems


def test_weekday_window_with_no_items_is_flagged(db_session: Session) -> None:
    src = fake_sources([])
    wednesday = datetime(2026, 10, 7, 6, tzinfo=UTC)
    res = run(db_session, src, now=wednesday, since_override=wednesday)
    problems = db_session.get_one(Run, res.run_id).health["problems"]
    assert "arxiv fetched 0 items" in problems


def test_malformed_arxiv_entry_is_dead_lettered_once_and_cursor_advances(
    db_session: Session,
) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.arxiv.rejected = [
        RejectedEntry(
            source="arxiv", ref="http://arxiv.org/abs/2510.9v1", error="missing published date"
        )
    ]
    first = run(db_session, src)
    assert first.status == "partial" and first.counts["malformed"] == 1
    assert Repository(db_session).cursor("arxiv") == datetime(2026, 10, 12, tzinfo=UTC)
    # The daily overlap re-reads the same bad entry tomorrow: it is not recorded twice.
    second = run(db_session, src, now=NOW + timedelta(days=1))
    assert second.status == "succeeded" and second.counts["malformed"] == 0
    # Exactly one open malformed letter: the parked letter for the good paper is resolved.
    malformed = db_session.scalars(select(DeadLetter).where(DeadLetter.payload.is_(None))).all()
    assert len(malformed) == 1 and malformed[0].ref == "http://arxiv.org/abs/2510.9v1"


def test_malformed_feed_entry_is_dead_lettered_once_and_cursor_advances(
    db_session: Session,
) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")], feed_items=[blog("p1")])
    bad = RejectedEntry(source="feed", ref="https://lab.example/p2", error="entry has no title")

    def fetch_one(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        src.feeds.rejected.append(bad)  # the real client appends while reading the feed
        return [blog("p1")]

    src.feeds.fetch_one.side_effect = fetch_one
    first = run(db_session, src)
    assert first.status == "partial" and first.counts["malformed"] == 1
    assert Repository(db_session).cursor(f"feed:{FEED.url}") == NOW
    second = run(db_session, src, now=NOW + timedelta(days=1))
    assert second.counts["malformed"] == 0
    # The malformed entry is the one open feed letter; parked letters for stored posts resolve.
    letters = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "feed", DeadLetter.payload.is_(None))
    ).all()
    assert len(letters) == 1 and letters[0].ref == bad.ref


def test_feed_post_published_before_the_cursor_is_still_stored(db_session: Session) -> None:
    # Dated 05:50, visible in the feed only after the 06:00 run: the next run must catch it.
    late = dataclasses.replace(blog("late"), published_at=NOW - timedelta(minutes=10))
    src = fake_sources([paper("2510.1", "Agent harness")], feed_items=[])
    run(db_session, src)
    src.feeds.fetch_one.return_value = [late]
    tomorrow = NOW + timedelta(days=1)
    run(db_session, src, now=tomorrow)
    _, since, until = src.feeds.fetch_one.call_args.args
    assert since == NOW - FEED_OVERLAP and until == tomorrow
    assert row(db_session, "late").status == "admitted"


def test_moved_feed_does_not_store_its_posts_twice(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")], feed_items=[blog("p1")])
    run(db_session, src)
    moved = dataclasses.replace(blog("p1"), external_id="https://lab.example/new-feed.xml p1")
    src.feeds.fetch_one.return_value = [moved]
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert res.counts["duplicate_url"] == 1 and res.counts["new"] == 0
    assert len(db_session.scalars(select(Item).where(Item.source == "feed")).all()) == 1


def test_moved_feed_posts_are_not_parked_every_run(db_session: Session) -> None:
    src = fake_sources([], feed_items=[blog("p1")])
    run(db_session, src)
    moved = dataclasses.replace(blog("p1"), external_id="https://lab.example/new-feed.xml p1")
    src.feeds.fetch_one.return_value = [moved]
    run(db_session, src, now=NOW + timedelta(days=1))
    # The moved post's link is already stored, so it gets no dead-letter row at all.
    letters = db_session.scalars(
        select(DeadLetter).where(DeadLetter.ref == moved.external_id)
    ).all()
    assert letters == []


def test_stale_feed_is_flagged_in_health(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])

    def fetch_one(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        src.feeds.newest[feed.url] = NOW - timedelta(days=120)  # answers, publishes nothing new
        return []

    src.feeds.fetch_one.side_effect = fetch_one
    res = run(db_session, src)
    problems = db_session.get_one(Run, res.run_id).health["problems"]
    assert f"feed stale or empty: {FEED.url}" in problems


def test_run_records_health_verdict(db_session: Session) -> None:
    healthy = run(db_session, fake_sources([paper("2510.1", "Agent harness")]))
    row_ = db_session.get_one(Run, healthy.run_id)
    assert row_.health == {"healthy": True, "problems": [], "warnings": []}
    # The verdict also travels on the result: `paperwatch ingest` turns it into exit code 3, so a
    # row alone would never reach the process that systemd and the morning check look at.
    assert healthy.health == Health(True, ())
    # An arXiv fetch that returns nothing over a 7-day-plus window is a broken run.
    empty = run(db_session, fake_sources([]), now=NOW + timedelta(days=1))
    problems = db_session.get_one(Run, empty.run_id).health["problems"]
    assert "arxiv fetched 0 items" in problems
    assert empty.health == Health(False, ("arxiv fetched 0 items",))


def test_signal_source_outage_is_visible_and_cursor_advances(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 3 attempts"
    res = run(db_session, src)
    assert res.status == "partial" and res.counts["signal_failures"] == 1
    assert res.counts["s2_unchecked"] == 1  # still due: the next run asks again
    assert res.counts["s2_refused"] == 1  # a refused chunk is a refusal, not an unknown paper
    dl = db_session.scalars(select(DeadLetter).where(DeadLetter.source == "s2")).one()
    assert "429" in dl.error
    # arXiv was fetched fine, so its cursor moves; the next run's re-check repeats the lookup.
    assert Repository(db_session).cursor("arxiv") == datetime(2026, 10, 12, tzinfo=UTC)
    health = db_session.get_one(Run, res.run_id).health
    assert "1 signal sources degraded" in health["problems"]


def test_daily_runs_over_a_week_lose_nothing(db_session: Session) -> None:
    # Replays seven consecutive daily runs against a fake arXiv that, like OAI-PMH, returns a
    # paper on its announcement datestamp day. One paper is announced four days later; the feed
    # is down on day 3. Nothing may be lost or stored twice.
    day0 = NOW
    catalogue = [  # (id, announcement day offset from the first run)
        ("2510.11", 0),
        ("2510.12", 1),
        ("2510.13", 4),
        ("2510.14", 5),
    ]

    def announced(day: date) -> Iterator[list[SourceItem]]:
        offset = (day - day0.date()).days
        items = [
            paper(eid, "Coding agent harness")
            for eid, announced_offset in catalogue
            if announced_offset == offset
        ]
        return iter([items] if items else [])

    src = fake_sources([], feed_items=[blog("p1")])
    src.arxiv.harvest.side_effect = announced
    src.hf.papers.return_value = {eid: HfPaper(upvotes=50) for eid, _ in catalogue}
    statuses = []
    for day in range(7):
        now = day0 + timedelta(days=day)
        src.feeds.fetch_one.side_effect = RuntimeError("HTTP 503") if day == 3 else None
        statuses.append(run(db_session, src, now=now).status)
    assert statuses[3] == "succeeded" and statuses.count("succeeded") == 7
    admitted = sorted(
        db_session.scalars(
            select(Item.external_id).where(Item.source == "arxiv", Item.status == "admitted")
        )
    )
    assert admitted == [eid for eid, _ in catalogue]
    assert len(db_session.scalars(select(Item).where(Item.source == "feed")).all()) == 1


def test_second_run_is_skipped_while_the_lock_is_held(db_session: Session, engine: Engine) -> None:
    repo = Repository(db_session)
    # A run a hard kill left 'running' must be left untouched by a skipped run.
    stale = repo.start_run(NOW, NOW + timedelta(days=1), at=NOW)
    db_session.commit()
    lock_conn = db_session.get_bind().engine.connect()
    try:
        held = lock_conn.scalar(text("SELECT pg_try_advisory_lock(:k)"), {"k": INGEST_LOCK_KEY})
        assert held is True
        src = fake_sources([paper("2510.1", "Agent harness")])
        before = engine.pool.checkedout()
        with capture_logs() as logs:
            res = run(db_session, src)
        after = engine.pool.checkedout()
    finally:
        lock_conn.scalar(text("SELECT pg_advisory_unlock(:k)"), {"k": INGEST_LOCK_KEY})
        lock_conn.close()
    assert res.status == "skipped" and res.run_id is None
    assert res.counts == {} and res.tokens == 0
    # The skipped run opens its lock connection and closes it again, so the pool is unchanged.
    assert after == before
    # The skip leaves one log event, so M4's log-based alerts can see a lock that never frees.
    assert [event["event"] for event in logs] == ["ingestion_skipped"]
    # No run row was written, the stale row is untouched, and no source was called.
    assert len(db_session.scalars(select(Run)).all()) == 1
    assert db_session.get_one(Run, stale).status == "running"
    assert db_session.get_one(Run, stale).finished_at is None
    src.arxiv.harvest.assert_not_called()
    src.feeds.fetch_one.assert_not_called()


def test_failed_run_releases_the_lock(db_session: Session) -> None:
    src = fake_sources([])
    src.arxiv.harvest.side_effect = SystemExit("killed during fetch")
    # A crash is stamped with the clock too, so the row it closes shows when the run really died.
    later = NOW + timedelta(minutes=2)
    with pytest.raises(SystemExit):
        run(db_session, src, clock=lambda: later)
    # The failed row is closed with the clock, not the database's.
    failed = db_session.scalars(select(Run).where(Run.status == "failed")).one()
    assert failed.started_at == NOW and failed.finished_at == later
    # The finally released the lock, so a following run is not skipped.
    src.arxiv.harvest.side_effect = _serve_papers(src)
    res = run(db_session, src, now=NOW + timedelta(minutes=1))
    assert res.status == "succeeded"


def test_degraded_signal_letter_is_written_once_and_closed_on_recovery(
    db_session: Session,
) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    for day in range(3):  # three degraded runs leave one open letter, not three
        res = run(db_session, src, now=NOW + timedelta(days=day))
        assert res.status == "partial" and res.counts["signal_failures"] == 1
    letters = db_session.scalars(select(DeadLetter).where(DeadLetter.source == "s2")).all()
    assert len(letters) == 1 and letters[0].resolved_at is None
    # The next run reaches Semantic Scholar healthy, so the letter closes.
    src.s2.degraded, src.s2.failure = False, None
    res = run(db_session, src, now=NOW + timedelta(days=3))
    assert res.status == "succeeded" and res.counts["signal_failures"] == 0
    db_session.refresh(letters[0])
    assert letters[0].resolved_at == NOW + timedelta(days=3)


def test_arxiv_outage_letter_is_written_once_and_closed_on_recovery(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.arxiv.harvest.side_effect = RuntimeError("arXiv 503")
    assert run(db_session, src).status == "partial"
    assert run(db_session, src, now=NOW + timedelta(days=1)).status == "partial"
    window = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "arxiv", DeadLetter.ref == "window")
    ).all()
    assert len(window) == 1 and window[0].resolved_at is None
    # A good fetch closes the outage letter.
    src.arxiv.harvest.side_effect = _serve_papers(src)
    src.arxiv.papers = [paper("2510.2", "Coding agent")]
    res = run(db_session, src, now=NOW + timedelta(days=2))
    assert res.status == "succeeded"
    db_session.refresh(window[0])
    assert window[0].resolved_at == NOW + timedelta(days=2)


def test_an_arxiv_page_cap_degrades_the_run_and_names_set_and_day(db_session: Session) -> None:
    # Item 4's cap and repeat check raise out of the OAI loop; at run level that must degrade arXiv
    # (partial, "stopped early", the day pinned as the next run's start), keep the day it finished
    # and carry the message with the set and the day into the dead letter.
    src = fake_sources([paper("2510.1", "Agent harness")])
    broken_day = NOW.date() - timedelta(days=6)

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        if day == broken_day:
            raise ValueError(f"OAI-PMH page cap (200) reached for set cs on {day.isoformat()}")
        return iter([src.arxiv.papers] if day < broken_day else [])

    src.arxiv.harvest.side_effect = harvest
    res = run(db_session, src)
    assert res.status == "partial"
    assert res.health is not None
    assert "arxiv stopped early" in res.health.problems
    # The day before the failure completed, so the next run starts at the broken day, not before it.
    assert Repository(db_session).cursor("arxiv") == datetime(
        broken_day.year, broken_day.month, broken_day.day, tzinfo=UTC
    )
    assert row(db_session, "2510.1").status == "admitted"  # the completed day is not thrown away
    letter = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "arxiv", DeadLetter.ref == "window")
    ).one()
    assert f"page cap (200) reached for set cs on {broken_day.isoformat()}" in letter.error


def test_feed_outage_letter_closes_but_a_malformed_entry_stays_open(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 503")
    # The first miss is a warning; only a second run in a row with the same feed failing degrades.
    assert run(db_session, src).status == "succeeded"
    assert run(db_session, src, now=NOW + timedelta(days=1)).status == "partial"
    outage = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "feed", DeadLetter.ref == FEED.url)
    ).all()
    assert len(outage) == 1 and outage[0].resolved_at is None

    # The feed answers again but carries one malformed entry.
    bad = RejectedEntry(source="feed", ref="https://lab.example/p2", error="entry has no title")

    def good_feed(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        src.feeds.rejected.append(bad)
        src.feeds.newest[feed.url] = until
        return [blog("p1")]

    src.feeds.fetch_one.side_effect = good_feed
    res = run(db_session, src, now=NOW + timedelta(days=2))
    assert res.counts["malformed"] == 1
    db_session.refresh(outage[0])
    assert outage[0].resolved_at == NOW + timedelta(days=2)
    # The malformed-entry letter has a different ref, so the source-level resolver never closes it.
    malformed = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "feed", DeadLetter.ref == bad.ref)
    ).all()
    assert len(malformed) == 1 and malformed[0].resolved_at is None


def test_run_timestamps_come_from_the_run_clock(db_session: Session) -> None:
    future = NOW + timedelta(days=90)  # far from the wall clock: the run's now, not the database's
    finish = future + timedelta(minutes=47)
    src = fake_sources([paper("2510.1", "Agent harness")])
    res = run(db_session, src, now=future, clock=lambda: finish)
    assert res.run_id is not None
    run_row = db_session.get_one(Run, res.run_id)
    # A run starts at its `now` and ends at the wall clock, so its duration is real work.
    assert run_row.started_at == future and run_row.finished_at == finish
    # The letter phase 1 parked for the stored paper was resolved with the run's clock.
    resolved = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "2510.1")).one()
    assert resolved.resolved_at == future


def test_stale_running_row_is_closed_with_the_run_clock(db_session: Session) -> None:
    repo = Repository(db_session)
    stale_id = repo.start_run(NOW, NOW, at=NOW)  # a run a hard kill left 'running'
    db_session.commit()
    later = NOW + timedelta(days=3)
    run(db_session, fake_sources([paper("2510.1", "Agent harness")]), now=later)
    stale = db_session.get_one(Run, stale_id)
    assert stale.status == "failed" and stale.finished_at == later


def test_duplicate_url_resolution_uses_the_run_clock(db_session: Session) -> None:
    src = fake_sources([], feed_items=[blog("p1")])
    run(db_session, src)  # stores p1's link
    moved = dataclasses.replace(blog("p1"), external_id="https://lab.example/new-feed.xml p1")
    # An earlier run left an open letter for the moved id, for example a classification failure.
    repo = Repository(db_session)
    rid = repo.start_run(NOW, NOW, at=NOW)
    repo.add_dead_letter(
        rid, "feed", moved.external_id, "fetched, not yet stored", payload=item_to_payload(moved)
    )
    db_session.commit()
    src.feeds.fetch_one.return_value = [moved]
    later = NOW + timedelta(days=1)
    res = run(db_session, src, now=later)
    assert res.counts["duplicate_url"] == 1
    letter = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == moved.external_id)).one()
    assert letter.resolved_at == later


def test_stored_item_retry_resolution_uses_the_run_clock(db_session: Session) -> None:
    emb = FlakyEmbedder(failures=0)
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src, embedder=emb)  # stored and admitted
    v2 = dataclasses.replace(paper("2510.1", "Agent harness", version=2), text="Rewritten.")
    src.arxiv.papers = [v2]
    emb.failures = 1
    run(db_session, src, now=NOW + timedelta(days=1), embedder=emb)  # re-embedding fails: letter
    letter = db_session.scalars(
        select(DeadLetter).where(DeadLetter.ref == "2510.1", DeadLetter.resolved_at.is_(None))
    ).one()
    later = NOW + timedelta(days=2)
    run(db_session, src, now=later, embedder=emb)  # the retry succeeds and resolves the letter
    db_session.refresh(letter)
    assert letter.resolved_at == later


def test_lock_holder_is_not_idle_in_transaction(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    states: list[str] = []

    def peek(day: date) -> Iterator[list[SourceItem]]:
        # The advisory lock holder is this run, mid-flight: it must not sit in a transaction.
        states.extend(
            db_session.execute(
                text(
                    "SELECT a.state FROM pg_locks l JOIN pg_stat_activity a USING (pid) "
                    "WHERE l.locktype = 'advisory' AND l.granted"
                )
            ).scalars()
        )
        return iter([[paper("2510.1", "Agent harness")]])

    src.arxiv.harvest.side_effect = peek
    assert run(db_session, src).status == "succeeded"
    assert states  # the lock is taken while the run is in flight
    assert "idle in transaction" not in states


def test_no_transaction_is_open_during_a_source_fetch(db_session: Session) -> None:
    # A fetch can take a minute, and a transaction held across it pins a snapshot, holds a
    # connection 'idle in transaction' and blocks vacuum for that whole time. Every network block
    # therefore starts after a commit, like the follow-up loop's wait already did.
    # The fake sources record the flag instead of asserting it: the feed loop catches everything,
    # so an assertion raised in there is logged as a missed feed and the run carries on, which
    # would make this test pass with its fix removed.
    src = fake_sources([paper("2510.1", "Agent harness")])
    checked: list[str] = []
    in_transaction: list[tuple[str, bool]] = []

    def harvest(day: date) -> Iterator[list[SourceItem]]:
        checked.append("arxiv")
        in_transaction.append(("arxiv", db_session.in_transaction()))
        return iter([[paper("2510.1", "Agent harness")]])

    def feed(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        checked.append("feed")
        in_transaction.append(("feed", db_session.in_transaction()))
        src.feeds.newest[feed.url] = until
        # Items on the first feed, so its cursor write is pending when the second fetch starts.
        return [blog("p1")] if feed.url == FEED.url else []

    def papers(since: date, until: date) -> dict[str, HfPaper]:
        checked.append("hf")
        in_transaction.append(("hf", db_session.in_transaction()))
        return {"2510.1": HfPaper(upvotes=50)}

    def lookup(ids: list[str]) -> dict[str, Signals]:
        checked.append("s2")
        in_transaction.append(("s2", db_session.in_transaction()))
        src.s2.checked_ids, src.s2.refused_ids = set(ids), set()
        return {}

    src.feed_list = [FEED, OTHER_FEED]
    src.arxiv.harvest.side_effect = harvest
    src.feeds.fetch_one.side_effect = feed
    src.hf.papers.side_effect = papers
    src.s2.lookup.side_effect = lookup
    res = run(db_session, src)
    assert res.status == "succeeded" and res.health is not None and res.health.healthy
    # The harvest runs once per (set, day) of the window, the feeds once each, the signals once.
    assert set(checked) == {"arxiv", "feed", "hf", "s2"}
    assert checked.count("feed") == len(src.feed_list) == 2
    assert checked.count("hf") == checked.count("s2") == 1
    assert not any(open_txn for _source, open_txn in in_transaction)


def test_a_kill_while_parking_leaves_the_cursor_and_the_letters_together(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ordering inside one feed (park every item, then move the cursor) is what makes a kill
    # safe: a kill in the middle of parking must leave the cursor where it was, with whatever
    # letters that transaction wrote rolled back together. Killing during the *next* fetch cannot
    # tell the two orders apart, because both are committed by then.
    src = fake_sources([])
    src.feed_list = [FEED, OTHER_FEED]

    def fetch_one(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        src.feeds.newest[feed.url] = until
        return [blog("p1"), blog("p2")]

    src.feeds.fetch_one.side_effect = fetch_one
    real, calls = Repository.add_dead_letter, []

    def boom(self: Repository, run_id: object, source: str, ref: str, error: str, **kw: object):
        calls.append(ref)
        if source == "feed" and len(calls) == 2:
            raise SystemExit("killed while parking the second item")
        return real(self, run_id, source, ref, error, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(Repository, "add_dead_letter", boom)
    with pytest.raises(SystemExit):
        run(db_session, src)
    cursor = Repository(db_session).cursor(f"feed:{FEED.url}")
    parked = db_session.scalars(
        select(DeadLetter).where(DeadLetter.source == "feed", DeadLetter.resolved_at.is_(None))
    ).all()
    assert (cursor is not None) == (len(parked) == 2)  # neither, or both


def test_a_feed_crash_cannot_advance_a_cursor_past_its_unparked_items(
    db_session: Session,
) -> None:
    # A feed's items are parked in the same transaction as its cursor, so a kill during the next
    # feed leaves either both or neither: never a cursor ahead of items that nothing holds (S6).
    src = fake_sources([])
    src.feed_list = [FEED, OTHER_FEED]

    def fetch_one(feed: Feed, since: datetime, until: datetime) -> list[SourceItem]:
        if feed.url == OTHER_FEED.url:
            raise SystemExit("killed mid-run")
        src.feeds.newest[feed.url] = until
        return [blog("p1")]

    src.feeds.fetch_one.side_effect = fetch_one
    with pytest.raises(SystemExit):
        run(db_session, src)
    parked = db_session.scalars(
        select(DeadLetter).where(DeadLetter.ref == "p1", DeadLetter.resolved_at.is_(None))
    ).all()
    cursor = Repository(db_session).cursor(f"feed:{FEED.url}")
    # Whatever the kill caught, the two agree: a committed cursor has parked items behind it.
    assert (cursor is not None) == (len(parked) == 1)
    assert db_session.get_one(Run, db_session.scalars(select(Run.id)).one()).status == "failed"


def test_lock_is_held_for_the_whole_run(db_session: Session, engine: Engine) -> None:
    outer = fake_sources([paper("2510.1", "Agent harness")])
    seen: dict[str, object] = {}
    probing = True

    def probe_second_run(day: date) -> Iterator[list[SourceItem]]:
        # Mid-run, on its own connection, a second run must still see the lock held.
        nonlocal probing
        if not probing:
            return iter([[paper("2510.1", "Agent harness")]])
        probing = False
        second = make_session_factory(engine)()
        try:
            before = outer.feeds.fetch_one.call_count
            result = run(second, outer, now=NOW + timedelta(minutes=1))
            seen["result"] = result
            seen["feed_calls"] = outer.feeds.fetch_one.call_count - before
            # Count through db_session: the outer run's row is not yet committed to the DB.
            seen["run_rows"] = len(db_session.scalars(select(Run)).all())
        finally:
            second.close()
        return iter([[paper("2510.1", "Agent harness")]])

    outer.arxiv.harvest.side_effect = probe_second_run
    result = run(db_session, outer)
    assert result.status == "succeeded"
    inner = seen["result"]
    assert isinstance(inner, RunResult)
    assert inner.status == "skipped" and inner.run_id is None
    assert seen["feed_calls"] == 0
    assert seen["run_rows"] == 1


def kill_lock_holders(killer: Session) -> None:
    # Terminate the advisory-lock holder's backend from a second session, so the run's unlock
    # later runs against a dead connection.
    for pid in killer.execute(
        text("SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted")
    ).scalars():
        killer.execute(text("SELECT pg_terminate_backend(:p)"), {"p": pid})
    killer.commit()


def test_dead_lock_connection_still_returns_succeeded(db_session: Session, engine: Engine) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    killer = make_session_factory(engine)()
    before = engine.pool.checkedout()

    def kill_mid_run(day: date) -> Iterator[list[SourceItem]]:
        kill_lock_holders(killer)
        return iter([[paper("2510.1", "Agent harness")]])

    src.arxiv.harvest.side_effect = kill_mid_run
    try:
        assert run(db_session, src).status == "succeeded"
    finally:
        killer.close()
    # The unlock on the dead connection is suppressed and its connection closed: no leak.
    assert engine.pool.checkedout() == before


def test_dead_lock_connection_keeps_the_runs_own_error(
    db_session: Session, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    killer = make_session_factory(engine)()
    before = engine.pool.checkedout()

    def kill_mid_run(day: date) -> Iterator[list[SourceItem]]:
        kill_lock_holders(killer)
        return iter([[paper("2510.1", "Agent harness")]])

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("run boom")

    src.arxiv.harvest.side_effect = kill_mid_run
    monkeypatch.setattr(Repository, "recheck_targets", boom)
    try:
        # The unlock error on the dead connection must not replace the run's own error.
        with pytest.raises(RuntimeError, match="run boom"):
            run(db_session, src)
    finally:
        killer.close()
    assert engine.pool.checkedout() == before


def test_raising_try_lock_closes_its_connection(
    db_session: Session, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = engine.pool.checkedout()
    opened: list[Connection] = []
    original_connect = Engine.connect
    original_scalar = Connection.scalar

    def capture_connect(eng: Engine, *args: object, **kwargs: object) -> Connection:
        # Hold a reference to every connection the run opens, so a missing close cannot hide
        # behind the garbage collector returning a leaked connection to the pool.
        conn = original_connect(eng, *args, **kwargs)
        opened.append(conn)
        return conn

    def raise_on_try_lock(
        conn: Connection, statement: object, *args: object, **kwargs: object
    ) -> object:
        if "pg_try_advisory_lock" in str(statement):
            raise RuntimeError("try-lock boom")
        return original_scalar(conn, statement, *args, **kwargs)

    monkeypatch.setattr(Engine, "connect", capture_connect)
    monkeypatch.setattr(Connection, "scalar", raise_on_try_lock)
    with pytest.raises(RuntimeError, match="try-lock boom"):
        run(db_session, fake_sources([]))
    # A failing try-lock still closes the connection it opened.
    assert opened and all(conn.closed for conn in opened)
    assert engine.pool.checkedout() == before


# --- Follow-up passes (Task 24, ADR 0008) -------------------------------------------------------
# A keyless Semantic Scholar burst can refuse every attempt for hours. A scheduled run may then
# wait and ask again for exactly the refused papers, until 22:00 Europe/Rome, at most 40 times.
# These tests drive the loop with an injected sleep and clock: the real schedule is arithmetic,
# so nothing waits.

VENUE = "ICLR"
S2_VENUE = {"2510.2": Signals(venue=VENUE)}


def travelling_clock(
    start: datetime,
) -> tuple[list[float], list[datetime], Callable[[float], None]]:
    """A sleep that moves a clock, and the waits it was asked for: the loop's own time is fake."""
    sleeps: list[float] = []
    wall = [start]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        wall[0] += timedelta(seconds=seconds)

    return sleeps, wall, sleep


def test_a_refused_paper_is_admitted_in_the_first_follow_up_pass(db_session: Session) -> None:
    # The morning refusal clears before the first retry: the paper's own venue admits it, the run
    # ends healthy, and no day of evidence is lost.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded, src.s2.failure = {"2510.2"}, True, "HTTP 429 after 5 attempts"
    sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        # From the next pass on Semantic Scholar answers: the stub resets its flags on every call,
        # exactly as the real client does.
        sleep(seconds)
        src.s2.refused, src.s2.degraded, src.s2.failure = set(), False, None

    res = run(db_session, src, followup=True, sleep=recover, clock=lambda: wall[0])
    assert sleeps == [300]  # the first wait: 5 minutes
    assert res.status == "succeeded" and res.health is not None and res.health.healthy
    assert res.counts["followup_passes"] == 1
    assert res.counts["s2_unchecked"] == 0 and res.counts["s2_refused"] == 0
    assert res.health.followup == {"passes": 1, "stopped": "resolved"}
    assert row(db_session, "2510.2").tier == "peer_reviewed"
    # The check time is the run's `now`, not the moment of the pass, so the cadence is unchanged.
    assert (row(db_session, "2510.1").s2_checked_at, row(db_session, "2510.2").s2_checked_at) == (
        NOW,
        NOW,
    )
    # The refusal's outage letter is closed by the recovery, as a healthy source's always is.
    letters = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "signals")).all()
    assert [letter.resolved_at for letter in letters] == [NOW]


def test_a_pass_check_is_stamped_with_the_runs_now_and_due_the_next_day(
    db_session: Session,
) -> None:
    # The pass runs at 06:05 but stamps `s2_checked_at` with the run's 06:00 `now`, so the daily
    # cadence is unchanged and the next run finds the paper due, not skipped for a day.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = {}  # answered, but no venue or citations yet: still a candidate
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    _sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded = set(), False

    run(db_session, src, followup=True, sleep=recover, clock=lambda: wall[0])
    item = row(db_session, "2510.2")
    assert item.status == "candidate" and item.s2_checked_at == NOW
    # The next morning: the paper is due again, and its own day's stamp is the next run's start.
    src.arxiv.papers = []
    run(db_session, src, now=NOW + timedelta(days=1))
    [ids] = src.s2.lookup.call_args.args
    assert ids == ["2510.2"]
    assert row(db_session, "2510.2").s2_checked_at == NOW + timedelta(days=1)


def test_a_refusal_that_never_clears_ends_at_the_end_of_the_day(db_session: Session) -> None:
    # 06:00 start, a refusal that lasts all day: the waits are 5, 10, 20, 40 minutes and then every
    # 30, and the loop stops when the next pass would start after 22:00 Europe/Rome. The first day
    # of an S2 outage is a warning, like one missed lab-feed day.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    sleeps, wall, sleep = travelling_clock(NOW)
    res = run(db_session, src, followup=True, sleep=sleep, clock=lambda: wall[0])
    # This run starts at 06:00 UTC, which is 08:00 in Rome (CEST), so its window closes at 20:00
    # UTC. The ramp ends at 07:15 UTC and a 30-minute pass only fits while it starts by 20:00:
    # 07:15 + 30 min x 25 = 19:45, and the 26th would start at 20:15. So 4 ramp passes plus 25 = 29,
    # and 29 waits. (A 06:00 Rome start gets 33 passes, ADR 0008's 34 S2 lookups.)
    assert sleeps == [300, 600, 1200, 2400] + [1800] * 25
    assert res.counts["followup_passes"] == 29
    assert res.health is not None
    assert res.health.followup == {"passes": 29, "stopped": "time_of_day"}
    # First day: a warning, a succeeded run, and the papers still refused (and due tomorrow).
    assert res.status == "succeeded" and res.health.healthy
    assert res.health.warnings == ("s2 still refused after 29 follow-up passes: 1 ids",)
    assert res.counts["s2_refused"] == 1 and res.counts["s2_unchecked"] == 1
    assert row(db_session, "2510.1").s2_checked_at is None
    # The letter this run opened stays open, so tomorrow counts as the second day.
    letter = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "signals")).one()
    assert letter.resolved_at is None
    assert db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "signals")).all() == [
        letter
    ]


def test_a_second_day_of_refusals_is_a_failure(db_session: Session) -> None:
    # The s2/signals letter was opened by an earlier run: this is the second day in a row, so the
    # run is degraded and unhealthy, the same shape as a lab feed's second missed day.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    first = run(db_session, src)
    assert first.status == "partial"  # without --followup nothing waits

    sleeps, wall, sleep = travelling_clock(NOW + timedelta(days=1))
    res = run(
        db_session,
        src,
        now=NOW + timedelta(days=1),
        followup=True,
        sleep=sleep,
        clock=lambda: wall[0],
    )
    assert sleeps == [300, 600, 1200, 2400] + [1800] * 25  # the loop really ran
    assert res.status == "partial" and res.health is not None and not res.health.healthy
    assert res.health.warnings == ()
    assert "1 signal sources degraded" in res.health.problems
    assert "s2 still refused after 29 follow-up passes: 1 ids" in res.health.problems
    # One letter per continuing outage, not one per run or per pass.
    assert len(db_session.scalars(select(DeadLetter).where(DeadLetter.ref == "signals")).all()) == 1


def test_the_pass_cap_stops_a_loop_that_never_reaches_its_deadline(db_session: Session) -> None:
    # A clock that never reaches 22:00 (a frozen test clock) leaves the pass cap as the only bound.
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    sleeps: list[float] = []
    res = run(db_session, src, followup=True, sleep=sleeps.append)
    assert sleeps == [300, 600, 1200, 2400] + [1800] * 36
    assert res.counts["followup_passes"] == 40
    assert res.health is not None and res.health.followup == {"passes": 40, "stopped": "attempts"}
    assert res.health.warnings == ("s2 still refused after 40 follow-up passes: 1 ids",)


def test_a_run_with_nothing_refused_never_waits(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    sleeps: list[float] = []
    res = run(db_session, src, followup=True, sleep=sleeps.append)
    assert sleeps == []
    assert res.counts["followup_passes"] == 0
    assert res.health is not None and res.health.followup is None
    assert "followup" not in (res.health.as_dict())


def test_a_follow_up_pass_fetches_nothing(db_session: Session) -> None:
    # A pass retries Semantic Scholar only: arXiv, the feeds, Hugging Face and the classifier ran
    # once, at 06:00, and are never repeated. Proved by the call counts recorded across passes.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    calls: list[tuple[int, int, int]] = []
    left = [3]

    def sleep(seconds: float) -> None:
        calls.append(
            (
                src.arxiv.harvest.call_count,
                src.feeds.fetch_one.call_count,
                src.hf.papers.call_count,
            )
        )
        left[0] -= 1
        if left[0] == 0:  # the third pass answers
            src.s2.refused, src.s2.degraded = set(), False

    res = run(db_session, src, followup=True, sleep=sleep)
    # One capture before each of the three passes; the first three are the main pass's counts and
    # a pass that fetched would raise the numbers after the first.
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert res.counts["followup_passes"] == 3
    assert res.health is not None and res.health.healthy


def test_a_failed_admission_is_not_retried_when_s2_later_supplies_a_venue(
    db_session: Session,
) -> None:
    # The paper is refused by S2 and admitted from its Hugging Face upvotes in the morning, and
    # that admission fails. The pass now has a venue to propose, so only the "failed this run"
    # filter can stop it from being retried in-day: the attempt cap belongs to the next run.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    embedder = FlakyEmbedder(failures=1)
    _sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded = set(), False

    res = run(
        db_session,
        src,
        embedder=embedder,
        followup=True,
        sleep=recover,
        clock=lambda: wall[0],
    )
    assert res.counts["followup_passes"] == 1  # the pass ran and S2 answered
    assert src.s2.lookup.call_count == 2
    assert res.counts["failed"] == 1
    # One embed attempt in the whole run: the pass did not retry the failed admission.
    assert embedder.calls == 1
    assert row(db_session, "2510.2").status == "candidate"
    assert row(db_session, "2510.2").admit_attempts == 1


def test_the_lock_is_held_and_the_row_is_finished_while_waiting(
    db_session: Session, engine: Engine
) -> None:
    # During the wait the advisory lock is still held (a manual run would be skipped) and the row
    # is finished with a provisional verdict, never 'running', so the morning check reads a run
    # that is waiting rather than one that hung.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    seen: dict[str, object] = {}

    def sleep(seconds: float) -> None:
        # The row is read through the run's own session: `db_session` lives in a transaction the
        # test rolls back, so another connection could not see it. The advisory lock is the
        # cluster-wide fact, so that one is probed from its own connection.
        probe_row = db_session.scalars(select(Run)).one()
        seen["run_id"] = probe_row.id
        seen["status"] = probe_row.status
        seen["finished"] = probe_row.finished_at
        seen["health"] = probe_row.health
        with engine.connect() as conn:
            held = conn.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": INGEST_LOCK_KEY}
            ).scalar()
            if held:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": INGEST_LOCK_KEY})
                conn.commit()
        seen["lock_free"] = bool(held)
        src.s2.refused, src.s2.degraded = set(), False

    res = run(db_session, src, followup=True, sleep=sleep)
    assert seen["lock_free"] is False
    assert seen["status"] == "partial" and seen["finished"] is not None
    assert seen["health"] == {
        "healthy": False,
        "problems": ["status=partial", "1 signal sources degraded"],
        "warnings": [],
        "followup": {
            "pass": 0,
            "of": 40,
            "next_at": (NOW + timedelta(minutes=5)).isoformat(),
            "pending": {"s2": 1},
        },
    }
    assert res.run_id == seen["run_id"]


def test_the_listing_is_released_before_the_follow_up_wait(db_session: Session) -> None:
    # A window can carry thousands of records; like the models, the listing is dropped before a
    # wait that can last hours, so the idle process does not hold it.
    import weakref

    src = fake_sources([paper("2510.2", "Coding agent")])
    src.s2.lookup.return_value = {}
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    refs: list[weakref.ref[HfPaper]] = []

    def listing(since: date, until: date) -> dict[str, HfPaper]:
        record = HfPaper(upvotes=50)
        refs.append(weakref.ref(record))
        return {"2510.2": record}

    src.hf.papers.side_effect = listing
    seen: list[bool] = []

    def sleep(seconds: float) -> None:
        seen.append(refs[0]() is None)
        src.s2.refused, src.s2.degraded = set(), False

    res = run(db_session, src, followup=True, sleep=sleep)
    assert seen == [True]
    assert res.health is not None and res.health.healthy


def test_a_pass_that_admits_nothing_does_not_rebuild_the_models(db_session: Session) -> None:
    import weakref

    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = {}  # S2 answers, but has no venue or citations yet
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    refs: list[weakref.ref[FakeEmbedder]] = []

    def build() -> tuple[Classifier, Embedder]:
        embedder = FakeEmbedder()
        refs.append(weakref.ref(embedder))
        return FakeClassifier(PROFILES), embedder

    seen: list[int] = []

    def sleep(seconds: float) -> None:
        # Everything built so far is released before the wait: the factory's objects are dead.
        assert all(ref() is None for ref in refs), [ref() for ref in refs]
        seen.append(len(refs))
        src.s2.refused, src.s2.degraded = set(), False

    res = run_ingestion(
        db_session,
        src,
        build,
        Ledger(10**9),
        ADM,
        NOW,
        clock=lambda: NOW,
        followup=True,
        sleep=sleep,
    )
    # Built once for the main pass; the pass answered with no signals, so nothing was embedded.
    assert seen == [1] and len(refs) == 1
    assert res.health is not None and res.health.healthy


def test_a_pass_that_admits_something_rebuilds_the_models(db_session: Session) -> None:
    import weakref

    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE  # a venue: the pass admits a first time, so it embeds
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    refs: list[weakref.ref[FakeEmbedder]] = []

    def build() -> tuple[Classifier, Embedder]:
        embedder = FakeEmbedder()
        refs.append(weakref.ref(embedder))
        return FakeClassifier(PROFILES), embedder

    def sleep(seconds: float) -> None:
        assert all(ref() is None for ref in refs)
        src.s2.refused, src.s2.degraded = set(), False

    res = run_ingestion(
        db_session,
        src,
        build,
        Ledger(10**9),
        ADM,
        NOW,
        clock=lambda: NOW,
        followup=True,
        sleep=sleep,
    )
    # The main pass's pair was released, then rebuilt for the pass's first admission.
    assert len(refs) == 2
    assert row(db_session, "2510.2").tier == "peer_reviewed"
    assert res.health is not None and res.health.healthy


def test_an_exception_inside_the_loop_still_fails_the_run(db_session: Session) -> None:
    # The loop runs inside the run's try: a SystemExit or KeyboardInterrupt between passes must
    # still close the row as failed, not leave it waiting forever.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    api_call = src.s2.lookup.side_effect
    calls = {"n": 0}

    def die_after_the_main_pass(ids: list[str]) -> object:
        calls["n"] += 1
        if calls["n"] > 1:  # the main pass is call 1; anything after it is a follow-up pass
            raise SystemExit("killed between passes")
        return api_call(ids)

    src.s2.lookup.side_effect = die_after_the_main_pass
    _sleeps, wall, sleep = travelling_clock(NOW)
    with pytest.raises(SystemExit, match="killed between passes"):
        run(db_session, src, followup=True, sleep=sleep, clock=lambda: wall[0])
    failed = db_session.scalars(select(Run).where(Run.status == "failed")).one()
    assert failed.health is None and failed.finished_at is not None


def test_a_pass_that_succeeds_late_in_the_window_still_rescues_the_day(db_session: Session) -> None:
    # The field evidence (2026-10-23): S2 refused at 06:03 and again at 09:52, so a burst can last
    # hours. A pass late in the window must still recover the day and leave it healthy.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    sleeps, wall, sleep = travelling_clock(NOW)
    left = [12]

    def recover_late(seconds: float) -> None:
        sleep(seconds)
        left[0] -= 1
        if left[0] == 0:  # the 12th pass, about 13:30 in Rome
            src.s2.refused, src.s2.degraded = set(), False

    res = run(db_session, src, followup=True, sleep=recover_late, clock=lambda: wall[0])
    assert sleeps == [300, 600, 1200, 2400] + [1800] * 8
    assert res.health is not None and res.health.healthy
    assert res.health.followup == {"passes": 12, "stopped": "resolved"}
    assert res.health.warnings == ()
    assert row(db_session, "2510.2").tier == "peer_reviewed"


def test_a_run_started_after_the_deadline_takes_no_pass(db_session: Session) -> None:
    # A Persistent=true catch-up can start the run at 23:00 Rome time. No pass fits before the
    # window closes, so the run waits for nothing and says so; the refused papers stay due.
    late = datetime(2026, 10, 12, 21, tzinfo=UTC)  # 23:00 in Rome, past the 22:00 deadline
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    sleeps: list[float] = []
    res = run(db_session, src, now=late, followup=True, sleep=sleeps.append)
    assert sleeps == []
    assert res.counts["followup_passes"] == 0
    assert res.health is not None and res.health.followup == {"passes": 0, "stopped": "time_of_day"}
    assert res.health.warnings == ("s2 refused and no pass fit before 22:00: 1 ids",)
    assert res.health.healthy  # the first day of an outage is still a warning


def test_each_pass_logs_one_line_with_the_idle_footprint(db_session: Session) -> None:
    # The journal is the captain's view of a long morning, and a refusal can last hours: one line
    # per pass (never one per attempt from the pipeline's side), carrying the waiting process's RSS,
    # plus one line when the loop ends.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    left = [3]

    def sleep(seconds: float) -> None:
        left[0] -= 1
        if left[0] == 0:  # the third pass answers
            src.s2.refused, src.s2.degraded = set(), False

    with capture_logs() as logs:
        run(db_session, src, followup=True, sleep=sleep)
    passes = [event for event in logs if event["event"] == "followup_pass"]
    finished = [event for event in logs if event["event"] == "followup_finished"]
    assert [event["pass_"] for event in passes] == [1, 2, 3]
    assert all("rss_mb" in event and event["pending"] == 1 for event in passes)
    assert [(event["passes"], event["stopped"], event["pending"]) for event in finished] == [
        (3, "resolved", 0)
    ]


def test_an_early_start_cannot_outlive_the_unit_timeout(db_session: Session) -> None:
    # A Persistent catch-up can start the run at 00:30 Rome time. The clock face alone would let it
    # sleep until 19:45, and systemd's 17 h timeout would kill it in the middle of a wait; the
    # window is also capped at 16 h relative to the start, so it stops before that.
    rome = ZoneInfo("Europe/Rome")
    start = datetime(2026, 10, 13, 0, 30, tzinfo=rome).astimezone(UTC)
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    sleeps, wall, sleep = travelling_clock(start)
    res = run(db_session, src, now=start, followup=True, sleep=sleep, clock=lambda: wall[0])
    assert sum(sleeps) <= 16 * 3600
    assert res.health is not None and res.health.followup == {
        "passes": 33,
        "stopped": "time_of_day",
    }
    # The last pass starts at 16:15 Rome, half an hour inside the 16 h window that closes at 16:30.
    assert start + timedelta(seconds=sum(sleeps)) == datetime(
        2026, 10, 13, 16, 15, tzinfo=rome
    ).astimezone(UTC)


def test_a_pass_starting_exactly_at_the_deadline_does_not_run(db_session: Session) -> None:
    # The window closes at 22:00, not after it: a pass that would start at exactly 22:00 is too
    # late. `now` 21:55 Rome plus the 5-minute first wait lands exactly on the deadline.
    rome = ZoneInfo("Europe/Rome")
    start = datetime(2026, 10, 12, 21, 55, tzinfo=rome).astimezone(UTC)
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    sleeps: list[float] = []
    res = run(db_session, src, now=start, followup=True, sleep=sleeps.append)
    assert sleeps == []
    assert res.counts["followup_passes"] == 0
    assert res.health is not None and res.health.followup == {"passes": 0, "stopped": "time_of_day"}


def test_a_model_that_cannot_be_built_fails_the_run_not_the_items(db_session: Session) -> None:
    # The first build happens before phase 2: a missing model file fails the run (exit 1, row
    # failed) instead of being charged to every new item's attempt cap.
    src = fake_sources([paper("2510.1", "Agent harness")])

    def build() -> tuple[Classifier, Embedder]:
        raise RuntimeError("model files missing")

    with pytest.raises(ModelUnavailable, match="model files missing"):
        run_ingestion(db_session, src, build, Ledger(10**9), ADM, NOW, clock=lambda: NOW)
    failed = db_session.scalars(select(Run).where(Run.status == "failed")).one()
    assert failed.health is None
    # Nothing was classified, admitted or dead-lettered: the item was never touched.
    assert db_session.scalars(select(Item)).all() == []
    assert db_session.scalars(select(DeadLetter)).all() == []


def test_a_rebuild_failure_in_a_pass_fails_the_run_not_the_item(db_session: Session) -> None:
    # The pair released before the wait cannot be built again: that is the run's failure, not an
    # admission failure charged to the paper the pass was about to admit.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    builds = {"n": 0}

    def build() -> tuple[Classifier, Embedder]:
        builds["n"] += 1
        if builds["n"] > 1:
            raise RuntimeError("model files vanished")
        return FakeClassifier(PROFILES), FakeEmbedder()

    _sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded = set(), False

    with pytest.raises(ModelUnavailable, match="model files vanished"):
        run_ingestion(
            db_session,
            src,
            build,
            Ledger(10**9),
            ADM,
            NOW,
            clock=lambda: wall[0],
            followup=True,
            sleep=recover,
        )
    failed = db_session.scalars(select(Run).where(Run.status == "failed")).one()
    assert failed.health is None
    item = row(db_session, "2510.2")
    assert item.status == "candidate" and item.admit_attempts == 0
    assert Repository(db_session).open_item_dead_letters() == []


def test_an_anomaly_is_not_masked_by_a_recovered_refusal(db_session: Session) -> None:
    # The morning S2 call both refused one chunk and answered another with a shape this client
    # cannot read. The pass clears the refusal; the anomaly is not a throttle, so the run stays
    # degraded instead of reading healthy (the reviewer's masking probe, inverted).
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = {}
    src.s2.refused = {"2510.2"}
    src.s2.degraded = True
    src.s2.failure = "s2 chunk at 0: 400 rows for 500 ids"
    src.s2.anomaly = src.s2.failure
    sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded, src.s2.failure = set(), False, None

    res = run(db_session, src, followup=True, sleep=recover, clock=lambda: wall[0])
    assert sleeps == [300]
    assert res.status == "partial" and res.health is not None and not res.health.healthy
    assert "1 signal sources degraded" in res.health.problems
    # The refusal itself is resolved, so there is no first-day warning to add.
    assert res.health.warnings == ()
    assert res.counts["s2_refused"] == 0


def test_a_pass_asks_only_the_refused_ids(db_session: Session) -> None:
    # The main pass asks for every target once; a pass asks for the refused ids alone, never for an
    # id S2 already answered and never for one it simply does not know (a retry there would spend
    # the window on papers no retry can help).
    src = fake_sources(
        [
            paper("2510.1", "Agent harness"),
            paper("2510.2", "Coding agent"),
            paper("2510.3", "Tool use survey"),
        ]
    )
    src.hf.papers.return_value = {}
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    # S2 knows 2510.1, refuses 2510.2 and has never heard of 2510.3 (it is omitted from the answer).
    unknown = {"2510.3"}

    def s2_lookup(ids: list[str]) -> dict[str, Signals]:
        src.s2.checked_ids = set(ids) - src.s2.refused - unknown
        src.s2.refused_ids = set(ids) & src.s2.refused
        if not src.s2.degraded:
            src.s2.anomaly = None
        return {k: v for k, v in S2_VENUE.items() if k in src.s2.checked_ids}

    src.s2.lookup.side_effect = s2_lookup
    _sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded = set(), False

    res = run(db_session, src, followup=True, sleep=recover, clock=lambda: wall[0])
    asked = [call.args[0] for call in src.s2.lookup.call_args_list]
    assert asked == [["2510.1", "2510.2", "2510.3"], ["2510.2"]]
    # The id S2 does not know is still counted as unchecked, and it is not a refusal.
    assert res.counts["s2_unchecked"] == 1 and res.counts["s2_refused"] == 0


def test_an_id_s2_does_not_know_never_starts_a_loop(db_session: Session) -> None:
    # An id S2 leaves out of an otherwise fine answer is not a refusal: it stays unchecked and due,
    # but it must never open a window of retries that could not possibly help.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}

    def unknown(ids: list[str]) -> dict[str, Signals]:
        src.s2.checked_ids = set()  # omitted from the answer
        src.s2.refused_ids = set()  # and not refused either
        return {}

    src.s2.lookup.side_effect = unknown
    sleeps: list[float] = []
    res = run(db_session, src, followup=True, sleep=sleeps.append)
    assert sleeps == []
    assert res.counts["followup_passes"] == 0
    assert res.counts["s2_unchecked"] == 1 and res.counts["s2_refused"] == 0
    assert res.health is not None and res.health.followup is None


def test_no_transaction_is_open_while_waiting(db_session: Session) -> None:
    # The row is committed before the wait, not merely written: the morning check reads the
    # database through its own psql session, and an open transaction would hide the row for hours.
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True

    def sleep(seconds: float) -> None:
        assert not db_session.in_transaction()
        src.s2.refused, src.s2.degraded = set(), False

    res = run(db_session, src, followup=True, sleep=sleep)
    assert res.health is not None and res.health.healthy


def test_an_exhausted_budget_takes_no_follow_up_pass(db_session: Session) -> None:
    # A run that spent its token budget on the morning admission cannot admit anything later, so it
    # does not wait: the refused papers stay due for the next run.
    class Costly(FakeEmbedder):
        def __init__(self, ledger: Ledger) -> None:
            super().__init__()
            self.ledger = ledger

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.ledger.charge("fake", 1000)
            return super().embed(texts)

    ledger = Ledger(500)
    src = fake_sources([paper("2510.2", "Coding agent")])  # 50 HF upvotes: a first admission
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    sleeps: list[float] = []
    res = run(
        db_session, src, embedder=Costly(ledger), ledger=ledger, followup=True, sleep=sleeps.append
    )
    assert sleeps == []
    assert res.status == "budget_exhausted"
    assert res.counts["followup_passes"] == 0
    assert res.health is not None and res.health.followup is None
    assert res.counts["s2_refused"] == 1


def test_a_budget_stop_inside_a_pass_ends_the_loop(db_session: Session) -> None:
    # The pass is the first thing that needs an embedding, and it cannot afford one: the run stops
    # as a budget-exhausted run and the loop records that, rather than waiting for a pass that could
    # never admit.
    class Costly(FakeEmbedder):
        def __init__(self, ledger: Ledger) -> None:
            super().__init__()
            self.ledger = ledger

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.ledger.charge("fake", 1000)
            return super().embed(texts)

    ledger = Ledger(500)
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}  # nothing is admitted in the morning, so the budget survives
    src.s2.lookup.return_value = S2_VENUE
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded = set(), False

    res = run(
        db_session,
        src,
        embedder=Costly(ledger),
        ledger=ledger,
        followup=True,
        sleep=recover,
        clock=lambda: wall[0],
    )
    assert sleeps == [300]
    assert res.status == "budget_exhausted"
    assert res.counts["followup_passes"] == 1
    assert res.health is not None
    assert res.health.followup == {"passes": 1, "stopped": "budget_exhausted"}
    assert row(db_session, "2510.2").status == "candidate"


def test_an_anomaly_that_first_appears_in_a_pass_survives_a_later_clean_one(
    db_session: Session,
) -> None:
    # A pass can be the first to see the bad shape (its answer is short or unpaired) while a later
    # pass answers cleanly: the anomaly belongs to the run, so the day stays degraded.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.refused, src.s2.degraded, src.s2.failure = {"2510.2"}, True, "HTTP 429 after 5 attempts"
    _sleeps, wall, sleep = travelling_clock(NOW)
    calls = {"n": 0}

    def hook(seconds: float) -> None:
        sleep(seconds)
        calls["n"] += 1
        if calls["n"] == 1:  # pass 1: refused again, and an answer shape the client cannot read
            src.s2.failure = "s2 chunk at 0: 3 rows for 10 ids"
            src.s2.anomaly = src.s2.failure
        else:  # pass 2: clean
            src.s2.refused, src.s2.degraded, src.s2.failure, src.s2.anomaly = (
                set(),
                False,
                None,
                None,
            )

    res = run(db_session, src, followup=True, sleep=hook, clock=lambda: wall[0])
    assert calls["n"] == 2
    assert res.status == "partial" and res.health is not None and not res.health.healthy
    assert "1 signal sources degraded" in res.health.problems


def test_a_run_still_refused_at_the_deadline_keeps_an_anomaly_on_its_first_day(
    db_session: Session,
) -> None:
    # The first day of a refusal is a warning, but an anomaly is not the refusal: it keeps the run
    # degraded, so the warning cannot make a month of bad answers look like a healthy morning.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    src.s2.failure = "s2 chunk at 0: 400 rows for 500 ids"
    src.s2.anomaly = src.s2.failure
    _sleeps, wall, sleep = travelling_clock(NOW)
    res = run(db_session, src, followup=True, sleep=sleep, clock=lambda: wall[0])
    assert res.health is not None and any("still refused" in w for w in res.health.warnings)
    assert res.status == "partial" and not res.health.healthy
    assert "1 signal sources degraded" in res.health.problems


def test_a_recovered_pass_with_an_anomaly_closes_the_outage_letter(db_session: Session) -> None:
    # The source is answering again, so its outage letter closes; tomorrow's first refusal is a
    # first day, not a second one. The anomaly itself stays in this run's verdict.
    src = fake_sources([paper("2510.1", "Agent harness"), paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.refused, src.s2.degraded = {"2510.2"}, True
    src.s2.failure = "s2 chunk at 0: 400 rows for 500 ids"
    src.s2.anomaly = src.s2.failure
    _sleeps, wall, sleep = travelling_clock(NOW)

    def recover(seconds: float) -> None:
        sleep(seconds)
        src.s2.refused, src.s2.degraded, src.s2.failure = set(), False, None

    res = run(db_session, src, followup=True, sleep=recover, clock=lambda: wall[0])
    assert not Repository(db_session).has_open_dead_letter("s2", "signals")
    assert res.status == "partial" and res.health is not None and not res.health.healthy


def test_the_window_cap_counts_from_the_start_not_from_the_end_of_the_main_pass(
    db_session: Session,
) -> None:
    # The 16 h cap is measured from the run's `now`, so a main pass that took two hours shortens the
    # window instead of pushing it two hours later, which would eat the margin below the unit's
    # 17 h timeout.
    rome = ZoneInfo("Europe/Rome")
    start = datetime(2026, 10, 13, 0, 30, tzinfo=rome).astimezone(UTC)
    src = fake_sources([paper("2510.2", "Coding agent")])
    src.hf.papers.return_value = {}
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 5 attempts"
    # The main pass takes two hours: the clock is already two hours past the run's `now`.
    sleeps, wall, sleep = travelling_clock(start + timedelta(hours=2))
    run(db_session, src, now=start, followup=True, sleep=sleep, clock=lambda: wall[0])
    assert wall[0] <= start + timedelta(hours=16)
    assert sum(sleeps) <= 14 * 3600  # the two hours the main pass took are inside the cap
