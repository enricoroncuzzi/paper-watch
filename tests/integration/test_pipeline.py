import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import DEFAULT, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperwatch.admission import load_admission
from paperwatch.config import (
    ARXIV_ANNOUNCEMENT_LAG,
    FEED_OVERLAP,
    HF_RECHECK_HORIZON,
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
from paperwatch.ingest.pipeline import RunResult, Sources, run_ingestion
from paperwatch.llm.classifier import Classifier, FakeClassifier
from paperwatch.llm.embedder import Embedder, FakeEmbedder
from paperwatch.llm.ledger import Ledger
from paperwatch.sources.feeds import Feed
from paperwatch.store.models import DeadLetter, Item, Run, TierHistory
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)
PROFILES = load_profiles(Path("config/profiles.yaml"))
ADM = load_admission(Path("config/admission.yaml"))
FEED = Feed(lab="TestLab", url="https://lab.example/feed.xml", region="us-eu")


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


def fake_sources(papers: list[SourceItem], feed_items: list[SourceItem] | None = None) -> Sources:
    s = Sources(
        arxiv=MagicMock(), hf=MagicMock(), s2=MagicMock(), feeds=MagicMock(), feed_list=[FEED]
    )
    s.arxiv.fetch.return_value = papers
    s.arxiv.rejected = []  # filled by the real client when an entry cannot be parsed
    # Signal sources report outages through these flags instead of raising (Task 4).
    s.hf.degraded, s.hf.failure = False, None
    s.s2.degraded, s.s2.failure = False, None
    s.hf.upvotes.return_value = {p.external_id: 50 for p in papers}
    s.s2.lookup.return_value = {}
    s.s2.refused = set()  # ids whose Semantic Scholar chunk the stub refuses

    def s2_lookup(ids: list[str]) -> object:
        # Like the real client: ids in an answered chunk are checked. A test refuses chunks
        # through `refused`, or every chunk by setting `degraded` alone (a whole outage).
        outage = s.s2.degraded and not s.s2.refused
        s.s2.checked_ids = set() if outage else set(ids) - s.s2.refused
        return DEFAULT  # then the mock returns lookup.return_value as usual

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
) -> RunResult:
    return run_ingestion(
        session,
        sources,
        classifier or FakeClassifier(PROFILES),
        embedder or FakeEmbedder(),
        ledger or Ledger(10**9),
        ADM,
        now,
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
    src.hf.upvotes.return_value = {}  # no evidence: 2510.1 stays a candidate
    for day in range(8):  # the whole announcement-lag overlap re-sends both papers daily
        run(db_session, src, now=NOW + timedelta(days=day), classifier=clf)
    assert clf.calls == 2
    statuses = dict(db_session.execute(select(Item.external_id, Item.status)).all())
    assert statuses == {"2510.1": "candidate", "2510.2": "irrelevant"}


def test_candidate_admitted_when_upvotes_arrive_late(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.hf.upvotes.return_value = {"2510.1": 3}
    first = run(db_session, src)
    assert first.counts["stored"] == 0 and row(db_session, "2510.1").status == "candidate"
    day20 = NOW + timedelta(days=20)
    src.arxiv.fetch.return_value = []  # long out of the arXiv fetch window
    src.hf.upvotes.return_value = {"2510.1": 14}
    res = run(db_session, src, now=day20)
    assert src.hf.upvotes.call_args.args == ((day20 - HF_RECHECK_HORIZON).date(), day20.date())
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


def test_lab_report_admitted_same_day_then_basis_improves(db_session: Session) -> None:
    # A lab team's report is admitted on day one by authorship; when community evidence
    # arrives the tier stays, but the recorded basis says it is no longer authority alone.
    lab = dataclasses.replace(paper("2510.1", "Agent harness"), authors=("DeepSeek-AI",))
    src = fake_sources([lab])
    src.hf.upvotes.return_value = {}
    first = run(db_session, src)
    assert first.counts["stored"] == 1
    assert (row(db_session, "2510.1").tier, row(db_session, "2510.1").admission_basis) == (
        "vetted_preprint",
        "lab_author",
    )
    day3 = NOW + timedelta(days=3)
    src.arxiv.fetch.return_value = []
    src.hf.upvotes.return_value = {"2510.1": 120}
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
    src.hf.upvotes.return_value = {}
    src.s2.lookup.return_value = {"2510.1": Signals(venue="arXiv.org", citation_count=6)}
    run(db_session, src)
    day90 = NOW + timedelta(days=90)
    src.arxiv.fetch.return_value = []
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
    src.hf.upvotes.return_value = {}
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
    src.hf.upvotes.return_value = {"2510.31": 12}
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
    src.hf.upvotes.return_value = {}  # no evidence: both stay candidates, checked daily
    run(db_session, src)
    day1 = NOW + timedelta(days=1)
    src.arxiv.fetch.return_value = []
    src.s2.refused = {"2510.2"}
    src.s2.degraded, src.s2.failure = True, "s2 chunk at 500: HTTP 429"
    res = run(db_session, src, now=day1)
    assert res.status == "partial" and res.counts["s2_unchecked"] == 1
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
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src, classifier=clf, embedder=emb)
    v2 = dataclasses.replace(
        paper("2510.1", "Agent harness", version=2),
        text="Agent harness. We add a memory module for tool use.",
    )
    src.arxiv.fetch.return_value = [v2]
    res = run(db_session, src, now=NOW + timedelta(days=1), classifier=clf, embedder=emb)
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
    src.arxiv.fetch.return_value = [v2]
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


def test_source_failure_records_dead_letter_and_partial_run(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.feeds.fetch_one.side_effect = RuntimeError("HTTP 500")
    res = run(db_session, src)
    assert res.status == "partial" and res.counts["stored"] == 1
    # The parked-letter row for the fetched paper is resolved; the feed failure is the open one.
    dl = db_session.scalars(select(DeadLetter).where(DeadLetter.ref == FEED.url)).one()
    assert dl.window_end == NOW


def test_failed_source_window_is_replayed_next_run(db_session: Session) -> None:
    src = fake_sources([], feed_items=[blog("p1")])
    src.arxiv.fetch.side_effect = RuntimeError("arXiv down")
    first = run(db_session, src)
    assert first.status == "partial" and first.counts["stored"] == 1
    src.arxiv.fetch.side_effect = None
    src.arxiv.fetch.return_value = [paper("2510.5", "Coding agent")]
    src.hf.upvotes.return_value = {"2510.5": 50}
    later = NOW + timedelta(days=7)
    second = run(db_session, src, now=later)
    since, until = src.arxiv.fetch.call_args.args
    # The pinned cursor is the missed window's start; the fetch reaches back the lag before it.
    assert since == NOW - timedelta(days=7) - ARXIV_ANNOUNCEMENT_LAG and until == later
    assert second.status == "succeeded" and second.counts["stored"] == 1


def test_admission_failure_is_retried_next_run(db_session: Session) -> None:
    emb = FlakyEmbedder(failures=2)
    src = fake_sources([paper("2510.6", "Agent harness")])
    first = run(db_session, src, embedder=emb)
    assert first.status == "partial" and first.counts["failed"] == 1
    assert row(db_session, "2510.6").status == "candidate"
    # The failed embed rolls the S2 check time back with the item, so the paper stays due.
    assert row(db_session, "2510.6").s2_checked_at is None
    src.arxiv.fetch.return_value = []
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
    src.arxiv.fetch.return_value = [trigger, poison]
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
    src.arxiv.fetch.return_value = []
    res = run(db_session, src, now=NOW + timedelta(days=1))
    assert res.status == "succeeded"
    assert row(db_session, "2510.old").status == "admitted"


def test_corrupt_dead_letter_payload_is_counted_not_fatal(db_session: Session) -> None:
    class Boom(FakeClassifier):
        def classify(self, item: SourceItem) -> Classification:
            raise RuntimeError("model 500")

    repo = Repository(db_session)
    rid = repo.start_run(NOW, NOW + timedelta(days=1))
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
    src.arxiv.fetch.side_effect = SystemExit("killed during fetch")
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
    src.arxiv.fetch.return_value = []
    run(db_session, src, now=NOW + timedelta(days=1))
    db_session.refresh(stale)
    assert stale.status == "failed" and stale.finished_at is not None


def test_arxiv_refetch_overlaps_announcement_lag(db_session: Session) -> None:
    # A paper submitted before the cursor but announced after it must still be fetched.
    src = fake_sources([paper("2510.1", "Agent harness")])
    run(db_session, src)
    tomorrow = NOW + timedelta(days=1)
    run(db_session, src, now=tomorrow)
    since, until = src.arxiv.fetch.call_args.args
    assert since == NOW - ARXIV_ANNOUNCEMENT_LAG and until == tomorrow
    assert Repository(db_session).cursor("arxiv") == tomorrow


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
    assert Repository(db_session).cursor("arxiv") == NOW
    # The lag overlap re-reads the same bad entry tomorrow: it is not recorded twice.
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
    assert row_.health == {"healthy": True, "problems": []}
    # An arXiv fetch that returns nothing over a 7-day-plus window is a broken run.
    empty = run(db_session, fake_sources([]), now=NOW + timedelta(days=1))
    problems = db_session.get_one(Run, empty.run_id).health["problems"]
    assert "arxiv fetched 0 items" in problems


def test_signal_source_outage_is_visible_and_cursor_advances(db_session: Session) -> None:
    src = fake_sources([paper("2510.1", "Agent harness")])
    src.s2.degraded, src.s2.failure = True, "HTTP 429 after 3 attempts"
    res = run(db_session, src)
    assert res.status == "partial" and res.counts["signal_failures"] == 1
    assert res.counts["s2_unchecked"] == 1  # still due: the next run asks again
    dl = db_session.scalars(select(DeadLetter).where(DeadLetter.source == "s2")).one()
    assert "429" in dl.error
    # arXiv was fetched fine, so its cursor moves; the next run's re-check repeats the lookup.
    assert Repository(db_session).cursor("arxiv") == NOW
    health = db_session.get_one(Run, res.run_id).health
    assert "1 signal sources degraded" in health["problems"]


def test_daily_runs_over_a_week_lose_nothing(db_session: Session) -> None:
    # Replays seven consecutive daily runs against a fake arXiv that, like the real one,
    # only returns papers already announced. One paper is announced three days after
    # submission; the feed is down on day 3. Nothing may be lost or stored twice.
    day0 = NOW
    catalogue = [  # (id, submitted, announced)
        ("2510.11", day0 - timedelta(hours=30), day0 - timedelta(hours=8)),
        ("2510.12", day0 + timedelta(hours=10), day0 + timedelta(hours=20)),
        ("2510.13", day0 + timedelta(days=1, hours=12), day0 + timedelta(days=4, hours=14)),
        ("2510.14", day0 + timedelta(days=5, hours=2), day0 + timedelta(days=5, hours=13)),
    ]

    def announced(since: datetime, until: datetime) -> list[SourceItem]:
        return [
            dataclasses.replace(paper(eid, "Coding agent harness"), published_at=sub)
            for eid, sub, ann in catalogue
            if since <= sub < until and ann <= until
        ]

    src = fake_sources([], feed_items=[blog("p1")])
    src.arxiv.fetch.side_effect = announced
    src.hf.upvotes.return_value = {eid: 50 for eid, _, _ in catalogue}
    statuses = []
    for day in range(7):
        now = day0 + timedelta(days=day)
        src.feeds.fetch_one.side_effect = RuntimeError("HTTP 503") if day == 3 else None
        statuses.append(run(db_session, src, now=now).status)
    assert statuses[3] == "partial" and statuses.count("succeeded") == 6
    admitted = sorted(
        db_session.scalars(
            select(Item.external_id).where(Item.source == "arxiv", Item.status == "admitted")
        )
    )
    assert admitted == [eid for eid, _, _ in catalogue]
    assert len(db_session.scalars(select(Item).where(Item.source == "feed")).all()) == 1
