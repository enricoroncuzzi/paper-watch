"""Integration tests for the repository against a real migrated Postgres."""

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from paperwatch.config import MAX_ITEM_ATTEMPTS, S2_RECHECK_SLACK
from paperwatch.domain import AdmissionBasis, Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.store.models import DeadLetter, Item, Run, TierHistory
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)
ITEM = SourceItem(
    "arxiv",
    "2510.00001",
    "Agent harness design",
    ("A",),
    "We study harnesses.",
    "https://arxiv.org/abs/2510.00001",
    datetime(2026, 10, 1, tzinfo=UTC),
)
VEC = FakeEmbedder().embed([ITEM.text])[0]


def test_upsert_is_idempotent_on_source_and_external_id(db_session: Session) -> None:
    repo = Repository(db_session)
    a = repo.upsert_item(ITEM, Tier.VETTED_PREPRINT, "agentic-engineering", 0.9, Signals(), VEC)
    # A second upsert with a stronger claim must not admit again or add a history row.
    b = repo.upsert_item(
        ITEM,
        Tier.PEER_REVIEWED,
        "agentic-engineering",
        0.95,
        Signals(venue="ICLR", citation_count=10),
        VEC,
    )
    assert a == b
    assert repo.existing_ids("arxiv", ["2510.00001", "2510.00002"]) == {"2510.00001"}
    hist = db_session.scalars(select(TierHistory).where(TierHistory.item_id == a)).all()
    assert len(hist) == 1
    assert db_session.get_one(Item, a).tier == Tier.VETTED_PREPRINT.value


def test_candidate_is_stored_without_tier_or_embedding(db_session: Session) -> None:
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9)
    assert repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9) == item_id
    stored = repo.stored("arxiv", "2510.00001")
    assert stored is not None and stored.status == "candidate" and stored.tier is None
    assert repo.has_item_url("arxiv", ITEM.url) and not repo.has_item_url("feed", ITEM.url)
    assert db_session.get_one(Item, item_id).embedding is None


def test_admitted_item_without_embedding_is_refused(db_session: Session) -> None:
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9)
    with pytest.raises(ValueError):
        repo.admit_item(
            item_id, Tier.VETTED_PREPRINT, AdmissionBasis.HF_UPVOTES, None, Signals(), NOW
        )
    row = db_session.get_one(Item, item_id)
    # Bypassing the repository must still be refused by the database. With tier and basis set,
    # ck_items_admitted_has_embedding is the only constraint that can refuse this write.
    row.status = "admitted"
    row.tier = Tier.VETTED_PREPRINT.value
    row.admission_basis = AdmissionBasis.HF_UPVOTES.value
    with pytest.raises(IntegrityError, match="ck_items_admitted_has_embedding"):
        db_session.flush()


def test_admit_item_refuses_an_irrelevant_item(db_session: Session) -> None:
    # An irrelevant item has no tier or embedding; admitting it would make it searchable.
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "irrelevant", None, 0.0)
    with pytest.raises(ValueError, match="was classified irrelevant and cannot be admitted"):
        repo.admit_item(
            item_id, Tier.VETTED_PREPRINT, AdmissionBasis.HF_UPVOTES, VEC, Signals(), NOW
        )
    # Nothing was written before the refusal, so no tier history row exists for the item.
    assert db_session.scalars(select(TierHistory).where(TierHistory.item_id == item_id)).all() == []
    assert db_session.get_one(Item, item_id).status == "irrelevant"


def test_admission_and_promotion_are_recorded_with_evidence(db_session: Session) -> None:
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9)
    repo.record_admit_failure(item_id, "embedding service 503")  # a failure before success
    upvoted = repo.merge_signals(item_id, Signals(hf_upvotes=14))
    repo.admit_item(item_id, Tier.VETTED_PREPRINT, AdmissionBasis.HF_UPVOTES, VEC, upvoted, NOW)
    accepted = repo.merge_signals(item_id, Signals(venue="ICLR", venue_aliases=("ICLR",)))
    repo.admit_item(
        item_id, Tier.PEER_REVIEWED, AdmissionBasis.VENUE, None, accepted, NOW + timedelta(days=90)
    )
    hist = db_session.scalars(select(TierHistory).order_by(TierHistory.id)).all()
    row = db_session.get_one(Item, item_id)
    assert row.admit_attempts == 0 and row.admission_basis == "venue"
    assert [(h.from_tier, h.to_tier, h.basis, h.changed_at) for h in hist] == [
        (None, "vetted_preprint", "hf_upvotes", NOW),
        ("vetted_preprint", "peer_reviewed", "venue", NOW + timedelta(days=90)),
    ]
    assert hist[1].evidence["venue"] == "ICLR" and hist[1].evidence["hf_upvotes"] == 14


def test_basis_improvement_keeps_the_tier_and_is_recorded(db_session: Session) -> None:
    # A lab's own report later vetted by the community: same tier, stronger recorded basis.
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9)
    repo.admit_item(item_id, Tier.VETTED_PREPRINT, AdmissionBasis.LAB_AUTHOR, VEC, Signals(), NOW)
    upvoted = repo.merge_signals(item_id, Signals(hf_upvotes=40))
    repo.admit_item(
        item_id,
        Tier.VETTED_PREPRINT,
        AdmissionBasis.HF_UPVOTES,
        None,
        upvoted,
        NOW + timedelta(days=2),
    )
    stored = repo.stored("arxiv", ITEM.external_id)
    assert stored is not None and stored.basis == AdmissionBasis.HF_UPVOTES
    hist = db_session.scalars(select(TierHistory).order_by(TierHistory.id)).all()
    assert [(h.from_tier, h.to_tier, h.basis) for h in hist] == [
        (None, "vetted_preprint", "lab_author"),
        ("vetted_preprint", "vetted_preprint", "hf_upvotes"),
    ]


def test_merge_keeps_the_strongest_evidence(db_session: Session) -> None:
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9)
    repo.merge_signals(
        item_id, Signals(venue="ICLR", venue_aliases=("ICLR",), citation_count=9, hf_upvotes=20)
    )
    # A later, weaker answer (outage, preprint server, fewer counts) never erases evidence.
    merged = repo.merge_signals(item_id, Signals(venue="arXiv.org", citation_count=4))
    assert merged == Signals(venue="ICLR", venue_aliases=("ICLR",), citation_count=9, hf_upvotes=20)
    merged = repo.merge_signals(item_id, Signals())
    assert merged.venue == "ICLR" and merged.hf_upvotes == 20


def test_preprint_venue_is_replaced_by_a_real_one(db_session: Session) -> None:
    repo = Repository(db_session)
    item_id = repo.record_classified(ITEM, "candidate", "agentic-engineering", 0.9)
    repo.merge_signals(item_id, Signals(venue="arXiv.org"))
    merged = repo.merge_signals(item_id, Signals(venue="NeurIPS", venue_aliases=("NeurIPS",)))
    assert merged.venue == "NeurIPS" and merged.venue_aliases == ("NeurIPS",)


def test_replace_text_updates_version_and_embedding(db_session: Session) -> None:
    repo = Repository(db_session)
    item_id = repo.upsert_item(
        ITEM, Tier.VETTED_PREPRINT, "agentic-engineering", 0.9, Signals(), VEC
    )
    v2 = dataclasses.replace(ITEM, text="We study harnesses and memory.", version=2)
    new_vec = FakeEmbedder().embed([v2.text])[0]
    repo.replace_text(item_id, v2, new_vec)
    row = db_session.get_one(Item, item_id)
    assert (row.text, row.version) == (v2.text, 2)
    assert list(row.embedding) == pytest.approx(new_vec)


def test_recheck_targets_respect_status_horizon_and_attempts(db_session: Session) -> None:
    repo = Repository(db_session)

    def add(eid: str, age_days: int, source: str = "arxiv") -> SourceItem:
        return dataclasses.replace(
            ITEM, source=source, external_id=eid, published_at=NOW - timedelta(days=age_days)
        )

    young = repo.record_classified(add("young", 10), "candidate", "agentic-engineering", 0.9)
    repo.record_classified(add("old", 400), "candidate", "agentic-engineering", 0.9)
    repo.record_classified(add("noise", 10), "irrelevant", None, 0.0)
    vetted = repo.upsert_item(
        add("vetted", 100), Tier.VETTED_PREPRINT, "agentic-engineering", 0.9, Signals(), VEC
    )
    repo.upsert_item(
        add("peer", 100), Tier.PEER_REVIEWED, "agentic-engineering", 0.9, Signals(), VEC
    )
    feed = repo.record_classified(add("post", 900, "feed"), "candidate", "agentic-engineering", 0.9)
    poison = repo.record_classified(add("poison", 5), "candidate", "agentic-engineering", 0.9)
    for _ in range(MAX_ITEM_ATTEMPTS):
        repo.record_admit_failure(poison, "embedding input too long")
    old_new = repo.record_classified(add("refetched", 500), "candidate", "agentic-engineering", 0.9)
    targets = repo.recheck_targets(NOW, timedelta(days=365), also=[old_new, young])
    assert sorted(t.id for t in targets) == sorted([young, vetted, feed, old_new])
    assert db_session.get_one(Item, poison).last_error == "embedding input too long"
    # A human resolving the item's dead letter puts it back into the re-check.
    rid = repo.start_run(NOW, NOW, at=NOW)
    repo.add_dead_letter(rid, "arxiv", "poison", "gave up after 3 admission attempts")
    dl_id = db_session.scalars(select(DeadLetter.id).where(DeadLetter.ref == "poison")).one()
    repo.resolve_dead_letter(dl_id, at=NOW)
    assert poison in {t.id for t in repo.recheck_targets(NOW, timedelta(days=365))}


def test_recheck_follows_the_age_cadence(db_session: Session) -> None:
    # Keyless: daily up to 30 days after submission, weekly up to 90, monthly up to a year.
    repo = Repository(db_session)

    def add(eid: str, age_days: int, checked_days_ago: int | None) -> uuid.UUID:
        item = dataclasses.replace(
            ITEM, external_id=eid, published_at=NOW - timedelta(days=age_days)
        )
        item_id = repo.record_classified(item, "candidate", "agentic-engineering", 0.9)
        if checked_days_ago is not None:
            repo.mark_s2_checked([item_id], NOW - timedelta(days=checked_days_ago))
        return item_id

    daily = add("d10", 10, 1)
    weekly = add("d45", 45, 8)
    monthly = add("d200", 200, 31)
    never = add("never", 300, None)
    add("w45", 45, 3)  # weekly, checked 3 days ago: not yet due
    add("w200", 200, 20)  # monthly, checked 20 days ago: not yet due
    just = add("just", 10, 0)
    post = repo.record_classified(
        dataclasses.replace(ITEM, source="feed", external_id="p1"),
        "candidate",
        "agentic-engineering",
        0.9,
    )
    horizon = timedelta(days=365)
    due = {t.id for t in repo.recheck_targets(NOW, horizon)}
    assert due == {daily, weekly, monthly, never, post}
    assert just in {t.id for t in repo.recheck_targets(NOW, horizon, also=[just])}
    # The cap takes the most overdue papers (never checked, then the oldest check) and leaves
    # the rest due for the next call; feed candidates are not counted against it.
    first = repo.recheck_targets(NOW, horizon, paper_limit=2)
    assert {t.id for t in first} == {never, monthly, post}
    repo.mark_s2_checked([t.id for t in first if t.item.source == "arxiv"], NOW)
    second = repo.recheck_targets(NOW, horizon, paper_limit=2)
    assert {t.id for t in second} == {weekly, daily, post}


def test_recheck_never_due_past_the_last_cadence_band(db_session: Session) -> None:
    # A paper older than the last cadence band has no re-check interval, so it is never due even
    # when it was never checked. A horizon wider than that band must not pull it back in.
    repo = Repository(db_session)
    old = dataclasses.replace(ITEM, external_id="past-band", published_at=NOW - timedelta(days=400))
    old_id = repo.record_classified(old, "candidate", "agentic-engineering", 0.9)
    targets = repo.recheck_targets(NOW, timedelta(days=500))
    assert old_id not in {t.id for t in targets}


def test_cursor_advances_per_key(db_session: Session) -> None:
    repo = Repository(db_session)
    assert repo.cursor("arxiv") is None
    repo.advance_cursor("arxiv", datetime(2026, 10, 8, tzinfo=UTC))
    repo.advance_cursor("arxiv", datetime(2026, 10, 15, tzinfo=UTC))
    assert repo.cursor("arxiv") == datetime(2026, 10, 15, tzinfo=UTC)
    assert repo.cursor("feed:https://lab.example/feed.xml") is None


def test_finish_run_persists_status_counts_tokens_and_health(db_session: Session) -> None:
    repo = Repository(db_session)
    run_id = repo.start_run(NOW, NOW + timedelta(days=7), at=NOW)
    repo.finish_run(
        run_id,
        "ok",
        {"fetched": 3, "admitted": 1},
        1234,
        {"healthy": True, "problems": []},
        at=NOW,
    )
    row = db_session.get_one(Run, run_id)
    assert row.status == "ok"
    assert row.counts == {"fetched": 3, "admitted": 1}
    assert row.tokens == 1234
    assert row.health == {"healthy": True, "problems": []}
    assert row.finished_at is not None and row.finished_at.tzinfo is not None
    # A later run finished without a verdict stores SQL NULL, not the JSON value null.
    second = repo.start_run(NOW, NOW + timedelta(days=7), at=NOW)
    repo.finish_run(second, "partial", {}, 0, None, at=NOW)
    assert db_session.get_one(Run, second).health is None


def test_run_clock_stamps_run_and_resolved_letter(db_session: Session) -> None:
    repo = Repository(db_session)
    future = NOW + timedelta(days=90)  # far from the wall clock: the run's now, not the database's
    run_id = repo.start_run(NOW, NOW + timedelta(days=1), at=future)
    repo.add_dead_letter(run_id, "arxiv", "2510.9", "boom", payload={"external_id": "2510.9"})
    [(dl_id, _)] = repo.open_item_dead_letters()
    repo.resolve_dead_letter(dl_id, at=future)
    repo.finish_run(run_id, "succeeded", {}, 0, None, at=future)
    row = db_session.get_one(Run, run_id)
    assert row.started_at == future and row.finished_at == future
    assert db_session.get_one(DeadLetter, dl_id).resolved_at == future


def test_fail_open_runs_uses_the_run_clock(db_session: Session) -> None:
    repo = Repository(db_session)
    future = NOW + timedelta(days=90)
    # A row a hard kill left 'running' is closed with the run's now, not the database clock.
    run_id = repo.start_run(NOW, NOW, at=NOW)
    repo.fail_open_runs(at=future)
    assert db_session.get_one(Run, run_id).finished_at == future


def test_resolve_open_dead_letters_closes_only_source_letters(db_session: Session) -> None:
    repo = Repository(db_session)
    feed_url = "https://lab.example/feed.xml"
    later = NOW + timedelta(days=2)
    run_id = repo.start_run(NOW, NOW + timedelta(days=1), at=NOW)
    repo.add_dead_letter(run_id, "feed", feed_url, "HTTP 503")  # a source outage: no payload
    # An item letter that shares the outage's exact (source, ref) and differs only by payload.
    repo.add_dead_letter(run_id, "feed", feed_url, "boom", payload={"external_id": "x"})
    repo.add_dead_letter(run_id, "feed", "https://lab.example/p2", "no title")  # malformed
    repo.add_dead_letter(run_id, "feed", "p1", "boom", payload={"external_id": "p1"})  # item
    repo.resolve_open_dead_letters("feed", feed_url, at=later)
    open_null = set(
        db_session.scalars(
            select(DeadLetter.ref).where(
                DeadLetter.resolved_at.is_(None), DeadLetter.payload.is_(None)
            )
        )
    )
    open_payload = set(
        db_session.scalars(
            select(DeadLetter.ref).where(
                DeadLetter.resolved_at.is_(None), DeadLetter.payload.is_not(None)
            )
        )
    )
    # Only the outage letter closes; a payload letter for the same ref and the malformed-entry
    # letter (another ref) stay open, so the payload filter is load-bearing.
    assert open_null == {"https://lab.example/p2"}
    assert open_payload == {feed_url, "p1"}


def test_item_dead_letters_open_until_resolved(db_session: Session) -> None:
    repo = Repository(db_session)
    rid = repo.start_run(
        datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 8, tzinfo=UTC), at=NOW
    )
    repo.add_dead_letter(rid, "arxiv", "window", "HTTP 503")  # a source failure, no payload
    repo.add_dead_letter(rid, "arxiv", "2510.7", "boom", payload={"external_id": "2510.7"})
    [(dl_id, payload)] = repo.open_item_dead_letters()
    assert payload["external_id"] == "2510.7"
    assert repo.has_open_dead_letter("arxiv", "2510.7")
    repo.resolve_dead_letter(dl_id, at=NOW)
    assert repo.open_item_dead_letters() == []
    assert not repo.has_open_dead_letter("arxiv", "2510.7")


def test_dead_letter_retries_stop_at_the_attempt_cap(db_session: Session) -> None:
    repo = Repository(db_session)
    rid = repo.start_run(
        datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 8, tzinfo=UTC), at=NOW
    )
    repo.add_dead_letter(rid, "arxiv", "2510.8", "boom", payload={"external_id": "2510.8"})
    [(dl_id, _)] = repo.open_item_dead_letters()
    counts = [repo.record_retry_failure(dl_id, "boom again") for _ in range(MAX_ITEM_ATTEMPTS - 1)]
    assert counts == list(range(2, MAX_ITEM_ATTEMPTS + 1))
    # Given up: still open and visible, but no longer handed back for retry.
    assert repo.open_item_dead_letters() == []
    assert repo.has_open_dead_letter("arxiv", "2510.8")


def test_recheck_slack_absorbs_an_earlier_timer_start(db_session: Session) -> None:
    # The timer fires a few seconds earlier some mornings. A paper checked 24 h ago then must be
    # due again, or a whole day is skipped: the slack is what makes a daily paper due before its
    # full interval, and it is far too short to pull a weekly paper in early.
    repo = Repository(db_session)

    def add(eid: str, age_days: int, checked_ago: timedelta) -> uuid.UUID:
        item = dataclasses.replace(
            ITEM, external_id=eid, published_at=NOW - timedelta(days=age_days)
        )
        item_id = repo.record_classified(item, "candidate", "agentic-engineering", 0.9)
        repo.mark_s2_checked([item_id], NOW - checked_ago)
        return item_id

    boundary = timedelta(days=1) - S2_RECHECK_SLACK
    earlier_start = add("daily-earlier", 10, boundary)
    one_second_short = add("daily-short", 10, boundary - timedelta(seconds=1))
    still_recent = add("daily-recent", 10, timedelta(hours=21))
    weekly_recent = add("weekly-recent", 45, timedelta(days=6))
    due = {t.id for t in repo.recheck_targets(NOW, timedelta(days=365))}
    assert earlier_start in due  # exactly a day minus the slack: the boundary is due
    assert one_second_short not in due  # one second less and it waits for tomorrow
    assert still_recent not in due and weekly_recent not in due  # the slack is not a day
