"""unit tests for the pure suggestion ranking (Task 34)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from paperwatch.config import PROFILES_PATH, SUGGESTIONS_PATH
from paperwatch.domain import AdmissionBasis, Profile, Tier, load_profiles
from paperwatch.suggestions.config import WEIGHT_NAMES, Weights, load_suggestions
from paperwatch.suggestions.ranking import Candidate, Ranking, rank

NOW = datetime(2026, 11, 2, 9, tzinfo=UTC)
CFG = load_suggestions(SUGGESTIONS_PATH)
PROFILES: dict[str, Profile] = {p.name: p for p in load_profiles(PROFILES_PATH)}
DAY = timedelta(days=1)


def cand(**overrides: object) -> Candidate:
    # Sensible defaults for a pick-grade fresh paper; each test changes only what it is about.
    base: dict[str, object] = {
        "item_id": uuid4(),
        "source": "arxiv",
        "external_id": "2510.00001",
        "title": "A paper",
        "url": "https://arxiv.org/abs/2510.00001",
        "tier": Tier.PEER_REVIEWED,
        "basis": AdmissionBasis.VENUE,
        "profile": "agentic-engineering",
        "relevance": 0.93,
        "published_at": NOW - DAY,
        "authors": ("A. Person",),
        "org": None,
        "github_repo": None,
        "github_stars": None,
        "project_page": None,
        "text_chars": 2000,
        "hf_upvotes": None,
        "hf_listed_at": None,
        "upvote_series": (),
        "upvote_baseline": None,
        "promoted_to_peer_at": None,
        "repo_first_seen_at": None,
        "embedding": (),
    }
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


def lab_post(**overrides: object) -> Candidate:
    base: dict[str, object] = {
        "source": "feed",
        "external_id": "https://example.test/feed/1",
        "tier": Tier.LAB_BLOG,
        "basis": AdmissionBasis.FEED,
        # A lab post's author is the lab itself, which is how a recognisable lab is recognised.
        "authors": ("OpenAI",),
        "published_at": NOW - DAY,
    }
    base.update(overrides)
    return cand(**base)


def _rank(
    candidates: list[Candidate],
    *,
    config: object = CFG,
    excluded: frozenset[UUID] = frozenset(),
) -> Ranking:
    return rank(candidates, now=NOW, config=config, profiles=PROFILES, excluded=excluded)  # type: ignore[arg-type]


def _ids(ranking: Ranking) -> set[UUID]:
    return {scored.candidate.item_id for scored in (*ranking.picks, *ranking.rising)}


def test_only_items_inside_the_age_window_are_eligible() -> None:
    old_paper = cand(external_id="a", published_at=NOW - 22 * DAY)
    old_post = lab_post(external_id="b", published_at=NOW - 15 * DAY)
    fresh_paper = cand(external_id="c", published_at=NOW - 20 * DAY)
    fresh_post = lab_post(external_id="d", published_at=NOW - 13 * DAY)

    ranking = _rank([old_paper, old_post, fresh_paper, fresh_post])

    assert _ids(ranking) == {fresh_paper.item_id, fresh_post.item_id}


def test_excluded_items_never_appear() -> None:
    kept = cand(external_id="a")
    dismissed = cand(external_id="b")
    drafted = lab_post(external_id="c")

    ranking = _rank(
        [kept, dismissed, drafted], excluded=frozenset({dismissed.item_id, drafted.item_id})
    )

    assert _ids(ranking) == {kept.item_id}


def test_a_viral_lab_author_preprint_goes_to_rising_not_picks() -> None:
    viral = cand(
        external_id="viral",
        tier=Tier.VETTED_PREPRINT,
        basis=AdmissionBasis.LAB_AUTHOR,
        hf_upvotes=900,
        hf_listed_at=NOW - DAY,
    )
    reviewed = cand(external_id="reviewed")

    ranking = _rank([viral, reviewed])

    assert viral.item_id in {s.candidate.item_id for s in ranking.rising}
    assert viral.item_id not in {s.candidate.item_id for s in ranking.picks}


def test_recent_momentum_beats_old_raw_upvotes() -> None:
    # 400 upvotes that all arrived before the window: the baseline is the count the last observing
    # run saw, so nothing was gained inside the window and the paper has no momentum.
    old = cand(
        external_id="2510.00001",
        published_at=NOW - 20 * DAY,
        hf_upvotes=400,
        hf_listed_at=NOW - 20 * DAY,
        upvote_series=((NOW - 20 * DAY, 0), (NOW - 6 * DAY, 400)),
        upvote_baseline=(NOW - 5 * DAY, 400),
    )
    new = cand(
        external_id="2510.00002",
        published_at=NOW - 2 * DAY,
        hf_upvotes=60,
        hf_listed_at=NOW - 2 * DAY,
        upvote_series=((NOW - 2 * DAY, 0), (NOW - timedelta(hours=1), 60)),
        upvote_baseline=(NOW - 3 * DAY, 0),
    )

    ranking = _rank([old, new])

    assert ranking.picks[0].candidate.item_id == new.item_id
    assert ranking.picks[0].components["momentum"] > ranking.picks[1].components["momentum"]


def test_a_flat_paper_has_zero_momentum() -> None:
    # 200 upvotes in its first two days, then nothing for ten days. Snapshots are written only on
    # change, so the last observing run before the window saw the same 200: the rate is 0, not the
    # lifetime average of about 15 a day.
    flat = cand(
        external_id="flat",
        published_at=NOW - 12 * DAY,
        hf_upvotes=200,
        hf_listed_at=NOW - 12 * DAY,
        upvote_series=((NOW - 12 * DAY, 0), (NOW - 10 * DAY, 200)),
        upvote_baseline=(NOW - 8 * DAY, 200),
    )

    ranking = _rank([flat])

    assert ranking.picks[0].components["momentum"] == 0.0


def test_a_paper_first_listed_inside_the_window_uses_the_fallback() -> None:
    # No observing run saw either paper before its window, so the rate is upvotes per day since
    # listing plus one: 30 / 2 for one day, 30 / 6 for five.
    just_listed = cand(
        external_id="a",
        published_at=NOW - DAY,
        hf_upvotes=30,
        hf_listed_at=NOW - DAY,
    )
    listed_earlier = cand(
        external_id="b",
        published_at=NOW - 5 * DAY,
        hf_upvotes=30,
        hf_listed_at=NOW - 5 * DAY,
    )

    ranking = _rank([just_listed, listed_earlier])

    moments = {s.candidate.item_id: s.components["momentum"] for s in ranking.picks}
    assert moments[just_listed.item_id] > moments[listed_earlier.item_id]


def test_a_skipped_listing_run_widens_the_baseline_gap() -> None:
    # Both papers gained 100 upvotes, but one baseline comes from a run five days ago because the
    # runs in between never read the listing: dividing by the observed gap, not by momentum_days,
    # is what keeps that gap from being credited as speed. Under the wrong rule the two tie and
    # the external_id breaks the tie, putting "aaa" first.
    skipped = cand(
        external_id="aaa",
        published_at=NOW - 8 * DAY,
        hf_upvotes=200,
        upvote_series=((NOW - 8 * DAY, 100), (NOW, 200)),
        upvote_baseline=(NOW - 5 * DAY, 100),
    )
    tight = cand(
        external_id="zzz",
        published_at=NOW - 8 * DAY,
        hf_upvotes=200,
        upvote_series=((NOW - 8 * DAY, 100), (NOW, 200)),
        upvote_baseline=(NOW - 3 * DAY, 100),
    )

    ranking = _rank([skipped, tight])

    assert ranking.picks[0].candidate.item_id == tight.item_id


def test_peer_review_promotion_adds_the_event_bonus() -> None:
    promoted = cand(
        external_id="a",
        promoted_to_peer_at=NOW - 2 * DAY,
        repo_first_seen_at=NOW - 2 * DAY,
    )
    stale = cand(
        external_id="b",
        promoted_to_peer_at=NOW - 30 * DAY,
        repo_first_seen_at=NOW - 30 * DAY,
    )

    ranking = _rank([promoted, stale])

    by_id = {s.candidate.item_id: s for s in ranking.picks}
    assert by_id[promoted.item_id].components["event"] == pytest.approx(0.15)
    assert by_id[stale.item_id].components["event"] == 0.0
    assert by_id[promoted.item_id].score > by_id[stale.item_id].score


def test_near_duplicates_merge_into_one_card_with_the_paper_first() -> None:
    # The lab post is fresher, so it scores higher, but the card a reader opens is the paper.
    post = lab_post(
        external_id="post", published_at=NOW - timedelta(hours=2), embedding=(1.0, 0.0, 0.0)
    )
    paper = cand(external_id="paper", embedding=(1.0, 0.0, 0.0))

    ranking = _rank([post, paper])

    assert len(ranking.picks) == 1
    merged = ranking.picks[0]
    assert merged.candidate.item_id == paper.item_id
    assert [c.item_id for c in merged.also] == [post.item_id]


def test_profile_quota_keeps_at_most_three_picks_from_one_profile() -> None:
    crowd = [
        cand(external_id=f"a{i}", profile="agentic-engineering", relevance=0.95 - i * 0.01)
        for i in range(5)
    ]
    others = [
        cand(external_id=f"b{i}", profile="ai-forensics", relevance=0.95 - i * 0.01)
        for i in range(2)
    ]

    ranking = _rank([*crowd, *others])

    assert len(ranking.picks) == 5
    profiles = [s.candidate.profile for s in ranking.picks]
    assert profiles.count("agentic-engineering") == 3
    assert profiles.count("ai-forensics") == 2


def test_a_pick_grade_lab_post_is_always_among_the_picks() -> None:
    crowd = [
        cand(external_id=f"a{i}", profile="agentic-engineering", relevance=0.95 - i * 0.01)
        for i in range(3)
    ] + [
        cand(external_id=f"b{i}", profile="ai-forensics", relevance=0.95 - i * 0.01)
        for i in range(3)
    ]
    # A recognised lab's post, deliberately the weakest item in the pool, on a third profile so
    # the replacement cannot break a quota.
    post = lab_post(
        external_id="post",
        profile="ml-architectures",
        relevance=0.51,
        published_at=NOW - 10 * DAY,
    )

    ranking = _rank([*crowd, post])

    assert post.item_id in {s.candidate.item_id for s in ranking.picks}
    assert len(ranking.picks) == 5
    assert len({s.candidate.item_id for s in ranking.picks}) == 5


def test_short_lab_posts_are_never_suggested() -> None:
    too_short = lab_post(external_id="short", text_chars=399)
    long_enough = lab_post(external_id="long", text_chars=400)

    ranking = _rank([too_short, long_enough])

    assert too_short.item_id not in _ids(ranking)
    assert long_enough.item_id in _ids(ranking)


def test_components_sum_to_the_score() -> None:
    scored = cand(
        external_id="a",
        hf_upvotes=12,
        upvote_baseline=(NOW - 3 * DAY, 0),
        github_repo="https://github.com/example/repo",
        github_stars=5000,
        promoted_to_peer_at=NOW - DAY,
    )

    ranking = _rank([scored])
    card = ranking.picks[0]

    weighted = sum(
        getattr(CFG.weights, name) * card.components[name]
        for name in WEIGHT_NAMES
        if name != "credibility"
    )
    weighted += CFG.weights.credibility * card.components["credibility"]
    assert card.score == pytest.approx(weighted + card.components["event"])
    assert set(card.components) == {*WEIGHT_NAMES, "event"}


def test_changing_a_weight_changes_the_order() -> None:
    old_viral = cand(
        external_id="a",
        published_at=NOW - 12 * DAY,
        hf_upvotes=500,
        upvote_series=((NOW - 2 * DAY, 450), (NOW, 500)),
        upvote_baseline=(NOW - 3 * DAY, 450),
    )
    fresh = cand(external_id="b", published_at=NOW - timedelta(hours=6), hf_upvotes=3)

    default = _rank([old_viral, fresh])
    # Recency alone: the fresh paper wins, which the default weights do not allow.
    recency_only = replace(
        CFG,
        weights=Weights(
            momentum=0.0, recency=1.0, shareability=0.0, topic_fit=0.0, credibility=0.0
        ),
    )
    changed = _rank([old_viral, fresh], config=recency_only)

    assert default.picks[0].candidate.item_id == old_viral.item_id
    assert changed.picks[0].candidate.item_id == fresh.item_id


def test_equal_scores_order_by_source_and_external_id() -> None:
    b = cand(external_id="b")
    a = cand(external_id="a")
    other_source = cand(source="biorxiv", external_id="a")

    ranking = _rank([b, a, other_source])

    assert [s.candidate.external_id for s in ranking.picks] == ["a", "b", "a"]
    assert [s.candidate.source for s in ranking.picks] == ["arxiv", "arxiv", "biorxiv"]


def test_a_well_known_organisation_earns_the_shareability_credit() -> None:
    # The organisation is Hugging Face's display name, and its capitalisation varies, so the match
    # ignores case; a lab outside the list earns nothing for its name.
    known = cand(external_id="a", org="nvidia")
    unknown = cand(external_id="b", org="MirroS")
    known_by_author = cand(external_id="c", authors=("Hugging Face",))

    ranking = _rank([known, unknown, known_by_author])

    shares = {
        scored.candidate.external_id: scored.components["shareability"] for scored in ranking.picks
    }
    assert shares["a"] == pytest.approx(0.4)
    assert shares["c"] == pytest.approx(0.4)
    assert shares["b"] == 0.0
