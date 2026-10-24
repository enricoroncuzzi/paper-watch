"""The pure suggestion ranking: gate, score, merge near-duplicates, diversify (Task 34).

No clock, no I/O and no randomness: `now` and the config come in as arguments, so a ranking can be
replayed from stored data to see why the list looked the way it did. The pipeline is the one the
design describes: reliability is a gate, attention only reorders what passed it, and the six
component values are kept on every card so the score stays explainable.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import log10, sqrt
from typing import Literal
from uuid import UUID

from paperwatch.domain import AdmissionBasis, Profile, Tier
from paperwatch.suggestions.config import WEIGHT_NAMES, SuggestionsConfig

# Lab posts arrive as source "feed" (the tier-3 clients), and they age and are measured differently.
LAB_POST_SOURCE = "feed"
SECONDS_PER_DAY = 86_400.0
# A gain divided by almost no time is not a rate: the floor keeps a baseline observed in the same
# minute finite. A real run is never this close to the next one.
MIN_RATE_DAYS = 1.0 / 24
# Lab posts carry no upvote series, so they get a flat momentum: a lab announcing its own work is
# neither a hype signal nor a dead one.
LAB_POST_MOMENTUM = 0.5


@dataclass(frozen=True)
class Candidate:
    """One admitted item with every feature the ranking reads, gathered by the store (Task 36)."""

    item_id: UUID
    source: str
    external_id: str
    title: str
    url: str
    tier: Tier
    basis: AdmissionBasis
    profile: str | None
    relevance: float
    published_at: datetime
    authors: tuple[str, ...]
    org: str | None
    github_repo: str | None
    github_stars: int | None
    project_page: str | None
    text_chars: int
    hf_upvotes: int | None
    hf_listed_at: datetime | None
    # (observed_at, upvotes), oldest first, as the append-only signal_snapshots rows record them.
    upvote_series: tuple[tuple[datetime, int], ...]
    # The upvote count the last run that read the Hugging Face listing saw at or before the window
    # start. None means no such run saw the paper, so it was first listed inside the window.
    upvote_baseline: tuple[datetime, int] | None
    promoted_to_peer_at: datetime | None
    repo_first_seen_at: datetime | None
    # Empty means "no vector": similarity to it is then 0, which keeps eval and test data usable.
    embedding: tuple[float, ...]
    # The facts a card prints beside the basis label. Defaults keep eval and test fixtures short.
    venue: str | None = None
    citation_count: int = 0


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    score: float
    components: Mapping[str, float]
    lane: Literal["pick", "rising"]
    # The near-duplicates folded into this card, shown as "also reported by".
    also: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class Ranking:
    picks: tuple[Scored, ...]
    rising: tuple[Scored, ...]


def credibility(tier: Tier, basis: AdmissionBasis, config: SuggestionsConfig) -> float:
    # Only evidence a reader can check earns the top of the scale; a lab's own claim the least.
    levels = config.credibility
    if tier is Tier.PEER_REVIEWED:
        return levels.peer_reviewed
    if tier is Tier.LAB_BLOG:
        return levels.lab_blog
    if basis in (AdmissionBasis.HF_UPVOTES, AdmissionBasis.CITATIONS):
        return levels.vetted_by_upvotes_or_citations
    return levels.vetted_by_lab_author


def is_pick_grade(candidate: Candidate, config: SuggestionsConfig) -> bool:
    """Whether an item may take one of the picks, not only a place on the Rising shelf.

    Task 38 reuses this for the "early work" notice on a draft, so it is public.
    """
    if credibility(candidate.tier, candidate.basis, config) >= config.pick_min_credibility:
        return True
    # A lab post is the lab's own claim, so it only earns a pick when the lab is one a reader of
    # this corpus recognises; the signed-in page still shows it, it is not a public claim.
    return candidate.tier is Tier.LAB_BLOG and _recognised(candidate, config)


def rank(
    candidates: Sequence[Candidate],
    *,
    now: datetime,
    config: SuggestionsConfig,
    profiles: Mapping[str, Profile],
    excluded: frozenset[UUID],
) -> Ranking:
    eligible = [c for c in candidates if _eligible(c, now, config, excluded)]
    # Momentum is a rank percentile inside the eligible paper pool, so the pool is measured before
    # any single candidate's components are.
    rates = {c.item_id: _rate(c, now) for c in eligible if c.source != LAB_POST_SOURCE}
    pool = tuple(rates.values())
    pick_pool: list[Scored] = []
    rising_pool: list[Scored] = []
    for candidate in eligible:
        components = _components(candidate, now, config, profiles, rates, pool)
        score = sum(getattr(config.weights, name) * components[name] for name in WEIGHT_NAMES)
        score += components["event"]
        lane: Literal["pick", "rising"] = "pick" if is_pick_grade(candidate, config) else "rising"
        scored = Scored(candidate=candidate, score=score, components=components, lane=lane)
        (pick_pool if lane == "pick" else rising_pool).append(scored)
    return Ranking(picks=_picks(pick_pool, config), rising=_rising(rising_pool, config))


def _eligible(
    candidate: Candidate, now: datetime, config: SuggestionsConfig, excluded: frozenset[UUID]
) -> bool:
    if candidate.item_id in excluded:
        return False
    if candidate.text_chars < config.min_text_chars:
        return False
    window = (
        config.window_days.lab_post
        if candidate.source == LAB_POST_SOURCE
        else config.window_days.paper
    )
    return now - candidate.published_at <= timedelta(days=window)


def _rate(candidate: Candidate, now: datetime) -> float:
    # Upvotes gained per day over the momentum window. Measured from the observed baseline, never
    # from the item's whole life: snapshots are written only on change, so a paper that stopped
    # gaining upvotes must score 0 instead of keeping its early average.
    if candidate.source == LAB_POST_SOURCE:
        return LAB_POST_MOMENTUM
    current = candidate.upvote_series[-1][1] if candidate.upvote_series else candidate.hf_upvotes
    if current is None:
        return 0.0
    if candidate.upvote_baseline is not None:
        baseline_at, baseline_value = candidate.upvote_baseline
        days = max((now - baseline_at).total_seconds() / SECONDS_PER_DAY, MIN_RATE_DAYS)
        return max(0.0, (current - baseline_value) / days)
    # No observing run saw the paper at or before the window start, so it was first listed inside
    # the window and its whole life is the window.
    listed_at = candidate.hf_listed_at or candidate.published_at
    days = max((now - listed_at).total_seconds() / SECONDS_PER_DAY, 0.0) + 1
    return max(0.0, current / days)


def _percentile(value: float, pool: tuple[float, ...]) -> float:
    # The share of the pool this item outranks. A rate of zero earns nothing whatever the rest of
    # the pool does: a flat paper must not look fast because every other paper is flat too.
    if value <= 0:
        return 0.0
    if len(pool) < 2:
        return 1.0
    return sum(1 for other in pool if other < value) / (len(pool) - 1)


def _components(
    candidate: Candidate,
    now: datetime,
    config: SuggestionsConfig,
    profiles: Mapping[str, Profile],
    rates: Mapping[UUID, float],
    pool: tuple[float, ...],
) -> dict[str, float]:
    is_post = candidate.source == LAB_POST_SOURCE
    momentum = LAB_POST_MOMENTUM if is_post else _percentile(rates[candidate.item_id], pool)
    age = max(0.0, (now - candidate.published_at).total_seconds() / SECONDS_PER_DAY)
    half_life = config.half_life_days.lab_post if is_post else config.half_life_days.paper
    return {
        "momentum": momentum,
        "recency": 0.5 ** (age / half_life),
        "shareability": _shareability(candidate, config),
        "topic_fit": _topic_fit(candidate, profiles),
        "credibility": credibility(candidate.tier, candidate.basis, config),
        "event": _event_bonus(candidate, now, config),
    }


def _recognised(candidate: Candidate, config: SuggestionsConfig) -> bool:
    names = config.recognisable_names
    if candidate.org is not None and candidate.org in names:
        return True
    return any(author in names for author in candidate.authors)


def _shareability(candidate: Candidate, config: SuggestionsConfig) -> float:
    # Code or a demo a reader can try and a lab they recognise matter more than raw stars, and the
    # star term saturates so a hundred thousand stars is not a different league from ten thousand.
    code = 1.0 if candidate.github_repo or candidate.project_page else 0.0
    stars = candidate.github_stars or 0
    star_score = min(1.0, log10(1 + stars) / log10(1 + config.stars_full_at))
    return 0.4 * code + 0.4 * (1.0 if _recognised(candidate, config) else 0.0) + 0.2 * star_score


def _topic_fit(candidate: Candidate, profiles: Mapping[str, Profile]) -> float:
    # How far the item cleared its profile's calibrated threshold, scaled by the profile's editorial
    # weight, so a strong match in a secondary topic can still outrank a weak one in the main topic.
    profile = profiles.get(candidate.profile) if candidate.profile else None
    if profile is None:
        return 0.0
    span = 1.0 - profile.relevance_threshold
    if span <= 0:
        return 0.0
    cleared = (candidate.relevance - profile.relevance_threshold) / span
    return min(1.0, max(0.0, cleared)) * profile.weight


def _event_bonus(candidate: Candidate, now: datetime, config: SuggestionsConfig) -> float:
    bonus = config.event_bonus
    total = 0.0
    if _within(candidate.promoted_to_peer_at, now, bonus.promotion_window_days):
        total += bonus.promoted_to_peer_reviewed
    if _within(candidate.repo_first_seen_at, now, bonus.repo_window_days):
        total += bonus.repo_appeared
    return total


def _within(at: datetime | None, now: datetime, days: int) -> bool:
    return at is not None and timedelta(0) <= now - at <= timedelta(days=days)


def _ordered(pool: Sequence[Scored]) -> list[Scored]:
    # Ties break by (source, external_id), never by id, so a replay on another database orders the
    # same way and the stored order is reproducible.
    return sorted(pool, key=lambda s: (-s.score, s.candidate.source, s.candidate.external_id))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _merge(pool: Sequence[Scored], config: SuggestionsConfig) -> list[Scored]:
    # Two write-ups of the same work are one card, and the paper is the card a reader opens.
    groups: list[list[Scored]] = []
    for scored in pool:
        for group in groups:
            if _cosine(group[0].candidate.embedding, scored.candidate.embedding) >= (
                config.duplicate_cosine
            ):
                group.append(scored)
                break
        else:
            groups.append([scored])
    merged = []
    for group in groups:
        paper = next((s for s in group if s.candidate.source != LAB_POST_SOURCE), None)
        primary = paper or group[0]
        others = tuple(s.candidate for s in group if s is not primary)
        merged.append(replace(primary, also=others))
    return _ordered(merged)


def _mmr(pool: Sequence[Scored], config: SuggestionsConfig) -> list[Scored]:
    remaining = _ordered(pool)
    if not remaining:
        return []
    # The two terms are added, so the score is put on the same 0..1 scale as the similarity.
    top = max(scored.score for scored in remaining) or 1.0
    chosen = [remaining.pop(0)]
    while remaining:
        best_index = 0
        best_value = float("-inf")
        for index, scored in enumerate(remaining):
            redundancy = max(
                (_cosine(scored.candidate.embedding, pick.candidate.embedding) for pick in chosen),
                default=0.0,
            )
            value = config.mmr_lambda * (scored.score / top) - (1 - config.mmr_lambda) * redundancy
            if value > best_value:
                best_value = value
                best_index = index
        chosen.append(remaining.pop(best_index))
    return chosen


def _picks(pool: Sequence[Scored], config: SuggestionsConfig) -> tuple[Scored, ...]:
    merged = _merge(_ordered(pool)[: config.mmr_pool], config)
    picks: list[Scored] = []
    per_profile: Counter[str] = Counter()
    for scored in _mmr(merged, config):
        if len(picks) == config.picks:
            break
        profile = scored.candidate.profile
        if profile is not None and per_profile[profile] >= config.max_picks_per_profile:
            continue
        if profile is not None:
            per_profile[profile] += 1
        picks.append(scored)
    return tuple(_with_a_lab_post(picks, merged, config))


def _with_a_lab_post(
    picks: list[Scored], merged: Sequence[Scored], config: SuggestionsConfig
) -> list[Scored]:
    # A lab's own announcement is what makes this list feel alive, so one pick-grade lab post
    # always appears, in the lowest pick's slot.
    if not picks or any(p.candidate.tier is Tier.LAB_BLOG for p in picks):
        return picks
    posts = _ordered(
        [
            s
            for s in merged
            if s.candidate.tier is Tier.LAB_BLOG and is_pick_grade(s.candidate, config)
        ]
    )
    if not posts:
        return picks
    lowest = min(range(len(picks)), key=lambda i: picks[i].score)
    result = list(picks)
    result[lowest] = posts[0]
    return result


def _rising(pool: Sequence[Scored], config: SuggestionsConfig) -> tuple[Scored, ...]:
    merged = _merge(_ordered(pool)[: config.mmr_pool], config)
    return tuple(_ordered(merged)[: config.rising])
