"""The suggestion settings, read from config/suggestions.yaml (Task 34).

The ranking function is pure, so everything it may vary lives here: a tweak to what the list
prefers is a reviewed edit to a file, never a code change, and a card's own recorded components
are enough to replay why it ranked where it did.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

# The five scored components, in the order a card prints them. Their weights must sum to one, so
# the recorded components always add up to the score they explain.
WEIGHT_NAMES = ("momentum", "recency", "shareability", "topic_fit", "credibility")
# How far the weights may miss 1.0, for YAML's binary floats (0.30 + 0.25 + 0.20 + 0.15 + 0.10).
WEIGHT_SUM_TOLERANCE = 0.001


@dataclass(frozen=True)
class Weights:
    momentum: float
    recency: float
    shareability: float
    topic_fit: float
    credibility: float


@dataclass(frozen=True)
class HalfLifeDays:
    paper: float
    lab_post: float


@dataclass(frozen=True)
class WindowDays:
    paper: int
    lab_post: int


@dataclass(frozen=True)
class EventBonus:
    promoted_to_peer_reviewed: float
    promotion_window_days: int
    repo_appeared: float
    repo_window_days: int


@dataclass(frozen=True)
class CredibilityLevels:
    peer_reviewed: float
    vetted_by_upvotes_or_citations: float
    lab_blog: float
    vetted_by_lab_author: float


@dataclass(frozen=True)
class SuggestionsConfig:
    """Mirrors config/suggestions.yaml; every field is read by `ranking.rank`."""

    weights: Weights
    half_life_days: HalfLifeDays
    window_days: WindowDays
    momentum_days: int
    event_bonus: EventBonus
    credibility: CredibilityLevels
    picks: int
    rising: int
    pick_min_credibility: float
    min_text_chars: int
    mmr_lambda: float
    mmr_pool: int
    duplicate_cosine: float
    max_picks_per_profile: int
    stars_full_at: int
    # A presentation choice, not a reliability rule: the labs a reader recognises.
    recognisable_names: frozenset[str]


# A blank name would silently never match, so require a non-blank one.
_NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    momentum: float = Field(strict=True, ge=0, le=1)
    recency: float = Field(strict=True, ge=0, le=1)
    shareability: float = Field(strict=True, ge=0, le=1)
    topic_fit: float = Field(strict=True, ge=0, le=1)
    credibility: float = Field(strict=True, ge=0, le=1)


class _HalfLifeDays(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: float = Field(strict=True, gt=0)
    lab_post: float = Field(strict=True, gt=0)


class _WindowDays(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: int = Field(strict=True, ge=1)
    lab_post: int = Field(strict=True, ge=1)


class _EventBonus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promoted_to_peer_reviewed: float = Field(strict=True, ge=0, le=1)
    promotion_window_days: int = Field(strict=True, ge=1)
    repo_appeared: float = Field(strict=True, ge=0, le=1)
    repo_window_days: int = Field(strict=True, ge=1)


class _Credibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peer_reviewed: float = Field(strict=True, ge=0, le=1)
    vetted_by_upvotes_or_citations: float = Field(strict=True, ge=0, le=1)
    lab_blog: float = Field(strict=True, ge=0, le=1)
    vetted_by_lab_author: float = Field(strict=True, ge=0, le=1)


class _SuggestionsConfigFile(BaseModel):
    # extra="forbid" turns a typo'd key into a clear error instead of a silent ignore.
    model_config = ConfigDict(extra="forbid")

    weights: _Weights
    half_life_days: _HalfLifeDays
    window_days: _WindowDays
    momentum_days: int = Field(strict=True, ge=1)
    event_bonus: _EventBonus
    credibility: _Credibility
    picks: int = Field(strict=True, ge=1)
    rising: int = Field(strict=True, ge=1)
    pick_min_credibility: float = Field(strict=True, ge=0, le=1)
    min_text_chars: int = Field(strict=True, ge=1)
    mmr_lambda: float = Field(strict=True, ge=0, le=1)
    mmr_pool: int = Field(strict=True, ge=1)
    duplicate_cosine: float = Field(strict=True, gt=0, le=1)
    max_picks_per_profile: int = Field(strict=True, ge=1)
    stars_full_at: int = Field(strict=True, ge=1)
    recognisable_names: list[_NonBlank] = Field(min_length=1)


def load_suggestions(path: Path) -> SuggestionsConfig:
    # Ranked suggestions are configuration, so the file is validated like the admission rules and
    # every error names the path that has to be fixed.
    try:
        config = _SuggestionsConfigFile.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as exc:
        raise ValueError(f"invalid suggestions config in {path}: {exc}") from exc
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        # Bad YAML, a missing file, or a non-UTF-8 file also names the path rather than failing
        # opaquely (UnicodeDecodeError is a ValueError, not an OSError).
        raise ValueError(f"invalid suggestions config in {path}: {exc}") from exc
    weights = Weights(
        momentum=config.weights.momentum,
        recency=config.weights.recency,
        shareability=config.weights.shareability,
        topic_fit=config.weights.topic_fit,
        credibility=config.weights.credibility,
    )
    total = sum(getattr(weights, name) for name in WEIGHT_NAMES)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        # A score is the weighted sum its components explain, so weights that miss 1.0 would make
        # the recorded breakdown disagree with the ranking.
        raise ValueError(
            f"invalid suggestions config in {path}: the weights must sum to 1.0, got {total!r}"
        )
    return SuggestionsConfig(
        weights=weights,
        half_life_days=HalfLifeDays(
            paper=config.half_life_days.paper, lab_post=config.half_life_days.lab_post
        ),
        window_days=WindowDays(
            paper=config.window_days.paper, lab_post=config.window_days.lab_post
        ),
        momentum_days=config.momentum_days,
        event_bonus=EventBonus(
            promoted_to_peer_reviewed=config.event_bonus.promoted_to_peer_reviewed,
            promotion_window_days=config.event_bonus.promotion_window_days,
            repo_appeared=config.event_bonus.repo_appeared,
            repo_window_days=config.event_bonus.repo_window_days,
        ),
        credibility=CredibilityLevels(
            peer_reviewed=config.credibility.peer_reviewed,
            vetted_by_upvotes_or_citations=config.credibility.vetted_by_upvotes_or_citations,
            lab_blog=config.credibility.lab_blog,
            vetted_by_lab_author=config.credibility.vetted_by_lab_author,
        ),
        picks=config.picks,
        rising=config.rising,
        pick_min_credibility=config.pick_min_credibility,
        min_text_chars=config.min_text_chars,
        mmr_lambda=config.mmr_lambda,
        mmr_pool=config.mmr_pool,
        duplicate_cosine=config.duplicate_cosine,
        max_picks_per_profile=config.max_picks_per_profile,
        stars_full_at=config.stars_full_at,
        recognisable_names=frozenset(config.recognisable_names),
    )
