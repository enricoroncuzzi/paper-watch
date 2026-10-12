"""Explicit, testable reliability-tier admission rules (design section 3.1)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from paperwatch.domain import Classification, Signals, SourceItem, Tier


@dataclass(frozen=True)
class AdmissionConfig:
    # Thresholds and venue/author lists, loaded from config/admission.yaml.
    min_relevance: float
    accepted_venues: frozenset[str]
    min_hf_upvotes: int
    min_citations: int
    lab_authors: frozenset[str]


# A blank venue or author string would silently never match, so require non-blank.
_NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _AdmissionConfigFile(BaseModel):
    # extra="forbid" turns a typo'd key into a clear error instead of a silent ignore.
    model_config = ConfigDict(extra="forbid")

    min_relevance: float = Field(ge=0, le=1)
    # strict=True rejects booleans, which pydantic would otherwise coerce to 0/1.
    min_hf_upvotes: int = Field(strict=True, ge=1)
    min_citations: int = Field(strict=True, ge=1)
    accepted_venues: list[_NonBlank] = Field(min_length=1)
    lab_authors: list[_NonBlank] = Field(min_length=1)


def load_admission(path: Path) -> AdmissionConfig:
    # Thresholds are configuration, so the file is validated like the feed list.
    try:
        config = _AdmissionConfigFile.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as exc:
        raise ValueError(f"invalid admission config in {path}: {exc}") from exc
    return AdmissionConfig(
        min_relevance=config.min_relevance,
        accepted_venues=frozenset(config.accepted_venues),
        min_hf_upvotes=config.min_hf_upvotes,
        min_citations=config.min_citations,
        lab_authors=frozenset(config.lab_authors),
    )


_YEAR = re.compile(r"^(19|20)\d{2}$")
_WORKSHOP = re.compile(r"\bworkshops?\b", re.IGNORECASE)


def _venue_accepted(signals: Signals, cfg: AdmissionConfig) -> bool:
    # Exact acronym, or "<acronym> <4-digit year>". A workshop title is not the
    # parent conference, even when one of its aliases is the bare acronym.
    names = [name for name in (signals.venue, *signals.venue_aliases) if name]
    if any(_WORKSHOP.search(name) for name in names):
        return False
    for name in names:
        if name in cfg.accepted_venues:
            return True
        parts = name.split()
        if len(parts) == 2 and parts[0] in cfg.accepted_venues and _YEAR.fullmatch(parts[1]):
            return True
    return False


def admit(
    item: SourceItem, cls: Classification, signals: Signals, cfg: AdmissionConfig
) -> Tier | None:
    # Relevance gates everything: an unprofiled or low-score item is not stored.
    if cls.profile is None or cls.relevance < cfg.min_relevance:
        return None
    if item.source == "feed":
        # Feed items are tier 3 by construction, whatever paper signals they carry.
        return Tier.LAB_BLOG
    if _venue_accepted(signals, cfg):
        return Tier.PEER_REVIEWED
    if (signals.hf_upvotes or 0) >= cfg.min_hf_upvotes:
        return Tier.VETTED_PREPRINT
    if signals.citation_count >= cfg.min_citations:
        return Tier.VETTED_PREPRINT
    if cfg.lab_authors.intersection(item.authors):
        return Tier.VETTED_PREPRINT
    return None


# Order of paper tiers; lab_blog belongs to feed items only and is never ranked against them.
_PAPER_RANK = {Tier.VETTED_PREPRINT: 1, Tier.PEER_REVIEWED: 2}


def promote(current: Tier | None, proposed: Tier | None) -> Tier | None:
    # Evidence can lag or vanish for a day (an outage), so the stored tier is a floor.
    if proposed is None or proposed == current:
        return current
    if current is None:
        return proposed
    if current in _PAPER_RANK and proposed in _PAPER_RANK:
        return proposed if _PAPER_RANK[proposed] > _PAPER_RANK[current] else current
    return current
