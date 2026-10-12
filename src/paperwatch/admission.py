"""Explicit, testable reliability-tier admission rules (design section 3.1)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from paperwatch.domain import Classification, Signals, SourceItem, Tier


@dataclass(frozen=True)
class AdmissionConfig:
    # Thresholds and venue/author lists, loaded from config/admission.yaml.
    min_relevance: float
    accepted_venues: frozenset[str]
    min_hf_upvotes: int
    min_citations: int
    lab_authors: frozenset[str]


def load_admission(path: Path) -> AdmissionConfig:
    # Thresholds are configuration, not code, so a policy change is a data change.
    d = yaml.safe_load(path.read_text())
    return AdmissionConfig(
        float(d["min_relevance"]),
        frozenset(d["accepted_venues"]),
        int(d["min_hf_upvotes"]),
        int(d["min_citations"]),
        frozenset(d["lab_authors"]),
    )


def _venue_accepted(signals: Signals, cfg: AdmissionConfig) -> bool:
    # Accept a venue name, its acronym aliases, or the first word of a year-suffixed name.
    for name in (signals.venue, *signals.venue_aliases):
        if name and (name in cfg.accepted_venues or name.split()[0] in cfg.accepted_venues):
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
