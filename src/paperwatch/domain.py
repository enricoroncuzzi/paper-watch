"""Core domain types shared by every source, the store, and retrieval.

Nothing here depends on a framework or a database: these are plain frozen
dataclasses plus the tier enum and the topic-profile loader, so the same types
move unchanged from a source client into the store and back out of retrieval.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml


class Tier(StrEnum):
    # The three reliability tiers from the design spec (section 3.1).
    PEER_REVIEWED = "peer_reviewed"
    VETTED_PREPRINT = "vetted_preprint"
    LAB_BLOG = "lab_blog"


class AdmissionBasis(StrEnum):
    # Which admission rule produced an item's current tier, so a lab's own report stays
    # distinguishable from community-vetted work (captain decision 2026-10-13).
    VENUE = "venue"
    HF_UPVOTES = "hf_upvotes"
    CITATIONS = "citations"
    LAB_AUTHOR = "lab_author"
    FEED = "feed"
    # SEED is written only by the test and eval seeding helper, never by the pipeline.
    SEED = "seed"


@dataclass(frozen=True)
class SourceItem:
    """One normalised record as returned by any source client, before admission."""

    source: str
    external_id: str
    title: str
    authors: tuple[str, ...]
    text: str
    url: str
    published_at: datetime
    categories: tuple[str, ...] = ()
    # Licence of the stored text, not the PDF: arXiv metadata/abstracts are CC0.
    licence: str = "unknown"


@dataclass(frozen=True, kw_only=True)
class Signals:
    """Quality evidence used to admit or promote a preprint (keyword-only)."""

    venue: str | None = None
    citation_count: int = 0
    hf_upvotes: int | None = None
    # Semantic Scholar's publicationVenue.name plus alternate_names, e.g. "ICLR".
    venue_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Classification:
    """Result of the LLM relevance classifier: which profile and how strongly."""

    profile: str | None
    relevance: float
    reason: str


@dataclass(frozen=True)
class RejectedEntry:
    """A source entry that could not be parsed; Task 9 records it as a dead letter."""

    source: str
    ref: str
    error: str


@dataclass(frozen=True)
class Profile:
    """A topic profile read from config/profiles.yaml."""

    name: str
    weight: float
    categories: tuple[str, ...]
    terms: tuple[str, ...]
    # The text embedded as the profile's vector, and the cosine similarity at or above which an
    # item matches it; read by the embedding classifier, calibrated together for one model.
    description: str
    relevance_threshold: float


def _threshold(name: object, value: object, path: Path) -> float:
    # bool is an int subclass, so True would pass as 1.0 without the first check. Both ends are
    # excluded: 0 would match everything, and 1 only an identical text.
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise ValueError(
            f"profile {name!r} in {path}: relevance_threshold must be a number strictly "
            f"between 0 and 1, got {value!r}"
        )
    return float(value)


def load_profiles(path: Path) -> list[Profile]:
    # Profiles are configuration, not code, so they load at startup from YAML. A missing
    # description or threshold is refused here: either would make the embedding classifier
    # admit everything or nothing without any error.
    data = yaml.safe_load(path.read_text())
    profiles = []
    for p in data["profiles"]:
        description = p.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"profile {p['name']!r} in {path} needs a non-blank description")
        profiles.append(
            Profile(
                p["name"],
                float(p["weight"]),
                tuple(p["categories"]),
                tuple(p["terms"]),
                description.strip(),
                _threshold(p["name"], p.get("relevance_threshold"), path),
            )
        )
    return profiles
