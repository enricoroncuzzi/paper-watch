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


def load_profiles(path: Path) -> list[Profile]:
    # Profiles are configuration, not code, so they load at startup from YAML.
    data = yaml.safe_load(path.read_text())
    return [
        Profile(p["name"], float(p["weight"]), tuple(p["categories"]), tuple(p["terms"]))
        for p in data["profiles"]
    ]
