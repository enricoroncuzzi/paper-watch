"""Per-answer source aliases: the model may only cite what a tool put in this answer's registry.

The registry is created for one answer and thrown away with it. A tool registers an item and gets
back a short alias (`S1`, `S2`, ... in first-seen order); the model then refers to sources by that
alias. An alias the registry never issued cannot be cited, so a hallucinated source is a missing
lookup rather than a plausible-looking citation.

It is not thread-safe: registering computes the next alias from the size and then writes two
dicts, so callers must serialize. `make_tools` holds one lock around every tool body, which is
what keeps two concurrent searches from issuing the same alias to different papers.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from paperwatch.store.repository import ItemDetail


@dataclass(frozen=True)
class SourceRef:
    """The part of a source an answer may cite: identity, attribution and reliability."""

    alias: str
    item_id: uuid.UUID
    title: str
    url: str
    tier: str
    basis: str
    published_at: datetime


class SourceRegistry:
    def __init__(self) -> None:
        self._by_item: dict[uuid.UUID, SourceRef] = {}
        self._by_alias: dict[str, SourceRef] = {}

    def register(self, detail: ItemDetail) -> SourceRef:
        # The same item keeps its alias however often a tool returns it, so two searches in one
        # answer never make one paper look like two.
        ref = self._by_item.get(detail.item_id)
        if ref is None:
            alias = f"S{len(self._by_item) + 1}"
            ref = SourceRef(
                alias,
                detail.item_id,
                detail.title,
                detail.url,
                detail.tier,
                detail.basis,
                detail.published_at,
            )
            self._by_item[detail.item_id] = ref
            self._by_alias[alias] = ref
        return ref

    def get(self, alias: str) -> SourceRef | None:
        return self._by_alias.get(alias)

    def all(self) -> list[SourceRef]:
        # Insertion order is the alias order: S1 first, so the list reads like the answer.
        return list(self._by_alias.values())
