"""Per-answer source aliases: the model may only cite what a tool put in this answer's registry.

The registry is created for one answer and thrown away with it, unless the answer belongs to a
thread: a thread stores its registry with the turn, so the next turn starts from the same aliases
(`from_refs`) and `S2` from the first answer is still citable and checkable in the fifth. A tool
registers an item and gets back a short alias (`S1`, `S2`, ... in first-seen order); the model then
refers to sources by that alias. An alias the registry never issued cannot be cited, so a
hallucinated source is a missing lookup rather than a plausible-looking citation.

It is not thread-safe: registering computes the next alias from the size and then writes two
dicts, so callers must serialize. `make_tools` holds one lock around every tool body, which is
what keeps two concurrent searches from issuing the same alias to different papers.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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

    @classmethod
    def from_refs(cls, refs: Sequence[SourceRef]) -> "SourceRegistry":
        """A registry restored from a thread: aliases are kept exactly as they were issued."""
        registry = cls()
        for ref in refs:
            registry._by_item[ref.item_id] = ref
            registry._by_alias[ref.alias] = ref
        return registry

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

    def refs(self) -> list[SourceRef]:
        # Insertion order is the alias order: S1 first, so the list reads like the answer.
        return list(self._by_alias.values())


def ref_to_json(ref: SourceRef) -> dict[str, str]:
    """One ref as the thread's `registry` column stores it: JSON has no UUID or datetime."""
    return {
        "alias": ref.alias,
        "item_id": str(ref.item_id),
        "title": ref.title,
        "url": ref.url,
        "tier": ref.tier,
        "basis": ref.basis,
        "published_at": ref.published_at.isoformat(),
    }


def ref_from_json(data: Mapping[str, Any]) -> SourceRef:
    """The ref a stored thread turn carries; a stored row is data, so every field is coerced."""
    return SourceRef(
        alias=str(data["alias"]),
        item_id=uuid.UUID(str(data["item_id"])),
        title=str(data["title"]),
        url=str(data["url"]),
        tier=str(data["tier"]),
        basis=str(data["basis"]),
        published_at=datetime.fromisoformat(str(data["published_at"])),
    )
