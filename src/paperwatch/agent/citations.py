"""Check a finished answer's citations against the sources this answer's tools returned.

A model cites fluently whether or not the source exists, so grounding is verified by code after
generation. The checker can see only what the program knows: which aliases the registry issued
during this answer, and each source's tier from the database. It therefore does not judge whether
a sentence is true; it judges whether every claim points at evidence the reader can open, and it
refuses to pass off an answer with no evidence.

This module is pure: it takes the answer text and the answer's registry, and returns a fixed copy
of the text with unknown aliases masked. Rendering a citation's title into a prompt or a stream is
someone else's job, and must escape it there: stored text is untrusted.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Literal

from paperwatch.agent.sources import SourceRef, SourceRegistry
from paperwatch.domain import Tier

NO_SUPPORT = "I could not find anything in the corpus that answers this question."
# A shorter fragment ("In short:") carries no factual claim worth a citation.
CLAIM_MIN_WORDS = 6

_MARKER = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
# The phrasing our corpus can contradict: a preprint is never described as peer-reviewed.
_TIER_CLAIM = re.compile(r"peer[- ]reviewed|accepted at|published at", re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    alias: str
    item_id: uuid.UUID
    title: str
    url: str
    tier: str
    basis: str


@dataclass(frozen=True)
class Flag:
    kind: Literal["unknown_source", "uncited_claim", "tier_mislabel"]
    sentence: str
    detail: str


@dataclass(frozen=True)
class CheckedAnswer:
    text: str
    citations: tuple[Citation, ...]
    flags: tuple[Flag, ...]
    supported: bool


def check_answer(text: str, registry: SourceRegistry) -> CheckedAnswer:
    sentences = _sentences(text)
    citations: list[Citation] = []
    flags: list[Flag] = []
    seen: set[str] = set()

    for sentence in sentences:
        markers = list(_MARKER.finditer(sentence))
        if not markers:
            if _is_claim(sentence):
                flags.append(Flag("uncited_claim", _clean(sentence), "no source cited"))
            continue

        # Every known alias in the sentence, cited before or not: the tier check judges the
        # sentence, not the order it was first seen in.
        sentence_refs: list[SourceRef] = []
        for marker in markers:
            for alias in (part.strip() for part in marker.group(1).split(",")):
                ref = registry.get(alias)
                if ref is None:
                    flags.append(
                        Flag(
                            "unknown_source",
                            _clean(sentence),
                            f"{alias} was not returned by any tool for this answer",
                        )
                    )
                    continue
                sentence_refs.append(ref)
                if ref.alias not in seen:
                    seen.add(ref.alias)
                    citations.append(
                        Citation(
                            ref.alias,
                            ref.item_id,
                            ref.title,
                            ref.url,
                            ref.tier,
                            ref.basis,
                        )
                    )

        # A sentence may cite several sources; the label holds if any one of them supports it.
        if (
            _TIER_CLAIM.search(sentence)
            and sentence_refs
            and all(ref.tier != Tier.PEER_REVIEWED for ref in sentence_refs)
        ):
            tiers = ", ".join(f"{ref.alias} is {ref.tier}" for ref in sentence_refs)
            flags.append(Flag("tier_mislabel", _clean(sentence), f"not peer-reviewed: {tiers}"))

    checked = tuple(citations)
    outcome = tuple(flags)
    if not checked:
        return CheckedAnswer(NO_SUPPORT, (), outcome, False)
    return CheckedAnswer(
        _MARKER.sub(lambda m: _mask(m.group(1), registry), text), checked, outcome, not outcome
    )


def _mask(aliases: str, registry: SourceRegistry) -> str:
    kept = [a.strip() for a in aliases.split(",") if registry.get(a.strip()) is not None]
    return f"[{', '.join(kept)}]" if kept else "[?]"


def _sentences(text: str) -> list[str]:
    parts = [_BULLET.sub("", part).strip() for part in _BOUNDARY.split(text)]
    out: list[str] = []
    for part in filter(None, parts):
        # "...bugs. [S1]" puts the marker after the full stop; a part made only of markers
        # belongs to the sentence before it.
        if out and not _MARKER.sub("", part).strip():
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _is_claim(sentence: str) -> bool:
    # Short connective lines ("In short:") and questions carry no factual claim.
    words = _MARKER.sub("", sentence).split()
    return len(words) >= CLAIM_MIN_WORDS and not sentence.rstrip().endswith("?")


def _clean(sentence: str) -> str:
    # A flag names the sentence itself, so the reader can find it in the answer.
    return _MARKER.sub("", sentence).strip().rstrip(".").strip()
