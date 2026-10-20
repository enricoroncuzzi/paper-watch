"""Check a finished answer's citations against the sources this answer's tools returned.

A model cites fluently whether or not the source exists, so grounding is verified by code after
generation. The checker can see only what the program knows: which aliases the registry issued
during this answer, and each source's tier from the database. It therefore does not judge whether
a sentence is true; it judges whether every claim points at evidence the reader can open, and it
refuses to pass off an answer with no evidence.

Everything this module returns is untrusted in one of two ways, and must be escaped wherever it
is rendered into a prompt, an SSE event or a terminal: `CheckedAnswer.text`, `Flag.sentence` and
`Flag.detail` are model-authored, while `Citation.title` and `Citation.url` are stored third-party
data. Renderers show the text as plain text and take every link from `Citation.url`, so a link the
model wrote never becomes clickable.
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
# A model can echo an arbitrarily long alias; a flag repeats it back, so it is cut first.
ALIAS_MAX_CHARS = 32

# Any alias-like group, whatever its case, spacing or separator: [S1, S2], [s1], (S9), [S 9],
# [S1; S2], [S1 and S2]. A citation the checker does not recognise is one it cannot flag, so the
# detector is deliberately wider than the canonical form it rewrites.
# Zero-width characters are invisible to the reader, so they must not be invisible to the check.
_GAP = r"[\s\u200b\u200c\u200d\ufeff]"
_ALIAS = rf"[Ss]{_GAP}*\d+"
_SEP = rf"{_GAP}*(?:,|;|&|\band\b){_GAP}*"
_GROUP = re.compile(rf"[\[(]{_GAP}*({_ALIAS}(?:{_SEP}{_ALIAS})*){_GAP}*[\])]")
_ALIAS_PART = re.compile(rf"[Ss]{_GAP}*(\d+)")
# Aliases quoted as code or living inside a URL are content, not citations (N4).
_PROTECTED = re.compile(r"`[^`\n]*`|https?://\S+")

# The rules below turn every sentence end into a newline, so this split is the only one needed.
_BOUNDARY = re.compile(r"\n+")
_TERMINATOR_CLOSERS = re.compile(r"([.!?][\"')\]*_]*\s+)")
_TERMINATOR_CAPITAL = re.compile(r"([.!?][\"')\]*_]?)(?=[A-Z])")
_MARKER_END = re.compile(r"(\]\s+)(?=[A-Z])")
_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)]|[•·])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s")
_TABLE = re.compile(r"^\s*\|")
_BOLD_LINE = re.compile(r"^\s*\*\*.*\*\*\s*$")
# "et al." and "e.g." end in a full stop but not a sentence; the mark survives the split and is
# restored after it.
_ABBREV = re.compile(r"\b(?:etc|al|e\.g|i\.e|vs|Fig|Dr)\.", re.IGNORECASE)
_ABBREV_MARK = "\x00"

# The phrasing our corpus can contradict, plus the synonyms models actually write. The separator
# class includes the non-breaking hyphen. A preprint is never described as peer-reviewed.
_TIER_CLAIM = re.compile(
    r"peer[\s\-\u2010-\u2015]*review(?:ed)?"
    r"|refereed"
    r"|accepted\s+(?:at|to|for)"
    r"|published\s+(?:at|in)",
    re.IGNORECASE,
)
# An honest label the product wants ("a preprint, not yet peer-reviewed") must not be flagged.
_NEGATED = re.compile(
    r"(?:\bnot\b|\bnon\b|\bwithout\b|\bawaiting\b|\byet\s+to\s+be\b|\bun)(?:\W+\w+){0,2}\W*$",
    re.IGNORECASE,
)

FlagKind = Literal["unknown_source", "uncited_claim", "tier_mislabel"]


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
    kind: FlagKind
    sentence: str
    detail: str


@dataclass(frozen=True)
class CheckedAnswer:
    text: str
    citations: tuple[Citation, ...]
    flags: tuple[Flag, ...]
    supported: bool


def check_answer(text: str, registry: SourceRegistry) -> CheckedAnswer:
    citations: list[Citation] = []
    flags: list[Flag] = []
    seen: set[str] = set()
    evidence = False

    for sentence in _sentences(text):
        groups = list(_groups(sentence))
        tier_claim = _has_tier_claim(sentence)
        # Every known alias of this sentence, cited before or not: the tier check judges the
        # sentence, not the order a source was first seen in.
        refs: dict[str, SourceRef] = {}
        for group in groups:
            for alias in _aliases(group.group(1)):
                ref = registry.get(alias)
                if ref is None:
                    flags.append(
                        Flag(
                            "unknown_source",
                            _clean(sentence),
                            f"{_clip_alias(alias)} was not returned by any tool for this answer",
                        )
                    )
                    continue
                refs[ref.alias] = ref
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

        if not groups:
            # A tier phrase with no citation at all is a claim whatever its length.
            if tier_claim or _is_claim(sentence):
                flags.append(Flag("uncited_claim", _clean(sentence), "no source cited"))
        # A sentence may cite several sources; the label holds if any one of them supports it.
        elif tier_claim and refs and all(ref.tier != Tier.PEER_REVIEWED for ref in refs.values()):
            tiers = ", ".join(f"{ref.alias} is {ref.tier}" for ref in refs.values())
            flags.append(Flag("tier_mislabel", _clean(sentence), f"not peer-reviewed: {tiers}"))

        # A citation is evidence only when it sits beside an actual claim; an answer made of
        # bare markers is the model declining to say anything.
        if refs and _is_claim(sentence):
            evidence = True

    found = _dedupe(flags)
    if not evidence:
        return CheckedAnswer(NO_SUPPORT, (), found, False)
    return CheckedAnswer(_rewrite(text, registry), tuple(citations), found, not found)


def _aliases(content: str) -> list[str]:
    # Normalise case and inner spacing, keeping Unicode digits: those are still aliases nobody
    # issued, and an unmatched alias must be masked, not missed.
    return list(dict.fromkeys(f"S{digits}" for digits in _ALIAS_PART.findall(content)))


def _canonical(content: str, registry: SourceRegistry) -> str:
    parts = ["?" if registry.get(alias) is None else alias for alias in _aliases(content)]
    return f"[{', '.join(parts)}]"


def _clip_alias(alias: str) -> str:
    return alias if len(alias) <= ALIAS_MAX_CHARS else f"{alias[:ALIAS_MAX_CHARS]}..."


def _dedupe(flags: list[Flag]) -> tuple[Flag, ...]:
    return tuple(dict.fromkeys(flags))


def _skip(match: re.Match[str], spans: list[tuple[int, int]]) -> bool:
    return any(start < match.end() and match.start() < end for start, end in spans)


def _groups(sentence: str) -> list[re.Match[str]]:
    spans = [(m.start(), m.end()) for m in _PROTECTED.finditer(sentence)]
    return [m for m in _GROUP.finditer(sentence) if not _skip(m, spans)]


def _strip_groups(sentence: str) -> str:
    spans = [(m.start(), m.end()) for m in _PROTECTED.finditer(sentence)]
    pieces: list[str] = []
    pos = 0
    for match in _GROUP.finditer(sentence):
        if _skip(match, spans):
            continue
        pieces.append(sentence[pos : match.start()])
        pos = match.end()
    pieces.append(sentence[pos:])
    return "".join(pieces)


def _rewrite(text: str, registry: SourceRegistry) -> str:
    spans = [(m.start(), m.end()) for m in _PROTECTED.finditer(text)]
    pieces: list[str] = []
    pos = 0
    for match in _GROUP.finditer(text):
        if _skip(match, spans):
            continue
        pieces.append(text[pos : match.start()])
        pieces.append(_canonical(match.group(1), registry))
        pos = match.end()
    pieces.append(text[pos:])
    return "".join(pieces)


def _has_tier_claim(sentence: str) -> bool:
    if _is_question(sentence) or _structural(sentence):
        return False
    return any(not _negated(sentence, match.start()) for match in _TIER_CLAIM.finditer(sentence))


def _negated(sentence: str, start: int) -> bool:
    return bool(_NEGATED.search(sentence[max(0, start - 40) : start]))


def _strip_fences(text: str) -> str:
    kept: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            kept.append(line)
    return "\n".join(kept)


def _sentences(text: str) -> list[str]:
    # Protect abbreviations before the terminators below split on their full stops.
    marked = _ABBREV.sub(lambda m: m.group(0)[:-1] + _ABBREV_MARK, _strip_fences(text))
    marked = _TERMINATOR_CLOSERS.sub(r"\1\n", marked)
    marked = _TERMINATOR_CAPITAL.sub(r"\1\n", marked)
    marked = _MARKER_END.sub(r"\1\n", marked)
    parts = [_BULLET.sub("", part).strip() for part in _BOUNDARY.split(marked)]
    out: list[str] = []
    for part in filter(None, parts):
        part = part.replace(_ABBREV_MARK, ".")
        # "...bugs. [S1]" puts the marker after the full stop; a part made only of markers
        # belongs to the sentence before it.
        if out and not _strip_groups(part).strip():
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _structural(sentence: str) -> bool:
    # Headings, table rows, code and colon lead-ins are layout, not assertions.
    return bool(
        _HEADING.match(sentence) or _TABLE.match(sentence) or sentence.rstrip().endswith(":")
    )


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def _word_count(text: str) -> int:
    # Punctuation alone (the full stop a marker left behind) is not a word.
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def _is_claim(sentence: str) -> bool:
    if _is_question(sentence) or _structural(sentence):
        return False
    # A wholly bold line is a heading, unless it carries a citation and so is a real sentence.
    if _BOLD_LINE.match(sentence) and not _groups(sentence):
        return False
    return _word_count(_strip_groups(sentence)) >= CLAIM_MIN_WORDS


def _clean(sentence: str) -> str:
    # A flag names the sentence itself, so the reader can find it in the answer.
    return " ".join(_strip_groups(sentence).split()).rstrip(".").strip()
