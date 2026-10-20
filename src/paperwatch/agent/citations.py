"""Check a finished answer's citations against the sources this answer's tools returned.

A model cites fluently whether or not the source exists, so grounding is verified by code after
generation. The checker can see only what the program knows: which aliases the registry issued
during this answer, and each source's tier from the database. It does not judge whether a sentence
is true; it judges whether each claim points at evidence the reader can open, and it refuses to
pass off an answer with no evidence.

This is a heuristic safety net, not a parser of English. Three guarantees hold, and everything
else is a known limit in `docs/learn/11`: an alias no tool returned never becomes a `Citation` and
is always visible as `?`; an answer with no cited claim is `NO_SUPPORT`; a cited source described
as peer-reviewed when it is not is flagged.

Everything this module returns is untrusted in one of two ways, and must be escaped wherever it
is rendered into a prompt, an SSE event or a terminal: `CheckedAnswer.text`, `Flag.sentence` and
`Flag.detail` are model-authored, while `Citation.title` and `Citation.url` are stored third-party
data. No markup the model wrote is rendered as a link and every clickable link comes from
`Citation.url`; a bare URL stays visible as plain text and is not checked against the corpus.
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
# Below this, a cited fragment ("ok [S1]", "See [S1].") is the model stalling, not answering.
EVIDENCE_MIN_WORDS = 3
# A model can echo an arbitrarily long alias; a flag repeats it back, so it is cut first.
ALIAS_MAX_CHARS = 32

# Any alias token, whatever its case or spacing. Zero-width characters are invisible to the reader,
# so they must not be invisible to the check.
_GAP = r"[\s\u200b\u200c\u200d\ufeff]"
_ALIAS = rf"(?<!\w)[Ss]{_GAP}*\d+(?!\w)"
_SEP = rf"{_GAP}*(?:,{_GAP}*(?:and{_GAP}*)?|;{_GAP}*|&{_GAP}*|and{_GAP}*){_GAP}*"
_ALIAS_PART = re.compile(rf"[Ss]{_GAP}*(\d+)")

# Code, a closed fence and a URL are content, never citations. One pass finds protectors and
# citation groups together, so detection, stripping and rewriting agree and stay linear.
_CODE = r"`[^`\n]*`"
_FENCE = r"^[ \t]*```.*?^[ \t]*```[^\n]*"
_URL = r"https?://\S+"
# A square bracket that holds an alias token anywhere is a citation group, whatever else it holds
# ("[S1, S2, and S9]", "[S9, p. 4]"). A paren group counts only when it is aliases and separators
# alone, so "(S3)" for storage stays a known limit.
_SQUARE = rf"\[[^\[\]\n]{{0,200}}?{_ALIAS}[^\[\]\n]*\]"
_PAREN = rf"\({_GAP}*{_ALIAS}(?:{_SEP}{_ALIAS})*{_GAP}*\)"
_SPAN = re.compile(
    rf"(?P<skip>{_FENCE}|{_CODE}|{_URL})|(?P<square>{_SQUARE})|(?P<paren>{_PAREN})",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# The rules below turn every sentence end into a newline, so this split is the only one needed.
_BOUNDARY = re.compile(r"\n+")
_TERMINATOR_CLOSERS = re.compile(r"([.!?][\"')\]*_]*\s+)")
_TERMINATOR_CAPITAL = re.compile(r"([.!?][\"')\]*_]?)(?=[A-Z])")
_MARKER_END = re.compile(r"(\]\s+)(?=[A-Z])")
_SQUARE_ALL = re.compile(r"\[[^\[\]\n]*\]")
_BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)]|[•·])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s")
_TABLE = re.compile(r"^\s*\|")
_BOLD_LINE = re.compile(r"^\s*\*\*.*\*\*\s*$")
# Abbreviations end in a full stop but not a sentence; every dot is marked and restored after the
# split. "etc." ends a sentence before a capital, so it is protected only before a lowercase word,
# a digit or punctuation.
_ABBREV = re.compile(
    r"\betc\.(?=\s+[a-z0-9,;(])"
    r"|\b(?:al|e\.g|i\.e|vs|Fig|Dr|Inc|Corp|Ltd|No|Prof|Sec|Eq|approx)\."
    r"|(?:[A-Z]\.){2,}"
)
_ABBREV_MARK = "\x00"

# The phrasing our corpus can contradict; the separator class includes the non-breaking hyphen.
_TIER_CLAIM = re.compile(
    r"peer[\s\-\u2010-\u2015]*review(?:ed)?"
    r"|refereed"
    r"|accepted\s+(?:at|to|for)"
    r"|published\s+at",
    re.IGNORECASE,
)
# An honest label the product asks for ("a preprint, not yet peer-reviewed") must not be flagged.
# "not only" and "not just" assert the opposite, so they are not negations.
_NEGATED = re.compile(
    r"(?:\bnot\b(?!\s+(?:only|just)\b)|\bnon\b|n't\b|\bnever\b|\bno\b|\bun"
    r"|\bwithout\b|\bawaiting\b|\bpending\b|\byet\s+to\s+be\b|\bunder\b|\bsubmitted\s+for\b)"
    r"(?:\W+\w+){0,3}\W*$",
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
        groups = _groups(sentence)
        tier_claim = _has_tier_claim(sentence)
        # Every known alias of this sentence, cited before or not: the tier check judges the
        # sentence, not the order a source was first seen in.
        refs: dict[str, SourceRef] = {}
        for group in groups:
            for alias in _tokens(group.group(0)):
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
            # A cited tier phrase is checked whatever its layout; an uncited one is a claim only
            # outside layout, so an uncited table row or heading stays unflagged.
            if _is_claim(sentence) or (tier_claim and not _structural(sentence)):
                flags.append(Flag("uncited_claim", _clean(sentence), "no source cited"))
        # A sentence may cite several sources; the label holds if any one of them supports it.
        elif tier_claim and refs and all(ref.tier != Tier.PEER_REVIEWED for ref in refs.values()):
            tiers = ", ".join(f"{ref.alias} is {ref.tier}" for ref in refs.values())
            flags.append(Flag("tier_mislabel", _clean(sentence), f"not peer-reviewed: {tiers}"))

        # A citation is evidence only beside actual words; a bare marker list is not an answer,
        # while a terse table row or bullet is.
        if refs and _word_count(_strip_groups(sentence)) >= EVIDENCE_MIN_WORDS:
            evidence = True

    found = _dedupe(flags)
    if not evidence:
        return CheckedAnswer(NO_SUPPORT, (), found, False)
    return CheckedAnswer(_rewrite(text, registry), tuple(citations), found, not found)


def _scan(text: str) -> list[re.Match[str]]:
    # A protector earlier in the alternation consumes its span, so a group cannot start inside it.
    return [m for m in _SPAN.finditer(text) if m.lastgroup != "skip"]


def _tokens(text: str) -> list[str]:
    # Normalise case and inner spacing, keeping Unicode digits: those are still aliases nobody
    # issued, and an unmatched alias must be masked, not missed.
    return list(dict.fromkeys(f"S{digits}" for digits in _ALIAS_PART.findall(text)))


def _render(alias: str, registry: SourceRegistry) -> str:
    return alias if registry.get(alias) is not None else "?"


def _canonical(match: re.Match[str], registry: SourceRegistry) -> str:
    if match.lastgroup == "paren":
        # A paren group is aliases and separators alone, so it becomes the canonical list.
        return f"[{', '.join(_render(alias, registry) for alias in _tokens(match.group(0)))}]"
    # A square group may say anything else too, so keep its wording and replace only aliases.
    return _ALIAS_PART.sub(lambda m: _render(_alias_of(m.group(0)), registry), match.group(0))


def _alias_of(token: str) -> str:
    digits = _ALIAS_PART.match(token)
    return f"S{digits.group(1)}" if digits else token


def _clip_alias(alias: str) -> str:
    return alias if len(alias) <= ALIAS_MAX_CHARS else f"{alias[:ALIAS_MAX_CHARS]}..."


def _dedupe(flags: list[Flag]) -> tuple[Flag, ...]:
    return tuple(dict.fromkeys(flags))


def _groups(sentence: str) -> list[re.Match[str]]:
    return _scan(sentence)


def _strip_groups(sentence: str) -> str:
    kept: list[str] = []
    pos = 0
    for match in _scan(sentence):
        kept.append(sentence[pos : match.start()])
        pos = match.end()
    kept.append(sentence[pos:])
    return "".join(kept)


def _rewrite(text: str, registry: SourceRegistry) -> str:
    kept: list[str] = []
    pos = 0
    for match in _scan(text):
        kept.append(text[pos : match.start()])
        kept.append(_canonical(match, registry))
        pos = match.end()
    kept.append(text[pos:])
    return "".join(kept)


def _has_tier_claim(sentence: str) -> bool:
    if _is_question(sentence):
        return False
    return any(not _negated(sentence, match.start()) for match in _TIER_CLAIM.finditer(sentence))


def _negated(sentence: str, start: int) -> bool:
    return bool(_NEGATED.search(sentence[max(0, start - 80) : start]))


def _strip_fences(text: str) -> str:
    # Only a fence closed by a later fence line is a fence; an unclosed one is ordinary text.
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith("```"):
            end = index + 1
            while end < len(lines) and not lines[end].lstrip().startswith("```"):
                end += 1
            if end < len(lines):
                index = end + 1
                continue
        kept.append(lines[index])
        index += 1
    return "".join(kept)


def _sentences(text: str) -> list[str]:
    # Protect abbreviations before the terminators below split on their full stops.
    marked = _ABBREV.sub(lambda m: m.group(0).replace(".", _ABBREV_MARK), _strip_fences(text))
    # A citation group can hold a full stop ("[S9, p. 4]"); it is part of the marker.
    marked = _SQUARE_ALL.sub(lambda m: m.group(0).replace(".", _ABBREV_MARK), marked)
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


def _word_count(text: str) -> int:
    # Punctuation alone (the full stop a marker left behind) is not a word.
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def _structural(sentence: str) -> bool:
    # Headings, table rows and colon lead-ins are layout, not assertions.
    return bool(
        _HEADING.match(sentence) or _TABLE.match(sentence) or sentence.rstrip().endswith(":")
    )


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def _is_claim(sentence: str) -> bool:
    # Only called on sentences with no citation group, so there is no marker to strip.
    if _is_question(sentence) or _structural(sentence):
        return False
    # A wholly bold line is a heading, unless it carries a citation and so is a real sentence.
    if _BOLD_LINE.match(sentence) and not _groups(sentence):
        return False
    return _word_count(sentence) >= CLAIM_MIN_WORDS


def _clean(sentence: str) -> str:
    # A flag names the sentence itself, so the reader can find it in the answer.
    return " ".join(_strip_groups(sentence).split()).rstrip(".").strip()
