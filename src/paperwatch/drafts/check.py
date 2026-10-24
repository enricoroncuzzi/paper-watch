"""The draft check: code, not a model, proves what a draft claims (Task 38).

Every number, quote and link in the draft must come from the stored title or text, a checked
summary quote, or the owner's own point of view. A first-person claim about building or testing
must be in his point of view too. A lab post must name the lab, a preprint's numbers must be
attributed to its authors, an early item must say so, and a long passage copied from the article
is flagged rather than silently accepted.

The check never edits the draft: each failure is a numbered note on the saved draft (ADR 0015), so
the reader sees exactly which claim to fix. Normalisation follows the summary check: the same
Unicode folding, HTML unescaping, whitespace collapsing and number comparison, written here so
the two checks keep their own rules as they evolve.
"""

import html
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from paperwatch.drafts.generate import DraftSource, DraftText

FlagKind = Literal[
    "number",
    "quote",
    "link",
    "first_person",
    "attribution",
    "preprint_fact",
    "early_notice",
    "long_copy",
]
# The order the page numbers its notes in: most factual first, so a wrong number is note 1.
KIND_ORDER: tuple[FlagKind, ...] = (
    "number",
    "quote",
    "link",
    "first_person",
    "attribution",
    "preprint_fact",
    "early_notice",
    "long_copy",
)
MIN_QUOTE_WORDS = 3
# Twelve consecutive words is a passage, not a turn of phrase; ten would catch ordinary phrasing
# shared with the abstract, and the point is to catch copying, not to force synonyms.
LONG_COPY_WORDS = 12

# A number with an optional decimal or thousands part and an optional percent sign.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s?%?")
# A sentence boundary: the whitespace after terminal punctuation.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# Straight or curly double quotes; single quotes are not delimiters because contractions use them.
_QUOTE = re.compile(r'"([^"\n]{1,})"|\u201c([^\u201d\n]{1,})\u201d')
# A URL with a scheme, a www. host, or a bare domain with a path. The third form keeps a filename
# such as "run.py" from being read as a link.
_URL = re.compile(
    r"https?://[^\s<>\"')]+|\bwww\.[^\s<>\"')]+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s<>\"')]+",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(
    r"\b(?:i|i'm|i've|i'd|i'll|my|mine|we|we're|we've|we'd|we'll|our|ours|us)\b"
)
# The experiences a draft may not invent. Each stem's own forms are matched, so a draft that says
# "built" is allowed only by a point of view that says "build" or "built".
_EXPERIENCE_VERBS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (stem, re.compile(rf"\b(?:{forms})\b"))
    for stem, forms in (
        ("build", "build|builds|building|built"),
        ("test", "test|tests|tested|testing"),
        ("try", "try|tries|tried|trying"),
        ("use", "use|uses|used|using"),
        ("run", "run|runs|ran|running"),
        ("ship", "ship|ships|shipped|shipping"),
        ("deploy", "deploy|deploys|deployed|deploying"),
        ("implement", "implement|implements|implemented|implementing"),
        ("benchmark", "benchmark|benchmarks|benchmarked|benchmarking"),
        ("evaluate", "evaluate|evaluates|evaluated|evaluating"),
        ("measure", "measure|measures|measured|measuring"),
    )
)
# A verb that turns a sentence into a report of someone else's claim.
_ATTRIBUTION_VERB = re.compile(
    r"\b(?:reports?|reported|says?|said|claims?|claimed|announces?|announced|writes?|wrote|"
    r"describes?|described)\b"
)
# The same cues, plus the phrasings a preprint attribution takes.
_ATTRIBUTION_CUE = re.compile(
    r"\b(?:reports?|reported|says?|said|claims?|claimed|announces?|announced|writes?|wrote|"
    r"describes?|described|according to|the authors|preprints?)\b"
)
_EARLY = re.compile(r"\bearly\b")
_NOT_YET_REVIEWED = re.compile(
    r"not yet independently reviewed|not (?:yet )?peer[- ]reviewed|"
    r"has(?: not|n't) been peer[- ]reviewed|without peer review"
)
_WORD = re.compile(r"[a-z0-9']+")
# Typographic variants a model produces that are the same character to a reader. Written as
# escapes so this file contains no dash character of its own.
_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


@dataclass(frozen=True)
class DraftFlag:
    """One claim the code could not verify, with the words it is about and why."""

    kind: FlagKind
    text: str
    detail: str


@dataclass(frozen=True)
class DraftCheck:
    """The numbered notes and, per kind, how many checks ran and how many passed."""

    flags: tuple[DraftFlag, ...]
    counts: Mapping[str, tuple[int, int]]


def check_draft(draft: DraftText, source: DraftSource, *, pov: str) -> DraftCheck:
    """Run every rule over the hook, body and hashtags together."""
    text = "\n".join((draft.hook, draft.body, " ".join(f"#{tag}" for tag in draft.hashtags)))
    folded_pov = _normalise(pov)
    folded_store = _normalise(f"{source.title}\n{source.text}")
    quotes = _summary_quotes(source.summary)
    # A summary quote is checked against the stored text when it was written, so its numbers are
    # stored evidence as much as the text's own.
    evidence = set(_numbers(_normalise(f"{source.title}\n{source.text}\n{quotes}\n{pov}")))
    counts: dict[FlagKind, list[int]] = {kind: [0, 0] for kind in KIND_ORDER}
    found: dict[FlagKind, list[DraftFlag]] = {kind: [] for kind in KIND_ORDER}

    _check_numbers(text, evidence, counts, found)
    _check_quotes(text, folded_store, folded_pov, counts, found)
    _check_links(draft, text, source, folded_pov, counts, found)
    _check_first_person(text, folded_pov, counts, found)
    _check_attribution(text, source, counts, found)
    _check_preprint_numbers(text, source, counts, found)
    _check_early_notice(text, source, counts, found)
    _check_long_copy(text, folded_store, counts, found)

    flags = tuple(flag for kind in KIND_ORDER for flag in found[kind])
    return DraftCheck(
        flags=flags,
        counts={str(kind): (checked, passed) for kind, (checked, passed) in counts.items()},
    )


def _check_numbers(
    text: str,
    evidence: set[str],
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    flagged: set[str] = set()
    for number in _numbers(_normalise(text)):
        counts["number"][0] += 1
        if number in evidence:
            counts["number"][1] += 1
            continue
        if number in flagged:
            continue
        flagged.add(number)
        found["number"].append(
            DraftFlag(
                "number",
                _sentence_with(text, number),
                f"{number} does not appear in the stored title or text, a checked summary "
                "quote, or your point of view.",
            )
        )


def _check_quotes(
    text: str,
    folded_store: str,
    folded_pov: str,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    for span in _quoted_spans(text):
        folded = _normalise(span)
        if len(folded.split()) < MIN_QUOTE_WORDS:
            # "so-called" or a two-word scare quote is ordinary writing, not a copied claim.
            continue
        counts["quote"][0] += 1
        if folded in folded_store or (folded_pov and folded in folded_pov):
            counts["quote"][1] += 1
            continue
        found["quote"].append(
            DraftFlag(
                "quote",
                span,
                "This quote is not in the stored text or in your point of view.",
            )
        )


def _check_links(
    draft: DraftText,
    text: str,
    source: DraftSource,
    folded_pov: str,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    # A link is allowed when it is the item, its code or project page, or one the owner wrote in
    # his point of view. Links the draft itself invents are exactly what this check exists for.
    allowed = {
        _url_key(url)
        for url in (source.url, source.github_repo, source.project_page, *_links(folded_pov))
        if url
    }
    flagged: set[str] = set()
    if draft.source_link:
        counts["link"][0] += 1
        key = _url_key(draft.source_link)
        if key == _url_key(source.url):
            counts["link"][1] += 1
        else:
            flagged.add(key)
            found["link"].append(
                DraftFlag(
                    "link",
                    draft.source_link,
                    f"The source link must be the item's own link: {source.url}",
                )
            )
    for link in _links(text):
        counts["link"][0] += 1
        key = _url_key(link)
        if key in allowed:
            counts["link"][1] += 1
            continue
        if key in flagged:
            continue
        flagged.add(key)
        found["link"].append(
            DraftFlag(
                "link",
                link,
                f"{link} is not the item's link, its repository or project page, or a link "
                "you wrote in your point of view.",
            )
        )


def _check_first_person(
    text: str,
    folded_pov: str,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    pov_sentences = [_normalise(sentence) for sentence in _SENTENCE.split(folded_pov)]
    for sentence in _sentences(text):
        folded = _normalise(sentence)
        if _FIRST_PERSON.search(folded) is None:
            continue
        stems = [stem for stem, pattern in _EXPERIENCE_VERBS if pattern.search(folded) is not None]
        if not stems:
            continue
        counts["first_person"][0] += 1
        unsupported = [stem for stem in stems if not _pov_supports(stem, pov_sentences)]
        if not unsupported:
            counts["first_person"][1] += 1
            continue
        found["first_person"].append(
            DraftFlag(
                "first_person",
                sentence,
                f"The draft says you {unsupported[0]} something, but your point of view does "
                "not say so.",
            )
        )


def _check_attribution(
    text: str,
    source: DraftSource,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    if source.tier != "lab_blog":
        return
    if not source.org:
        # Nothing to check against: a lab post without a stored organisation cannot be held to a
        # name, and inventing the requirement would flag every such draft.
        return
    counts["attribution"][0] = 1
    name = _normalise(source.org)
    for sentence in _sentences(text):
        folded = _normalise(sentence)
        if name in folded and _ATTRIBUTION_VERB.search(folded) is not None:
            counts["attribution"][1] = 1
            return
    found["attribution"].append(
        DraftFlag(
            "attribution",
            _first_sentence(text),
            f"A lab post is {source.org}'s own claim: name the lab next to an attribution verb, "
            f'as in "{source.org} reports ...".',
        )
    )


def _check_preprint_numbers(
    text: str,
    source: DraftSource,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    if source.tier == "peer_reviewed":
        return
    for sentence in _sentences(text):
        folded = _normalise(sentence)
        if _NUMBER.search(folded) is None:
            continue
        counts["preprint_fact"][0] += 1
        if _ATTRIBUTION_CUE.search(folded) is not None:
            counts["preprint_fact"][1] += 1
            continue
        found["preprint_fact"].append(
            DraftFlag(
                "preprint_fact",
                sentence,
                "A preprint's number is the authors' claim, not established fact: attribute it "
                'as in "the authors report ..." or "according to the preprint ...".',
            )
        )


def _check_early_notice(
    text: str,
    source: DraftSource,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    if not source.early:
        return
    counts["early_notice"][0] = 1
    folded = _normalise(text)
    if _EARLY.search(folded) is not None or _NOT_YET_REVIEWED.search(folded) is not None:
        counts["early_notice"][1] = 1
        return
    found["early_notice"].append(
        DraftFlag(
            "early_notice",
            _first_sentence(text),
            "This item is early work: the draft must say it is early or not yet independently "
            "reviewed, and must not present its claims as established.",
        )
    )


def _check_long_copy(
    text: str,
    folded_store: str,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    # A passage inside quotation marks is a quote, and the quote rule already checks it against the
    # source; copying is the same words outside quotation marks.
    unquoted = _normalise(_QUOTE.sub(" ", text))
    source_words = _WORD.findall(folded_store)
    if len(source_words) < LONG_COPY_WORDS:
        return
    source_grams = {
        tuple(source_words[index : index + LONG_COPY_WORDS])
        for index in range(len(source_words) - LONG_COPY_WORDS + 1)
    }
    draft_words = _WORD.findall(unquoted)
    run = 0
    copied: list[list[str]] = []
    for index in range(len(draft_words)):
        gram = tuple(draft_words[index : index + LONG_COPY_WORDS])
        if len(gram) < LONG_COPY_WORDS:
            break
        if gram in source_grams:
            if run == 0:
                copied.append(list(gram))
            run += 1
            copied[-1].append(draft_words[index + LONG_COPY_WORDS - 1])
        else:
            run = 0
    for run_words in copied:
        counts["long_copy"][0] += 1
        found["long_copy"].append(
            DraftFlag(
                "long_copy",
                " ".join(run_words),
                f"{len(run_words)} words copied from the stored text outside a quote; "
                "paraphrase it or quote it with quotation marks.",
            )
        )


def _summary_quotes(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return ""
    quotes = []
    for value in summary.values():
        if isinstance(value, Mapping) and isinstance(value.get("quote"), str):
            quotes.append(str(value["quote"]))
    return "\n".join(quotes)


def _numbers(normalised: str) -> list[str]:
    # Compared without spaces or thousands separators, so "1,000" and "1000" are one number.
    return [match.replace(" ", "").replace(",", "") for match in _NUMBER.findall(normalised)]


def _quoted_spans(text: str) -> list[str]:
    spans = []
    for match in _QUOTE.finditer(text):
        spans.append(match.group(1) if match.group(1) is not None else match.group(2))
    return spans


def _links(text: str) -> list[str]:
    return [
        match.group(0).rstrip(".,;:!?)]'")
        for match in _URL.finditer(text)
        if match.group(0).rstrip(".,;:!?)]'")
    ]


def _pov_supports(stem: str, pov_sentences: list[str]) -> bool:
    pattern = dict(_EXPERIENCE_VERBS)[stem]
    return any(
        _FIRST_PERSON.search(candidate) is not None and pattern.search(candidate) is not None
        for candidate in pov_sentences
    )


def _url_key(url: str) -> str:
    # A trailing slash, a fragment and the scheme's case are not part of which page a link is.
    key = url.strip().split("#", 1)[0].rstrip("/")
    if "://" in key:
        scheme, rest = key.split("://", 1)
        return f"{scheme.lower()}://{rest.lower()}"
    return key.lower()


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE.split(text.strip()) if sentence.strip()]


def _first_sentence(text: str) -> str:
    sentences = _sentences(text)
    return sentences[0] if sentences else text.strip()


def _sentence_with(text: str, number: str) -> str:
    # The normalised sentence's own numbers are compared the same way the check compared them.
    for sentence in _sentences(text):
        if number in _numbers(_normalise(sentence)):
            return sentence
    return _first_sentence(text)


def _normalise(value: str) -> str:
    # Both sides of every comparison go through this. The model reads the source HTML-escaped, so
    # "R&D" reaches it as "R&amp;D" and comes back that way: unescaping both sides makes a faithful
    # quote match instead of failing the draft for a character the prompt introduced.
    folded = unicodedata.normalize("NFKC", html.unescape(value)).translate(_TRANSLATION)
    return " ".join(folded.split()).casefold()


__all__ = [
    "KIND_ORDER",
    "DraftCheck",
    "DraftFlag",
    "check_draft",
]
