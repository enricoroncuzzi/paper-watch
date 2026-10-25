"""The draft check: code, not a model, proves what a draft claims (Task 38).

Every number, quote and link in the draft must come from the stored title or text, a checked
summary quote, or the owner's own point of view. Every first-person sentence must be one of the
owner's own sentences, word for word. A lab post must name the lab, a preprint's numbers must be
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
# The prompt caps a quote at 25 words; a longer verbatim quote is copying, not quoting.
MAX_QUOTE_WORDS = 25
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
# A first person as a reader writes it, matched on text that went through `_fold`, so a full-width
# or italic "I" is found. "us" is lower case only, so "A US lab" stays a country, and only the
# "i" of "i.e." is not the pronoun. A contraction is found by its "I" or "we".
_FIRST_PERSON = re.compile(
    r"\b(?:i(?!\.e\b)|me|my|mine|myself|we|our|ours|ourselves)\b",
    re.IGNORECASE,
)
_LOWER_US = re.compile(r"\bus\b")
# A sentence ends after . ! ? (with any closing quote or bracket) or an ellipsis, and at a newline.
# A comma or a dash does not end one; neither does an abbreviation: a full stop before a lower-case
# letter or after a single capital initial does not end a sentence. Closing characters a model puts
# after the stop, including markdown bold, may be one or two in a row.
_CLOSERS = "\"')\\]\u201d\u2019*"
_SENTENCE_END = re.compile(
    rf"(?<![A-Z]\.)(?<=[.!?])\s+(?![a-z])"
    rf"|(?<=[.!?][{_CLOSERS}])\s+(?![a-z])"
    rf"|(?<=[.!?][{_CLOSERS}][{_CLOSERS}])\s+(?![a-z])"
    r"|(?<=\u2026)\s+"
    r"|\n"
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
# Words that do not name a lab on their own. A label made only of these ("Temporal", "Modal",
# "AI") needs the whole label directly before an attribution verb; a label with a distinctive
# token ("Arize AI", "Mistral AI", "Microsoft Agent Framework") is named by that token.
_LAB_STOP_WORDS = frozenset(
    {
        "the",
        "of",
        "and",
        "for",
        "inc",
        "ltd",
        "llc",
        "corp",
        "corporation",
        "company",
        "group",
        "team",
        "research",
        "labs",
        "lab",
    }
)
_LAB_COMMON_WORDS = frozenset(
    {
        "ai",
        "agent",
        "agents",
        "framework",
        "frameworks",
        "temporal",
        "modal",
        "model",
        "models",
        "data",
        "system",
        "systems",
        "platform",
        "platforms",
        "tool",
        "tools",
    }
)
# The verbs that turn a name into an attribution, spelled out so the common-label rule can
# require one directly after the label.
_ATTRIBUTION_WORDS = (
    "reports?|reported|says?|said|claims?|claimed|announces?|announced|writes?|wrote|"
    "describes?|described"
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
        "\u2010": "-",
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
    # The hashtag line is a label, not a sentence: a tag such as #gpt4 is not a numeric claim.
    body_text = "\n".join((draft.hook, draft.body))
    folded_pov = _normalise(pov)
    folded_store = _normalise(f"{source.title}\n{source.text}")
    quotes = _summary_quotes(source.summary)
    # A summary quote is checked against the stored text when it was written, so its numbers are
    # stored evidence as much as the text's own.
    evidence = set(_numbers(_normalise(f"{source.title}\n{source.text}\n{quotes}\n{pov}")))
    counts: dict[FlagKind, list[int]] = {kind: [0, 0] for kind in KIND_ORDER}
    found: dict[FlagKind, list[DraftFlag]] = {kind: [] for kind in KIND_ORDER}

    _check_numbers(body_text, evidence, counts, found)
    _check_quotes(text, folded_store, folded_pov, counts, found)
    _check_links(draft, text, source, folded_pov, counts, found)
    _check_first_person(text, pov, folded_store, counts, found)
    _check_attribution(text, source, counts, found)
    _check_preprint_numbers(body_text, source, counts, found)
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
        if _number_supported(number, evidence):
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
            words = len(folded.split())
            if words > MAX_QUOTE_WORDS:
                counts["long_copy"][0] += 1
                found["long_copy"].append(
                    DraftFlag(
                        "long_copy",
                        span,
                        f"quote is {words} words, the limit is {MAX_QUOTE_WORDS}",
                    )
                )
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
    counts["link"][0] += 1
    key = _url_key(draft.source_link)
    if key == _url_key(source.url):
        counts["link"][1] += 1
    else:
        flagged.add(key)
        detail = (
            f"The draft has no source link; it must be the item's own link: {source.url}"
            if not draft.source_link
            else f"The source link must be the item's own link: {source.url}"
        )
        found["link"].append(DraftFlag("link", draft.source_link, detail))
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
    pov: str,
    folded_store: str,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    """A first-person sentence passes only when the owner wrote that sentence, word for word.

    The owner's rule (2026-10-25) is a whole-sentence one: the model may copy the owner's
    sentences unchanged, as separate sentences, but may not trim one or blend it with other text.
    A sentence passes when its words equal one owner sentence or a run of neighbouring ones, since
    a line wrap in the point of view may split one sentence in two. A verbatim quote of the item is
    the authors' voice, so it does not make a sentence first person; the quote rule already checks
    it. Whether the owner wrote the sentence is decided on its own characters, quotes and links
    included, so a quote or link added to, dropped from or swapped in an owner sentence is flagged.
    """
    owner = [
        key
        for key in (
            _sentence_key(pov[start:end])
            for start, end in _piece_spans(_SENTENCE_END, _blanked(pov, folded_store))
        )
        if key
    ]
    blanked = _blanked(text, folded_store)
    for start, end in _piece_spans(_SENTENCE_END, blanked):
        if not _is_first_person(blanked[start:end]):
            continue
        counts["first_person"][0] += 1
        if _owner_wrote(_sentence_key(text[start:end]), owner):
            counts["first_person"][1] += 1
            continue
        found["first_person"].append(
            DraftFlag(
                "first_person",
                text[start:end],
                "The draft speaks in first person here, but your point of view does not say "
                "this word for word.",
            )
        )


def _piece_spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    """The trimmed (start, end) of each piece `pattern` splits the text into."""
    pieces: list[tuple[int, int]] = []
    at = 0
    for match in pattern.finditer(text):
        pieces.append((at, match.start()))
        at = match.end()
    pieces.append((at, len(text)))
    spans: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        while piece_start < piece_end and text[piece_start].isspace():
            piece_start += 1
        while piece_end > piece_start and text[piece_end - 1].isspace():
            piece_end -= 1
        if piece_start < piece_end:
            spans.append((piece_start, piece_end))
    return spans


def _is_first_person(sentence: str) -> bool:
    folded = _fold(sentence)
    return _FIRST_PERSON.search(folded) is not None or _LOWER_US.search(folded) is not None


def _fold(value: str) -> str:
    # The folding of `_normalise` without its case change: "US" and "us" are different words.
    return unicodedata.normalize("NFKC", html.unescape(value)).translate(_TRANSLATION)


def _sentence_key(value: str) -> str:
    """The words of a sentence: case, punctuation, whitespace and a list marker do not change it."""
    # A list marker is the writer's layout, not a word of the sentence.
    unmarked = re.sub(r"^(?:\d+|[a-z])[).]\s+", "", _normalise(value))
    return " ".join(re.sub(r"\W+", " ", unmarked).split())


def _owner_wrote(key: str, owner: list[str]) -> bool:
    """Whether the words are one of the owner's sentences or a run of neighbouring ones."""
    for start in range(len(owner)):
        run = ""
        for sentence in owner[start:]:
            run = f"{run} {sentence}".strip()
            if run == key:
                return True
            if not key.startswith(f"{run} "):
                break
    return False


def _blanked(text: str, folded_store: str) -> str:
    """Blank the spans whose first person is not the owner's: a verbatim quote and every URL.

    A quote of `MIN_QUOTE_WORDS` or more words that is in the stored text is the authors' voice,
    which the quote rule already checks; a URL is vetted by the link rule. A shorter span is not a
    quote the quote rule checks, so it must not hide a pronoun: `"We" shipped it` is still the
    owner speaking. Quotes are blanked before URLs, so a URL inside a quote does not stop the quote
    being verbatim. The closing punctuation stays and the blank keeps the length, so a flagged
    sentence is shown with the reader's own words and a sentence that ends in a URL still ends
    there.
    """

    def blank(match: re.Match[str]) -> str:
        url = match.group(0)
        link = url.rstrip(".,;:!?)]'")
        return " " * len(link) + url[len(link) :]

    def quote(match: re.Match[str]) -> str:
        whole = match.group(0)
        span = _normalise(match.group(1) if match.group(1) is not None else match.group(2))
        if len(span.split()) >= MIN_QUOTE_WORDS and span in folded_store:
            # The full stop and its closing quote stay, so the sentence still ends there.
            tail = whole[-2:] if whole[-2] in ".!?" else ""
            return " " * (len(whole) - len(tail)) + tail
        return whole

    return _URL.sub(blank, _QUOTE.sub(quote, text))


def _check_attribution(
    text: str,
    source: DraftSource,
    counts: dict[FlagKind, list[int]],
    found: dict[FlagKind, list[DraftFlag]],
) -> None:
    if source.tier != "lab_blog":
        return
    if not source.org or not source.org.strip():
        # Nothing to check against: a lab post without a stored organisation cannot be held to a
        # name, and inventing the requirement would flag every such draft.
        return
    counts["attribution"][0] = 1
    # A distinctive token of the label names the lab: "Google DeepMind" accepts "DeepMind
    # reports", and "OpenAI, Inc." accepts "OpenAI reports". A label made only of common words
    # ("Temporal", "Modal") would otherwise match "Temporal reasoning papers claim ...", so it
    # needs the whole label directly before an attribution verb.
    tokens = re.findall(r"[a-z0-9]+", _normalise(source.org))
    if not any(token not in _LAB_STOP_WORDS for token in tokens):
        # A label made only of function words ("The") names no lab.
        return
    names = [
        token for token in tokens if token not in _LAB_STOP_WORDS and token not in _LAB_COMMON_WORDS
    ]
    label = " ".join(tokens)
    for sentence in _sentences(text):
        folded = _normalise(sentence)
        if names:
            named = any(
                re.search(rf"(?<!\w){re.escape(name)}(?!\w)", folded) is not None for name in names
            )
        else:
            named = (
                re.search(rf"(?<!\w){re.escape(label)}\s+(?:{_ATTRIBUTION_WORDS})\b", folded)
                is not None
            )
        if named and _ATTRIBUTION_VERB.search(folded) is not None:
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


def _number_supported(number: str, evidence: set[str]) -> bool:
    # The percent sign belongs to a number only when the draft claims it: a source's "41%" backs
    # "41", while a draft's "41%" is not backed by a source that only says "41".
    if number in evidence:
        return True
    return not number.endswith("%") and f"{number}%" in evidence


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
