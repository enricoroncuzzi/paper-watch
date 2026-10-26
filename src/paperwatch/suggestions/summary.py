"""30-second summaries whose quotes, numbers and names are checked by code (Task 35).

A model's summary reads true whether or not it is. So the model must hand back, for the claims that
matter, the exact sentence it took them from, and code proves that sentence is in the stored text,
that every number and name it used is there too, and that it added no superlative. Opinion is only
allowed in fields the page labels as interpretation. When the checks fail twice, the card shows the
abstract's own first sentences instead: a model failure must never put an invented claim on a card.
"""

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from paperwatch.links import bare_domains
from paperwatch.llm.chat import CallGuard, ChatModel, ModelUnavailable
from paperwatch.suggestions.prompts import (
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    summary_prompt,
)

log = structlog.get_logger()

# The four fields a summary has, in card order. Only the first two must be quoted, because only
# they state facts; the other two are labelled as interpretation wherever they are shown.
FIELD_NAMES = ("what", "result", "why", "catch")
QUOTED_FIELDS = ("what", "result")
INTERPRETATION_FIELDS = ("why", "catch")
# The one sentence a catch may be when the source states no limitation.
NOT_STATED = "Not stated in the abstract."
MAX_FIELD_WORDS = 35
# A quote shorter than this backs nothing a reader can check; it is a single word repeated.
MIN_QUOTE_CHARS = 20
MAX_TOTAL_WORDS = 110
EXCERPT_SENTENCES = 2
EXCERPT_MAX_WORDS = 60
# Words a summary may only use when the source does too: a claim of novelty is the one a reader
# would quote back at us, so the source has to make it first.
SUPERLATIVES = (
    "state-of-the-art",
    "breakthrough",
    "revolutionary",
    "first",
    "unprecedented",
    "outperforms all",
    "groundbreaking",
)
# A number with an optional decimal or thousands part and an optional percent sign.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s?%?")
# A URL, for the fields that are the model's own reading: a link there, or a bare domain, is a claim
# no stored text backs, and the source's own links live in the quoted fields.
_URL = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
# A sentence boundary: the whitespace after terminal punctuation.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# A word as a name check sees it: letters, digits and the punctuation real names carry.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.'\-]*")
_HYPHENATED_LETTERS = re.compile(r"[A-Za-z]-[A-Za-z]")
# A character that continues a token: the hyphen joins words for matching, so "first-order" is one.
_TOKEN_CHAR = re.compile(r"[\w-]")
# Typographic variants a model produces that are the same character to a reader. Written as escapes
# so this file contains no dash character of its own.
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
class SummaryField:
    text: str
    # The source sentence the text rests on, verbatim. None only for the interpretation fields.
    quote: str | None
    # True for `why` and `catch`: the page labels these as interpretation, not the authors' words.
    interpretation: bool


@dataclass(frozen=True)
class Summary:
    # "checked" passed every check; "excerpt" is the stored text's own first sentences.
    status: Literal["checked", "excerpt"]
    what: SummaryField | None
    result: SummaryField | None
    why: SummaryField | None
    catch: SummaryField | None
    excerpt: str
    # Named checks that failed, kept so a reviewer can see why a card fell back to the excerpt.
    failed_checks: tuple[str, ...]


def check_summary(raw: Mapping[str, Any], *, title: str, text: str) -> list[str]:
    """Every check a summary has to pass, as a list of names; empty means it is checked."""
    fields = _shape(raw)
    if fields is None:
        return ["shape"]
    source = _normalise(f"{title}\n{text}")
    source_numbers = _numbers(source)
    failed: list[str] = []
    for name in FIELD_NAMES:
        field = fields[name]
        body = _normalise(field["text"])
        quote = field.get("quote")
        if quote:
            folded_quote = _normalise(quote)
            if name in QUOTED_FIELDS and len(folded_quote) < MIN_QUOTE_CHARS:
                failed.append(f"quote_too_short:{name}")
            elif folded_quote not in source:
                failed.append(f"quote_not_in_text:{name}")
        elif name in QUOTED_FIELDS:
            failed.append(f"quote_missing:{name}")
        failed += [f"number:{name}:{n}" for n in sorted(_numbers(body) - source_numbers)]
        failed += [f"name:{name}:{t}" for t in _unknown_names(field["text"], source)]
        failed += [
            f"superlative:{name}:{word}"
            for word in SUPERLATIVES
            if _superlative_is_unbacked(body, source, word)
        ]
        if name in INTERPRETATION_FIELDS and (_URL.search(body) or bare_domains(body)):
            # The check cannot follow a link, and `why` and `catch` are where the model's own
            # reading goes: a URL there would put an unverifiable claim on the card.
            failed.append(f"url:{name}")
        if len(field["text"].split()) > MAX_FIELD_WORDS:
            failed.append(f"length:{name}")
    total = sum(len(fields[name]["text"].split()) for name in FIELD_NAMES)
    if total > MAX_TOTAL_WORDS:
        failed.append("length:total")
    return failed


def excerpt(text: str) -> str:
    """The source's own first two sentences, so an unchecked card still says something true."""
    sentences = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    joined = " ".join(sentences[:EXCERPT_SENTENCES])
    words = joined.split()
    if len(words) > EXCERPT_MAX_WORDS:
        # An ellipsis marks the cut; a reader must never read a truncated abstract as the whole.
        return " ".join(words[:EXCERPT_MAX_WORDS]) + "..."
    return joined


def excerpt_summary(text: str, failed_checks: Sequence[str]) -> Summary:
    """The honest fallback card: the source's own excerpt and the checks that sent it there."""
    return Summary(
        status="excerpt",
        what=None,
        result=None,
        why=None,
        catch=None,
        excerpt=excerpt(text),
        failed_checks=tuple(failed_checks),
    )


def text_hash(title: str, text: str) -> str:
    """The fingerprint of what a summary was written from, so a replaced abstract regenerates it."""
    return hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()


def summary_fields(summary: Summary) -> dict[str, object]:
    """The JSON stored in `item_summaries.fields`: the four fields, or the excerpt."""
    if summary.status == "excerpt":
        return {"excerpt": summary.excerpt}
    fields: dict[str, object] = {}
    for name in FIELD_NAMES:
        field = getattr(summary, name)
        fields[name] = {
            "text": field.text,
            "quote": field.quote,
            "interpretation": field.interpretation,
        }
    return fields


class SummaryWriter:
    """One summary per item, through the same call guard and provider chain the agent uses."""

    def __init__(self, models: list[ChatModel], guard_factory: Callable[[], CallGuard]) -> None:
        self._models = models
        # A factory, not one guard: an item is summarised later, in another thread, so the guard's
        # ledger belongs to that call and not to the request that queued it.
        self._guard_factory = guard_factory

    def write(self, *, title: str, text: str) -> Summary:
        # One retry that names the failed checks, then the honest fallback. A cap or a budget
        # decision is not a check failure and propagates to the caller, which decides when to retry.
        failed: tuple[str, ...] = ()
        for attempt in range(2):
            raw = self._ask(title, text, failed)
            if raw is None:
                # Unparseable JSON is its own failure: the retry asks for JSON again.
                failed = ("shape",)
                continue
            checks = check_summary(raw, title=title, text=text)
            if not checks:
                return _checked(raw)
            failed = tuple(checks)
            log.info("summary_check_failed", attempt=attempt + 1, checks=list(checks))
        return excerpt_summary(text, failed)

    def _ask(self, title: str, text: str, failed: Sequence[str]) -> Mapping[str, Any] | None:
        messages: list[BaseMessage] = [
            SystemMessage(SUMMARY_SYSTEM_PROMPT),
            HumanMessage(summary_prompt(title, text, failed)),
        ]
        return _parse(self._invoke(messages))

    def _invoke(self, messages: list[BaseMessage]) -> str:
        guard = self._guard_factory()
        failures: list[tuple[str, Exception]] = []
        for chat in self._models:
            # Outside the try on purpose: a cap or budget decision must not look like provider
            # trouble, and another provider would spend the same budget or bypass the same cap.
            reservation = guard.before_call(chat.provider)
            try:
                message = chat.model.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - any provider error moves to the next model
                try:
                    guard.after_failure(chat.provider, reservation)
                except Exception as store_exc:  # noqa: BLE001 - never hide the provider error
                    log.warning(
                        "chat_failure_record_failed",
                        provider=chat.provider,
                        error_type=type(store_exc).__name__,
                    )
                # Only the error's shape: a provider's message can echo the source text.
                log.warning(
                    "summary_model_failed",
                    provider=chat.provider,
                    error_type=type(exc).__name__,
                )
                failures.append((chat.provider, exc))
                continue
            guard.after_call(chat.provider, message, reservation)
            return str(message.content)
        attempts = ", ".join(f"{provider} ({type(exc).__name__})" for provider, exc in failures)
        raise ModelUnavailable(f"no chat model could write the summary: {attempts}")


def _shape(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    # Shape first: every later check reads these keys, and a model that answered with prose has to
    # be told to answer with JSON rather than with a missing quote.
    if not isinstance(raw, Mapping):
        return None
    fields: dict[str, dict[str, Any]] = {}
    for name in FIELD_NAMES:
        value = raw.get(name)
        if not isinstance(value, Mapping):
            return None
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        quote = value.get("quote")
        if quote is not None and not isinstance(quote, str):
            return None
        fields[name] = {"text": text, "quote": quote}
    return fields


def _checked(raw: Mapping[str, Any]) -> Summary:
    fields = _shape(raw)
    assert fields is not None  # the caller only gets here after check_summary passed
    return Summary(
        status="checked",
        what=_field(fields, "what"),
        result=_field(fields, "result"),
        why=_field(fields, "why"),
        catch=_field(fields, "catch"),
        excerpt="",
        failed_checks=(),
    )


def _field(fields: Mapping[str, Mapping[str, Any]], name: str) -> SummaryField:
    # The model read the source HTML-escaped, so unescape before storing: the card prints this
    # under "quotes from the stored text" and must not show "R&amp;D".
    quote = fields[name].get("quote")
    return SummaryField(
        text=html.unescape(str(fields[name]["text"]).strip()),
        quote=html.unescape(quote) if quote else None,
        interpretation=name in INTERPRETATION_FIELDS,
    )


def _parse(content: str) -> Mapping[str, Any] | None:
    body = content.strip()
    # Some models wrap JSON in a fence even when told not to; stripping it is cheaper than a retry.
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _normalise(value: str) -> str:
    # Both sides of every comparison go through this. The model reads the source HTML-escaped, so
    # "R&D" reaches it as "R&amp;D" and comes back that way: unescaping both sides makes a faithful
    # quote match instead of failing the card for a character the prompt introduced.
    folded = unicodedata.normalize("NFKC", html.unescape(value)).translate(_TRANSLATION)
    return " ".join(folded.split()).casefold()


def _numbers(normalised: str) -> set[str]:
    # Compared without spaces or thousands separators, so "1,000" and "1000" are one number.
    return {match.replace(" ", "").replace(",", "") for match in _NUMBER.findall(normalised)}


def _unknown_names(text: str, source: str) -> list[str]:
    unknown = []
    for sentence in _SENTENCE.split(text):
        for index, raw_token in enumerate(_TOKEN.findall(sentence)):
            # A sentence's own full stop is not part of a name: "GPT-5." reports as "GPT-5".
            token = raw_token.rstrip(".'")
            if not token or not _name_like(token, first=index == 0):
                continue
            # A whole word, not a substring: an invented "Bert" must not pass because the source
            # happens to contain "Albert".
            if re.search(rf"(?<!\w){re.escape(_normalise(token))}(?!\w)", source) is None:
                unknown.append(token)
    return unknown


def _superlative_is_unbacked(body: str, source: str, word: str) -> bool:
    """True when the summary uses the superlative and the source does not carry it."""
    # Detection is loose, so a compound cannot hide the claim: "first-ever" is still a "first".
    tokens = _hyphen_tokens(body, word)
    if not tokens:
        return False
    # Acceptance is strict: the source needs the bare word, or the summary's own compound or
    # plural, as a whole token. A hyphen joins words, so "first-order" never backs "the first
    # method".
    if _whole(source, word):
        return False
    return not all(_whole(source, token) for token in tokens)


def _hyphen_tokens(normalised: str, word: str) -> set[str]:
    # The hyphenated token each use of the word sits in, so "first-order" can be looked up whole.
    # A trailing plural is the same claim, so "breakthroughs" expands to its own whole token.
    tokens = set()
    for match in re.finditer(rf"(?<!\w){re.escape(word)}(?:s)?(?!\w)", normalised):
        start, end = match.span()
        while start and _TOKEN_CHAR.match(normalised[start - 1]):
            start -= 1
        while end < len(normalised) and _TOKEN_CHAR.match(normalised[end]):
            end += 1
        tokens.add(normalised[start:end])
    return tokens


def _whole(source: str, phrase: str) -> bool:
    # Letters, digits, underscore and hyphen all count as part of the word on both sides.
    return re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", source) is not None


def _name_like(token: str, *, first: bool) -> bool:
    # A token only a model could have supplied: it holds a digit, a capital inside the word, a
    # hyphen between letters, or a capital where a sentence does not demand one.
    if any(character.isdigit() for character in token):
        return True
    if any(character.isupper() for character in token[1:]):
        return True
    if _HYPHENATED_LETTERS.search(token):
        return True
    return not first and token[:1].isupper()


__all__ = [
    "NOT_STATED",
    "SUMMARY_PROMPT_VERSION",
    "Summary",
    "SummaryField",
    "SummaryWriter",
    "check_summary",
    "excerpt",
    "excerpt_summary",
    "summary_fields",
    "text_hash",
]
