"""One chat call that turns a stored item and the owner's point of view into a draft.

The chain is the draft chain from `llm/chat.py`: OpenRouter with `provider.data_collection: "deny"`
and no fallback. This module makes exactly one call per draft and no retry, because a second call
would spend the day's budget on a reply the check has already been able to judge. A reply that is
not the JSON the prompt asked for is `DraftUnusable`, and nothing is saved.

Nothing here logs the point of view, the prompt or the draft text: a log line names the item, the
model, the flag counts and the timing only.
"""

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from paperwatch.drafts.prompts import (
    DRAFT_PROMPT_VERSION,
    DRAFT_SYSTEM_PROMPT,
    EARLY_NOTE,
    draft_prompt,
)
from paperwatch.llm.chat import CallGuard, ChatModel, ModelUnavailable

log = structlog.get_logger()

# The `drafts.model` column width. A provider may name a routed variant longer than the one asked
# for, and a usage write that does not fit would lose the whole draft.
DRAFT_MODEL_MAX_CHARS = 128
# The one sentence a provider refusal becomes: the client gets a reason it can show, and the
# provider's own error text (which can echo the prompt) never reaches a log or a response. It names
# the setting that fixes the likely cause: the shared chat model's only endpoint may refuse
# training, so the draft model needs one that does not (PW_DRAFT_MODEL).
NO_COMPLIANT_MODEL = (
    "No endpoint that refuses training on inputs serves the draft model, so the draft was not "
    "generated. Set PW_DRAFT_MODEL to a model that has one."
)
# A model may wrap JSON in a fence even when told not to, so one fence is stripped before parsing.
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n")


class DraftUnusable(RuntimeError):
    """The reply was not the JSON the prompt asked for, so there is no draft to save."""


@dataclass(frozen=True)
class DraftSource:
    """Everything one draft is written from, gathered from the store by the route or the CLI."""

    item_id: UUID
    title: str
    url: str
    source: str
    tier: str
    basis_label: str
    venue: str | None
    org: str | None
    text: str
    github_repo: str | None
    project_page: str | None
    # The stored summary's fields, or None. It is context for the model, never a source of new
    # facts: a summary's quotes are checked against the text, so they add no claim of their own.
    summary: Mapping[str, Any] | None
    early: bool


@dataclass(frozen=True)
class DraftText:
    """The model's reply after parsing, exactly as it will be saved and checked."""

    hook: str
    body: str
    hashtags: tuple[str, ...]
    source_link: str


def generate_draft(
    models: Sequence[ChatModel],
    guard: CallGuard,
    source: DraftSource,
    *,
    pov: str,
    language: str,
) -> tuple[DraftText, str]:
    """Make the one call and return the draft plus the model id that answered.

    The cap and the per-answer budget are the guard's, the same ones an answer uses; a refusal or
    a cap decision propagates so the route can say what happened.
    """
    if not models:
        raise ModelUnavailable(NO_COMPLIANT_MODEL)
    # One model, and only the first: the draft chain is built without a fallback on purpose.
    model = models[0]
    # Outside the try: a cap decision is not provider trouble, and another provider would spend
    # the same budget.
    reservation = guard.before_call(model.provider)
    started = time.monotonic()
    try:
        message = model.model.invoke(draft_messages(source, pov, language))
    except Exception as exc:  # noqa: BLE001 - any provider error means no compliant answer
        try:
            guard.after_failure(model.provider, reservation)
        except Exception as store_exc:  # noqa: BLE001 - never hide the provider error
            log.warning(
                "chat_failure_record_failed",
                provider=model.provider,
                error_type=type(store_exc).__name__,
            )
        # The error's type only: a provider's message can echo the point of view back.
        log.warning(
            "draft_model_failed",
            item_id=str(source.item_id),
            provider=model.provider,
            error_type=type(exc).__name__,
        )
        raise ModelUnavailable(NO_COMPLIANT_MODEL) from None
    guard.after_call(model.provider, message, reservation)
    answered = _answered_model(message, model)
    parsed = _parse(str(message.content))
    if parsed is None:
        # The reply has no shape to check, so it is not a draft: nothing is saved and the reader
        # can try again without a half-written draft in Drafts.
        log.info(
            "draft_unusable",
            item_id=str(source.item_id),
            model=answered,
            ms=_ms(started),
        )
        raise DraftUnusable("the reply was not the JSON draft structure the prompt asked for")
    log.info(
        "draft_generated",
        item_id=str(source.item_id),
        model=answered,
        chars=len(draft_text(parsed)),
        ms=_ms(started),
    )
    return parsed, answered


def draft_text(draft: DraftText) -> str:
    """The "Copy for LinkedIn" text: hook, body, hashtags, link, plain text only.

    `web/src/lib/linkedin-text.ts` builds the same string for the clipboard; the route's `chars`
    comes from here, so the count on a card matches what a copy puts on the clipboard.
    """
    parts = [draft.hook, "", draft.body]
    tags = " ".join(f"#{tag}" for tag in draft.hashtags)
    if tags:
        parts += ["", tags]
    if draft.source_link:
        parts += ["", draft.source_link]
    return "\n".join(parts).strip()


def draft_messages(source: DraftSource, pov: str, language: str) -> list[BaseMessage]:
    meta = [
        f"requested language: {language}",
        f"tier: {source.tier}",
        f"basis: {source.basis_label}",
        f"item link: {source.url}",
    ]
    if source.venue:
        meta.append(f"venue: {source.venue}")
    if source.org:
        meta.append(f"organisation: {source.org}")
    if source.github_repo:
        meta.append(f"code: {source.github_repo}")
    if source.project_page:
        meta.append(f"project page: {source.project_page}")
    if source.early:
        meta.append(EARLY_NOTE)
    return [
        SystemMessage(DRAFT_SYSTEM_PROMPT),
        HumanMessage(
            draft_prompt(
                source.title,
                source.text,
                meta="\n".join(meta),
                summary=source.summary,
                pov=pov,
            )
        ),
    ]


def _parse(content: str) -> DraftText | None:
    """The first balanced JSON object in the reply, or None when there is no usable one."""
    body = content.strip()
    if body.startswith("```"):
        body = _FENCE.sub("", body).removesuffix("```").strip()
    start = body.find("{")
    while start != -1:
        candidate = _balanced_object(body, start)
        if candidate is not None:
            try:
                raw = json.loads(candidate)
            except json.JSONDecodeError:
                raw = None
            if isinstance(raw, Mapping):
                parsed = _structure(raw)
                if parsed is not None:
                    return parsed
        start = body.find("{", start + 1)
    return None


def _balanced_object(text: str, start: int) -> str | None:
    # Tracks strings and escapes so a brace inside the draft's own text does not end the object.
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _structure(raw: Mapping[str, Any]) -> DraftText | None:
    hook = raw.get("hook")
    body = raw.get("body")
    hashtags = raw.get("hashtags")
    source_link = raw.get("source_link")
    if not isinstance(hook, str) or not hook.strip():
        return None
    if not isinstance(body, str) or not body.strip():
        return None
    if not isinstance(hashtags, list) or any(not isinstance(tag, str) for tag in hashtags):
        return None
    if not isinstance(source_link, str):
        return None
    return DraftText(
        hook=hook.strip(),
        body=body.strip(),
        # The prompt asks for tags without the hash sign and without spaces; a model that adds
        # either is normalised here so the page and the clipboard never show "##ai agents".
        hashtags=tuple(
            tag.strip().lstrip("#").replace(" ", "") for tag in hashtags if tag.strip().lstrip("#")
        ),
        source_link=source_link.strip(),
    )


def _answered_model(message: AIMessage, model: ChatModel) -> str:
    configured = str(getattr(model.model, "model_name", "") or "unknown")
    name = str(message.response_metadata.get("model_name") or configured)
    return name[:DRAFT_MODEL_MAX_CHARS]


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


__all__ = [
    "DRAFT_PROMPT_VERSION",
    "DraftSource",
    "DraftText",
    "DraftUnusable",
    "draft_text",
    "generate_draft",
]
