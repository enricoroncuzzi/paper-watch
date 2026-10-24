"""Chat models on a free OpenAI-compatible API, behind a daily call cap and a token ledger.

OpenRouter (and Groq as an optional fallback) speak the request shape of OpenAI's chat
completions API, so one client - LangChain's `ChatOpenAI` - talks to either by changing a base
URL, a key and a model id (ADR 0006). Free tiers are withdrawn or rate-limited without notice, so
the provider is a configuration value and a second provider can answer when the first fails.

Every call goes through `CallGuard`: a `chat_calls` row is reserved before the provider is asked
(the count and the insert under one advisory transaction lock, so two answers at 799 cannot both
pass), and that row is completed with the provider's real usage after the call, or marked failed
when the attempt raised. The cap is counted from the table rather than from memory, so restarting
the process cannot forget calls already made today.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import Field

from paperwatch.config import (
    CHAT_CALL_MODEL_MAX_CHARS,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    Settings,
)
from paperwatch.llm.ledger import Ledger

log = structlog.get_logger()


class ChatNotConfigured(RuntimeError):
    """No chat provider has a key: the feature cannot start, and says which variable is missing."""


class DailyCapReached(RuntimeError):
    """The free tier's request cap is used up for today; retrying cannot help before midnight."""


class ModelUnavailable(RuntimeError):
    """Every configured model failed, so no provider can produce the answer.

    Defined here with the other chat errors, but raised by the agent graph (Task 16): only it
    knows whether a fallback provider is left to try.
    """


@dataclass(frozen=True)
class ChatModel:
    """One configured provider: its name, for logs and the usage record, and the model itself."""

    provider: str
    model: BaseChatModel


class ChatCallStore(Protocol):
    """The `chat_calls` operations the guard needs, so the store can be swapped.

    `Repository` satisfies it for tests that already hold a session; `ChatCallLog` is the
    production one, and commits each row on its own so the cap survives a request's rollback.
    """

    def reserve_chat_call(
        self, *, at: datetime, provider: str, cap: int
    ) -> tuple[int | None, int]: ...

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None: ...

    def fail_chat_call(self, call_id: int) -> None: ...


def make_chat_models(settings: Settings) -> list[ChatModel]:
    """Build the provider chain, primary first: OpenRouter, then Groq when fully configured."""
    if not settings.openrouter_api_key:
        # Named as the env var, not the field: the reader has to know what to put in .env.
        raise ChatNotConfigured("PW_OPENROUTER_API_KEY is not set, so no chat model can be built")
    models = [
        ChatModel(
            "openrouter",
            # The OpenAI SDK and LangChain also read OPENAI_ORG_ID (or its alias
            # OPENAI_ORGANIZATION), OPENAI_PROJECT_ID, OPENAI_PROXY and OPENAI_CUSTOM_HEADERS
            # (whose Authorization line replaces the API key passed here), so an OpenAI setup in
            # the same shell would reach these providers. OPENAI_BASE_URL, OPENAI_API_BASE and
            # OPENAI_API_KEY do not apply because this code always passes base_url and api_key.
            # Accepted: nothing here sets them; a stray credential or header would go to the
            # provider.
            ChatOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=settings.openrouter_api_key.get_secret_value(),
                model=settings.chat_model,
                timeout=settings.chat_timeout_s,
                # One response's output is capped: the timeout applies per read, not to a whole
                # stream, so a generation that never stalls is otherwise bounded only by the
                # context limit. langchain-openai sends this as `max_completion_tokens`.
                max_tokens=settings.chat_max_output_tokens,
                # No SDK retries: a retry inside the client is a request the daily cap never sees.
                # The agent's fallback loop (Task 16) owns retrying, so every attempt is counted.
                max_retries=0,
                # Ask for token usage on streamed responses too: without it a streamed answer
                # reports no usage, and the ledger would charge nothing for it. langchain-openai
                # leaves this off for a custom base URL, so it is required here, not redundant.
                stream_usage=True,
                # OpenRouter's app attribution headers, which put our app on its usage dashboard.
                default_headers={
                    "HTTP-Referer": "https://github.com/enricoroncuzzi/paper-watch",
                    "X-Title": "Paper Watch",
                },
            ),
        )
    ]
    # Groq answers only when both the key and a model id are set: its free model list changes too
    # often to pin a default here, and a guessed id would fail on every fallback attempt.
    if settings.groq_api_key and settings.groq_chat_model:
        models.append(
            ChatModel(
                "groq",
                ChatOpenAI(
                    base_url=GROQ_BASE_URL,
                    api_key=settings.groq_api_key.get_secret_value(),
                    model=settings.groq_chat_model,
                    timeout=settings.chat_timeout_s,
                    max_tokens=settings.chat_max_output_tokens,
                    max_retries=0,
                    stream_usage=True,
                ),
            )
        )
    return models


class CallGuard:
    """The two free-tier limits around one chat call: requests per day, and tokens per answer.

    `before_call` reserves a row and refuses once the day's cap is spent; `after_call` fills that
    row with the real usage and charges the per-answer ledger; `after_failure` marks it failed,
    because an attempt that reached the provider was spent even when it produced no answer.
    """

    def __init__(
        self,
        store: ChatCallStore,
        ledger: Ledger,
        max_calls_per_day: int,
        clock: Callable[[], datetime],
    ) -> None:
        # The guard is deliberately narrow: it reserves, fills and fails one row, so tests pass a
        # small in-memory stand-in for the store.
        self._store = store
        self._ledger = ledger
        self._max_calls_per_day = max_calls_per_day
        # The clock is injected so a test can place a call on either side of UTC midnight.
        self._clock = clock

    def before_call(self, provider: str) -> int:
        """Reserve this attempt against the daily cap and return the row's id.

        The store does the count and the insert under one transaction lock. Reserving before the
        provider is what makes concurrent answers safe: both would read "799" and both would
        reach the provider without it.
        """
        now = self._clock()
        # A naive clock would be read as local time by astimezone and silently shift the day.
        # A raise, not an assert: `python -O` strips asserts, and this guard must hold.
        if now.tzinfo is None:
            raise ValueError("the clock must return an aware datetime")
        # The cap is per UTC day: the store counts from this day's midnight, so it resets without
        # a job. The lock is transaction-scoped, held until the reservation commits.
        call_id, calls_today = self._store.reserve_chat_call(
            at=now, provider=provider, cap=self._max_calls_per_day
        )
        if call_id is None:
            midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            raise DailyCapReached(
                f"{calls_today} chat calls since {midnight.isoformat()}, "
                f"cap is {self._max_calls_per_day}"
            )
        return call_id

    def after_failure(self, provider: str, call_id: int) -> None:
        # An attempt that reached the provider is a request against the free tier even when it
        # raised, and a failure is exactly when the next request is most likely to be refused.
        # The reserved row stays and is marked failed, so it cannot be read as a real answer.
        self._store.fail_chat_call(call_id)

    def after_call(self, provider: str, message: AIMessage, call_id: int) -> None:
        usage = message.usage_metadata
        if usage is None:
            # A provider that reports no usage is a bug on its side, not a free call: charge zero
            # but still complete the row, so the daily cap stays honest.
            log.warning("chat_usage_missing", provider=provider)
            input_tokens = output_tokens = 0
        else:
            input_tokens = int(usage["input_tokens"])
            output_tokens = int(usage["output_tokens"])
        # The model id the provider says it used, which may differ from the one requested when a
        # provider routes to a variant; "unknown" keeps the record complete either way.
        model = _single_name(str(message.response_metadata.get("model_name") or "unknown"))
        # Fill before charging: the tokens were spent even when the ledger then refuses the
        # answer, so the row must survive the BudgetExceeded.
        self._store.fill_chat_call(
            call_id, model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        self._ledger.charge(model, input_tokens + output_tokens)


def _single_name(name: str) -> str:
    # langchain-core merges streamed chunks by concatenating string metadata, so a provider that
    # repeats its model id on several chunks leaves "idid". The smallest repeating unit is the id.
    # The clip is the `chat_calls.model` width: whatever a provider sends, the write still fits.
    return name[: (name + name).find(name, 1)][:CHAT_CALL_MODEL_MAX_CHARS]


def _word_chunks(
    content: object, response_metadata: dict[str, Any] | None = None
) -> list[AIMessageChunk]:
    # Word by word, so a test sees a real token stream rather than one blob.
    words = str(content).split(" ")
    return [
        AIMessageChunk(
            content=w if i == 0 else f" {w}", response_metadata=dict(response_metadata or {})
        )
        for i, w in enumerate(words)
    ]


class ScriptedChatModel(BaseChatModel):
    """Offline stand-in for a chat model: replays a script, records what it was asked."""

    # Exception first, not AIMessage first: pydantic tries the members in order, and AIMessage's
    # own validator crashes with an AttributeError on a non-dict instead of declining the value.
    script: list[Exception | AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    # Tool names bound for each call, [] for an unbound call: trajectory tests read it.
    bound_tools: list[list[str]] = Field(default_factory=list)
    # The binding keywords in force for each call, `parallel_tool_calls` and `tool_choice`
    # included, so a test can tell which binding served which model call without a live model.
    call_kwargs: list[dict[str, Any]] = Field(default_factory=list)
    # When set, every streamed chunk carries this id in response_metadata, the way some providers
    # name their model on each chunk; langchain-core's merge then concatenates the repeats.
    stream_model_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        # Like ChatOpenAI, binding returns a wrapper that passes the tools on every call.
        return self.bind(tools=[t.name for t in tools], **kwargs)

    def _next(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> AIMessage:
        # Each call consumes one script step, so a test proves the graph asked exactly as often
        # as it should; an empty script fails loudly instead of returning a made-up answer.
        self.seen.append(list(messages))
        # `tools` is recorded as names because bind_tools sends names; the rest of the binding
        # keywords (`parallel_tool_calls`, `tool_choice`) identify the binding for this call.
        self.bound_tools.append(kwargs.get("tools") or [])
        self.call_kwargs.append({k: v for k, v in kwargs.items() if k != "tools"})
        if not self.script:
            raise AssertionError("script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._next(messages, kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._next(messages, kwargs)
        # Some providers (OpenRouter among them) name the model on more than one chunk; the option
        # reproduces that so a test can prove the guard stores the id once.
        meta = {"model_name": self.stream_model_name} if self.stream_model_name else {}
        chunks: list[AIMessageChunk] = []
        # A provider streams text before the call it decides to make, and invoke() keeps that
        # text; replaying it first keeps the stream and the invoke in agreement.
        if msg.content:
            chunks.extend(_word_chunks(msg.content, meta))
        if msg.tool_calls:
            # A tool call streams as chunks that reassemble into the same call, which is how a
            # real provider delivers it and how LangGraph collects it.
            chunks.append(
                AIMessageChunk(
                    content="",
                    response_metadata=dict(meta),
                    tool_call_chunks=[
                        {
                            "name": c["name"],
                            "args": json.dumps(c["args"]),
                            "id": c["id"],
                            "index": i,
                        }
                        for i, c in enumerate(msg.tool_calls)
                    ],
                )
            )
        # Usage arrives last, as it does from a provider, so the guard charges it after the text.
        # `chunk_position="last"` marks the end of the stream: LangChain otherwise appends its own
        # empty final chunk, which carries no usage and would become the caller's last chunk.
        chunks.append(
            AIMessageChunk(
                content="",
                usage_metadata=msg.usage_metadata,
                response_metadata={**meta, **msg.response_metadata},
                chunk_position="last",
            )
        )
        for chunk in chunks:
            gen = ChatGenerationChunk(message=chunk)
            if run_manager:
                # How LangGraph sees each token as it is produced.
                run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen
