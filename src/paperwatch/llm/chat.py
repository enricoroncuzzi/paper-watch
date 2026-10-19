"""Chat models on a free OpenAI-compatible API, behind a daily call cap and a token ledger.

OpenRouter (and Groq as an optional fallback) speak the request shape of OpenAI's chat
completions API, so one client - LangChain's `ChatOpenAI` - talks to either by changing a base
URL, a key and a model id (ADR 0006). Free tiers are withdrawn or rate-limited without notice, so
the provider is a configuration value and a second provider can answer when the first fails.

Every call goes through `CallGuard`: the daily request cap is checked before the call and the
provider's real token usage is recorded after it. The cap is counted from the `chat_calls` table
rather than from memory, so restarting the process cannot forget calls already made today.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import Field

from paperwatch.config import GROQ_BASE_URL, OPENROUTER_BASE_URL, Settings
from paperwatch.llm.ledger import Ledger

if TYPE_CHECKING:
    # Type-only: the guard needs two repository methods, and importing the store at runtime would
    # pull SQLAlchemy into every module that touches a chat model, including the offline tests.
    from paperwatch.store.repository import Repository

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


def make_chat_models(settings: Settings) -> list[ChatModel]:
    """Build the provider chain, primary first: OpenRouter, then Groq when fully configured."""
    if not settings.openrouter_api_key:
        # Named as the env var, not the field: the reader has to know what to put in .env.
        raise ChatNotConfigured("PW_OPENROUTER_API_KEY is not set, so no chat model can be built")
    models = [
        ChatModel(
            "openrouter",
            ChatOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=settings.openrouter_api_key,
                model=settings.chat_model,
                timeout=settings.chat_timeout_s,
                max_retries=settings.chat_max_retries,
                # Ask for token usage on streamed responses too: without it a streamed answer
                # reports no usage, and the ledger would charge nothing for it.
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
                    api_key=settings.groq_api_key,
                    model=settings.groq_chat_model,
                    timeout=settings.chat_timeout_s,
                    max_retries=settings.chat_max_retries,
                    stream_usage=True,
                ),
            )
        )
    return models


class CallGuard:
    """The two free-tier limits around one chat call: requests per day, and tokens per answer.

    `before_call` runs first and refuses once the day's request cap is spent; `after_call` runs
    once the provider answered, records the real usage and charges the per-answer ledger.
    """

    def __init__(
        self,
        repo: "Repository",
        ledger: Ledger,
        max_calls_per_day: int,
        clock: Callable[[], datetime],
    ) -> None:
        # The guard is deliberately narrow: it reads only `chat_calls_since` and writes only
        # `record_chat_call`, so tests pass a small in-memory stand-in for the repository.
        self._repo = repo
        self._ledger = ledger
        self._max_calls_per_day = max_calls_per_day
        # The clock is injected so a test can place a call on either side of UTC midnight.
        self._clock = clock

    def before_call(self) -> None:
        now = self._clock()
        # The cap is per UTC day: count from this day's midnight, so it resets without a job.
        midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        calls_today = self._repo.chat_calls_since(midnight)
        if calls_today >= self._max_calls_per_day:
            raise DailyCapReached(
                f"{calls_today} chat calls since {midnight.isoformat()}, "
                f"cap is {self._max_calls_per_day}"
            )

    def after_call(self, provider: str, message: AIMessage) -> None:
        usage = message.usage_metadata
        if usage is None:
            # A provider that reports no usage is a bug on its side, not a free call: charge zero
            # but still record the request, so the daily cap stays honest.
            log.warning("chat_usage_missing", provider=provider)
            input_tokens = output_tokens = 0
        else:
            input_tokens = int(usage["input_tokens"])
            output_tokens = int(usage["output_tokens"])
        # The model id the provider says it used, which may differ from the one requested when a
        # provider routes to a variant; "unknown" keeps the record complete either way.
        model = str(message.response_metadata.get("model_name") or "unknown")
        # Record before charging: the tokens were spent even when the ledger then refuses the
        # answer, so the usage row must survive the BudgetExceeded.
        self._repo.record_chat_call(
            at=self._clock(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._ledger.charge(model, input_tokens + output_tokens)


class ScriptedChatModel(BaseChatModel):
    """Offline stand-in for a chat model: replays a script, records what it was asked."""

    # Exception first, not AIMessage first: pydantic tries the members in order, and AIMessage's
    # own validator crashes with an AttributeError on a non-dict instead of declining the value.
    script: list[Exception | AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    # Tool names bound for each call, [] for an unbound call: trajectory tests read it.
    bound_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        # Like ChatOpenAI, binding returns a wrapper that passes the tools on every call.
        return self.bind(tools=[t.name for t in tools], **kwargs)

    def _next(self, messages: list[BaseMessage], tools: list[str] | None) -> AIMessage:
        # Each call consumes one script step, so a test proves the graph asked exactly as often
        # as it should; an empty script fails loudly instead of returning a made-up answer.
        self.seen.append(list(messages))
        self.bound_tools.append(tools or [])
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
        message = self._next(messages, kwargs.get("tools"))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._next(messages, kwargs.get("tools"))
        chunks: list[AIMessageChunk] = []
        if msg.tool_calls:
            # A tool call streams as chunks that reassemble into the same call, which is how a
            # real provider delivers it and how LangGraph collects it.
            chunks.append(
                AIMessageChunk(
                    content="",
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
        else:
            # Word by word, so a test sees a real token stream rather than one blob.
            words = str(msg.content).split(" ")
            chunks.extend(
                AIMessageChunk(content=w if i == 0 else f" {w}") for i, w in enumerate(words)
            )
        # Usage arrives last, as it does from a provider, so the guard charges it after the text.
        # `chunk_position="last"` marks the end of the stream: LangChain otherwise appends its own
        # empty final chunk, which carries no usage and would become the caller's last chunk.
        chunks.append(
            AIMessageChunk(
                content="",
                usage_metadata=msg.usage_metadata,
                response_metadata=msg.response_metadata,
                chunk_position="last",
            )
        )
        for chunk in chunks:
            gen = ChatGenerationChunk(message=chunk)
            if run_manager:
                # How LangGraph sees each token as it is produced.
                run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen
