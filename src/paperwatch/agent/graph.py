"""The question-answering agent: a capped model/tool loop, then a deterministic citation check.

`build_agent` wires the four things the tests can see without a live model: the model chain
(primary, then fallback), the read-only tools, this answer's `SourceRegistry`, and the
`CallGuard` that charges every attempt. The loop is written out rather than taken from
LangGraph's prebuilt ReAct agent, because the guard, the round cap and the final check are the
parts this project wants to show.

`stream_answer` emits what a client needs as it happens (the tool calls, then the answer's tokens,
then one checked answer); `answer_question` is the synchronous form for tests and the CLI.
"""

import threading
import time
from collections.abc import Generator, Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, cast
from uuid import UUID

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from paperwatch.agent.citations import CheckedAnswer, check_answer
from paperwatch.agent.prompts import FINAL_ROUND_NOTE, SYSTEM_PROMPT
from paperwatch.agent.sources import SourceRef, SourceRegistry
from paperwatch.config import MAX_TOOL_ROUNDS
from paperwatch.llm.chat import CallGuard, ChatModel, ModelUnavailable

log = structlog.get_logger()


class AgentState(TypedDict):
    """One answer's state: the conversation, the tool rounds used plus one for the answer
    round, and the checked answer."""

    messages: Annotated[list[AnyMessage], add_messages]
    rounds: int
    answer: CheckedAnswer | None


@dataclass(frozen=True)
class ToolEvent:
    """The model asked for a tool. `args` is model-authored JSON and is escaped on render."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class TokenEvent:
    """A piece of the model's answer as it is produced, before the citation check."""

    text: str


@dataclass(frozen=True)
class SourceEvent:
    """A source a tool just returned, emitted once per alias the first time it appears."""

    ref: SourceRef


@dataclass(frozen=True)
class StepEvent:
    """One tool round finished: its number, how long it took, and how many sources it added."""

    round: int
    ms: int
    new_sources: int


@dataclass(frozen=True)
class AnswerEvent:
    """The checked answer: always the last event, and it replaces any text streamed before it."""

    answer: CheckedAnswer


AgentEvent = ToolEvent | SourceEvent | StepEvent | TokenEvent | AnswerEvent

# `CompiledStateGraph` is generic; the state is ours and nothing else is used, but mypy strict
# needs the parameters spelled out.
AgentGraph = CompiledStateGraph[AgentState, Any, Any, Any]

# A model as the graph uses it: something to invoke, with or without tools bound. `Any` on both
# sides because a bound runnable and a bare chat model are different types that both qualify.
_Model = Runnable[Any, Any]


class AnswerAborted(BaseException):
    """The consumer is gone, so the answer stops at the model's next token.

    A BaseException on purpose: the model-failure handler in `call_models` catches `Exception`,
    and an abort must not look like provider trouble and send the question to the fallback model,
    which would spend another request for nobody.

    It is raised from a graph callback or from `call_models`, so it unwinds whichever step is
    running: a model round, the step after a tool, or the executor's own loop. `run_question`
    returns quietly when it sees one, because there is no consumer left to tell.
    """


class _AbortOnDisconnect(BaseCallbackHandler):
    """Raises `AnswerAborted` at the next model event once the answer's consumer has gone away."""

    # langchain-core logs and swallows a handler's exceptions unless this is set.
    raise_error = True

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # `invoke` never streams, so `on_llm_new_token` would never fire for it. The start of a
        # request is the last moment an abort can keep it from being sent at all.
        if self._cancel.is_set():
            raise AnswerAborted

    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], **kwargs: Any) -> None:
        # Called for every streamed chunk, so one flag check per token is the cost of an answer
        # that can be stopped; a plain `invoke` without streaming never calls it.
        if self._cancel.is_set():
            raise AnswerAborted


def build_agent(
    models: list[ChatModel],
    tools: list[BaseTool],
    registry: SourceRegistry,
    guard: CallGuard,
    cancel: threading.Event | None = None,
) -> AgentGraph:
    """Build the graph for one run. Do not start a second run on it (see `call_agent`)."""
    if not models:
        raise ValueError("an agent graph needs at least one chat model")
    # Both providers accept `parallel_tool_calls` (it is in OpenRouter's chat request schema and
    # in Groq's chat completions schema) and langchain-openai sends it with every request.
    # Disabling it is defence in depth: ToolNode would otherwise run one message's tool calls
    # concurrently, and the tools share one session and one non-thread-safe registry.
    with_tools = [
        (m.provider, m.model.bind_tools(tools, parallel_tool_calls=False)) for m in models
    ]
    # The answer round keeps the tool definitions and forbids calls with tool_choice="none": a
    # request that carries tool history must also carry `tools` (OpenRouter's tool-calling
    # guide), and a strict upstream rejects history without them, which can turn a capped
    # question into ModelUnavailable. Groq documents the same "none" value.
    answer_only = [
        (m.provider, m.model.bind_tools(tools, parallel_tool_calls=False, tool_choice="none"))
        for m in models
    ]
    # A graph answers one question, once: its registry and ledger belong to that run, so a second
    # run is refused whatever the first one did (a retry after a provider failure included, since
    # the ledger is already spent).
    started = False

    def call_models(
        candidates: Sequence[tuple[str, _Model]], messages: list[AnyMessage]
    ) -> AIMessage:
        # An explicit fallback loop, not `.with_fallbacks()`: every attempt is a request against
        # the free tier, so the guard has to see each one and know which provider answered it.
        failures: list[tuple[str, Exception]] = []
        for provider, model in candidates:
            # The client is gone, so no request may start: the callback would abort it at its
            # first chunk, but the request would still be spent against the free daily cap.
            if cancel is not None and cancel.is_set():
                raise AnswerAborted
            # Outside the try on purpose: a cap or budget decision must not look like provider
            # trouble, because another provider would spend the same budget or bypass the same
            # cap.
            guard.before_call()
            try:
                message = model.invoke(messages)
            except AnswerAborted:
                # The request reached the provider before the client left, so it counts against
                # the daily cap. The store's own failure is logged, never raised over the abort.
                try:
                    guard.after_failure(provider)
                except Exception as store_exc:  # noqa: BLE001 - never hide the abort
                    log.warning(
                        "chat_failure_record_failed",
                        provider=provider,
                        error_type=type(store_exc).__name__,
                    )
                raise
            except Exception as exc:  # noqa: BLE001 - any provider error moves to the next model
                # The attempt reached the provider, so it counts against the daily cap (Task 13
                # Step 11). A store that cannot record it is logged, never raised: the provider
                # error is what the caller needs to see.
                try:
                    guard.after_failure(provider)
                except Exception as store_exc:  # noqa: BLE001 - never hide the provider error
                    # The store error's own text is statement parameters, not provider text, but
                    # a rendered traceback would chain to the provider error above it.
                    log.warning(
                        "chat_failure_record_failed",
                        provider=provider,
                        error_type=type(store_exc).__name__,
                    )
                # An SDK error's string is the whole response body, and OpenRouter's moderation
                # errors echo the flagged question, so only the error's shape is logged.
                log.warning(
                    "chat_model_failed",
                    provider=provider,
                    error_type=type(exc).__name__,
                    status=getattr(exc, "status_code", None),
                )
                failures.append((provider, exc))
                continue
            # Outside the try on purpose, as above: the ledger's refusal is this answer's budget.
            guard.after_call(provider, message)
            # Every chat model returns an AIMessage; the cast only keeps mypy strict honest
            # about the generic `_Model` above.
            return cast(AIMessage, message)
        # Provider names and exception classes only: this message can reach a client event, so
        # it must not carry provider response text (the cause stays chained for a debugger).
        attempts = ", ".join(f"{provider} ({type(exc).__name__})" for provider, exc in failures)
        raise ModelUnavailable(f"every chat model failed: {attempts}") from failures[-1][1]

    def call_agent(state: AgentState) -> dict[str, Any]:
        nonlocal started
        # The first model call of a run is the only one with just the question in the state; a
        # second run would share this answer's registry and ledger.
        if len(state["messages"]) == 1:
            if started:
                raise RuntimeError("an agent graph answers one question")
            started = True
        # The system prompt is prepended at every call and never stored, so the transcript a
        # client sees holds only the question, the tool calls and the answer.
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        if state["rounds"] >= MAX_TOOL_ROUNDS:
            # The cap works by forbidding tool calls (tool_choice="none") and by spending this
            # round below: whatever the reply says, no tool runs after it. The note tells the
            # model why it cannot search again.
            note = HumanMessage(FINAL_ROUND_NOTE)
            message = call_models(answer_only, [*messages, note])
            # A lenient provider (or a model writing a call as text) can still return a call.
            # Nothing runs it, so neither `route` nor the event stream may see it.
            if message.tool_calls:
                message = message.model_copy(update={"tool_calls": [], "invalid_tool_calls": []})
            return {"messages": [note, message], "rounds": state["rounds"] + 1}
        message = call_models(with_tools, messages)
        # A round is one model message, however many tool calls it carries; a plain answer ends
        # the loop at once.
        return {
            "messages": [message],
            "rounds": state["rounds"] + (1 if message.tool_calls else 0),
        }

    def route(state: AgentState) -> str:
        return "tools" if _may_run_tools(state) else "check"

    def check(state: AgentState) -> dict[str, Any]:
        # Deterministic and last: an answer with no valid citation becomes NO_SUPPORT here,
        # before any caller can read it.
        last = state["messages"][-1]
        # ChatOpenAI returns string content; a provider returning content blocks would be read as
        # their repr here, and the checker would treat that repr as plain text.
        return {"answer": check_answer(str(last.content), registry)}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    # handle_tool_errors=True: a ToolException reaches the model as a message it can act on,
    # instead of ending the answer.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("check", check)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "check"])
    graph.add_edge("tools", "agent")
    graph.add_edge("check", END)
    return graph.compile()


def _initial_state(question: str) -> AgentState:
    return {"messages": [HumanMessage(question)], "rounds": 0, "answer": None}


def _may_run_tools(state: AgentState) -> bool:
    """Whether the last message's tool calls may run.

    Calls from the answer round must not: the cap is spent, so the provider-independent half of
    the cap sends the message to the checker whatever it asked for. The node also drops a call
    the answer round returned (see `call_agent`), so `route` and the event stream never see it.
    """
    last = state["messages"][-1]
    return (
        isinstance(last, AIMessage) and bool(last.tool_calls) and state["rounds"] <= MAX_TOOL_ROUNDS
    )


def _run_config(cancel: threading.Event | None = None) -> RunnableConfig:
    """The config every run needs: an explicit recursion limit, plus the abort callback if any.

    LangGraph aborts a run that takes more supersteps than `recursion_limit`, and inheriting its
    default would tie the round cap to the framework: this version defaults to 10007 but the
    documented default is 25, and a run spends about two supersteps per tool round plus one for
    the answer round and one for the check. The limit is therefore derived from the round cap,
    read here at call time so a raised cap moves both together.
    """
    config: RunnableConfig = {"recursion_limit": 2 * MAX_TOOL_ROUNDS + 5}
    if cancel is not None:
        # The callback is attached to every run in the graph, so whichever model call is
        # streaming hears the abort; without a cancel signal there is nothing to watch.
        config["callbacks"] = [_AbortOnDisconnect(cancel)]
    return config


def stream_answer(
    graph: AgentGraph,
    question: str,
    cancel: threading.Event | None = None,
    registry: SourceRegistry | None = None,
) -> Generator[AgentEvent, None, None]:
    """Yield tool calls, sources and tokens as they happen, then exactly one `AnswerEvent`.

    `cancel` is the consumer's signal that nobody is listening any more: once it is set, the
    in-flight model call raises at its next token instead of running to the end. One graph, one
    run: build a new one for the next question (see `build_agent`).

    `registry` is the answer's registry, which the graph writes to through the tools. Without it
    the stream carries no `source` or `step` events: a caller that only wants the answer (a test,
    the eval) does not need them.
    """
    state = _initial_state(question)
    config = _run_config(cancel)
    # With several stream modes LangGraph yields `(mode, chunk)` pairs; its type stub only says
    # the modes' union, so the pair is named here once.
    stream = cast(
        Generator[tuple[str, Any], None, None],
        graph.stream(state, config=config, stream_mode=["messages", "updates"]),
    )
    # The page shows sources as they are found and the time each tool round took, so the stream
    # carries them next to the tool calls. `seen` counts the aliases already sent: the registry is
    # the source of truth, and it keeps an item's alias across rounds.
    seen = 0
    round_number = 0
    round_started: float | None = None
    # Closed explicitly, not left to garbage collection: closing the graph's stream joins its worker
    # thread, so the aborted call has written its usage row before the caller releases anything.
    with closing(stream):
        for mode, chunk in stream:
            if mode == "messages":
                message, meta = chunk
                # Only the agent node talks to a model, and only chunks with text are words: the
                # empty ones carry a tool call or the usage.
                if (
                    meta.get("langgraph_node") == "agent"
                    and isinstance(message, AIMessageChunk)
                    and message.content
                ):
                    # ChatOpenAI streams string content; a content-block provider would stream a
                    # list, whose repr is not answer text.
                    yield TokenEvent(str(message.content))
            elif mode == "updates":
                if "agent" in chunk:
                    last = chunk["agent"]["messages"][-1]
                    tool_calls = getattr(last, "tool_calls", [])
                    if tool_calls and round_started is None:
                        # Timed from the model's decision, so a round's duration includes the tool
                        # work the reader waited for.
                        round_started = time.monotonic()
                    # The call the model just decided to make, echoed so a client can show progress.
                    for tool_call in tool_calls:
                        yield ToolEvent(tool_call["name"], tool_call["args"])
                if "tools" in chunk:
                    round_number += 1
                    finished = time.monotonic()
                    started = round_started if round_started is not None else finished
                    round_started = None
                    refs = registry.all() if registry is not None else []
                    new_refs = refs[seen:]
                    seen = len(refs)
                    # Sources first, then the step that counts them: a client renders sources as
                    # they arrive and uses the step as the round's closing marker. Without a
                    # registry there is nothing to announce.
                    if registry is not None:
                        for ref in new_refs:
                            yield SourceEvent(ref)
                        yield StepEvent(
                            round=round_number,
                            ms=int((finished - started) * 1000),
                            new_sources=len(new_refs),
                        )
                if "check" in chunk:
                    yield AnswerEvent(chunk["check"]["answer"])


def answer_question(graph: AgentGraph, question: str) -> tuple[CheckedAnswer, list[str], str]:
    """Run the graph to the end: the checked answer, the tools it called, and the model's text.

    The third value is the model's own final message, before `check_answer` rewrote or replaced
    it. The QA eval records that raw text, so a replay re-runs the current checker instead of
    reading a verdict an older checker already applied. One graph, one run: build a new one for
    the next question (see `build_agent`).
    """
    state = graph.invoke(_initial_state(question), config=_run_config())
    answer: CheckedAnswer | None = state["answer"]
    if answer is None:  # every path ends in the check node, so this is a wiring bug
        raise RuntimeError("the agent finished without checking its answer")
    last = state["messages"][-1]
    tool_names = [
        tool_call["name"]
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    # The check node reads `str(last.content)` too, and every path ends with an agent message.
    return answer, tool_names, str(last.content)
