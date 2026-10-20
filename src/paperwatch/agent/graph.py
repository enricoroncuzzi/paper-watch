"""The question-answering agent: a capped model/tool loop, then a deterministic citation check.

`build_agent` wires the four things the tests can see without a live model: the model chain
(primary, then fallback), the read-only tools, this answer's `SourceRegistry`, and the
`CallGuard` that charges every attempt. The loop is written out rather than taken from
LangGraph's prebuilt ReAct agent, because the guard, the round cap and the final check are the
parts this project wants to show.

`stream_answer` emits what a client needs as it happens (the tool calls, then the answer's tokens,
then one checked answer); `answer_question` is the synchronous form for tests and the CLI.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, cast

import structlog
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from paperwatch.agent.citations import CheckedAnswer, check_answer
from paperwatch.agent.prompts import FINAL_ROUND_NOTE, SYSTEM_PROMPT
from paperwatch.agent.sources import SourceRegistry
from paperwatch.config import MAX_TOOL_ROUNDS
from paperwatch.llm.chat import CallGuard, ChatModel, ModelUnavailable

log = structlog.get_logger()


class AgentState(TypedDict):
    """One answer's state: the conversation, the tool rounds used, and the checked answer."""

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
class AnswerEvent:
    """The checked answer: always the last event, and it replaces any text streamed before it."""

    answer: CheckedAnswer


AgentEvent = ToolEvent | TokenEvent | AnswerEvent

# `CompiledStateGraph` is generic; the state is ours and nothing else is used, but mypy strict
# needs the parameters spelled out.
AgentGraph = CompiledStateGraph[AgentState, Any, Any, Any]

# A model as the graph uses it: something to invoke, with or without tools bound. `Any` on both
# sides because a bound runnable and a bare chat model are different types that both qualify.
_Model = Runnable[Any, Any]


def build_agent(
    models: list[ChatModel],
    tools: list[BaseTool],
    registry: SourceRegistry,
    guard: CallGuard,
) -> AgentGraph:
    """Build the graph for one answer. Do not ask it a second question (see `call_agent`)."""
    # Both providers accept `parallel_tool_calls` (it is in OpenRouter's chat request schema and
    # in Groq's chat completions schema) and langchain-openai sends it with every request.
    # Disabling it is defence in depth: ToolNode would otherwise run one message's tool calls
    # concurrently, and the tools share one session and one non-thread-safe registry.
    with_tools = [
        (m.provider, m.model.bind_tools(tools, parallel_tool_calls=False)) for m in models
    ]
    # The answer round keeps the tool definitions and forbids calls with tool_choice="none": a
    # request that carries tool history must also carry `tools` (OpenRouter's tool-calling
    # guide), and a strict upstream rejects history without them, which would turn every capped
    # question into ModelUnavailable. Groq documents the same "none" value.
    answer_only = [
        (m.provider, m.model.bind_tools(tools, parallel_tool_calls=False, tool_choice="none"))
        for m in models
    ]

    def call_models(
        candidates: Sequence[tuple[str, _Model]], messages: list[AnyMessage]
    ) -> AIMessage:
        # An explicit fallback loop, not `.with_fallbacks()`: every attempt is a request against
        # the free tier, so the guard has to see each one and know which provider answered it.
        failures: list[tuple[str, Exception]] = []
        for provider, model in candidates:
            # Outside the try on purpose: a cap or budget decision must not look like provider
            # trouble, because another provider would spend the same budget or bypass the same
            # cap.
            guard.before_call()
            try:
                message = model.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - any provider error moves to the next model
                # The attempt reached the provider, so it counts against the daily cap (Task 13
                # Step 11). A store that cannot record it is logged, never raised: the provider
                # error is what the caller needs to see.
                try:
                    guard.after_failure(provider)
                except Exception:  # recording a failure must not hide it
                    log.warning("chat_failure_record_failed", provider=provider, exc_info=True)
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
        # One graph holds one answer's registry (and ledger), so a second question must not start
        # on it: it could cite an alias the first question's tools issued.
        if len(state["messages"]) == 1 and registry.all():
            raise RuntimeError("an agent graph answers one question")
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


def stream_answer(graph: AgentGraph, question: str) -> Iterator[AgentEvent]:
    """Yield tool calls and tokens as they happen, then exactly one `AnswerEvent`.

    The graph answers one question: build a new one for the next (see `build_agent`).
    """
    state = _initial_state(question)
    # With several stream modes LangGraph yields `(mode, chunk)` pairs; its type stub only says
    # the modes' union, so the pair is named here once.
    stream = cast(
        Iterator[tuple[str, Any]], graph.stream(state, stream_mode=["messages", "updates"])
    )
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
                # The call the model just decided to make, echoed so a client can show progress.
                for tool_call in getattr(last, "tool_calls", []):
                    yield ToolEvent(tool_call["name"], tool_call["args"])
            if "check" in chunk:
                yield AnswerEvent(chunk["check"]["answer"])


def answer_question(graph: AgentGraph, question: str) -> tuple[CheckedAnswer, list[str]]:
    """Run the graph to the end: the checked answer, and the tools it called, in order.

    The graph answers one question: build a new one for the next (see `build_agent`).
    """
    state = graph.invoke(_initial_state(question))
    answer: CheckedAnswer | None = state["answer"]
    if answer is None:  # every path ends in the check node, so this is a wiring bug
        raise RuntimeError("the agent finished without checking its answer")
    tool_names = [
        tool_call["name"]
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    return answer, tool_names
