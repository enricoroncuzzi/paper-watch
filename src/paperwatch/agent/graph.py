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
from paperwatch.llm.chat import CallGuard, ChatModel, DailyCapReached, ModelUnavailable
from paperwatch.llm.ledger import BudgetExceeded

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
    # Both providers accept `parallel_tool_calls` (it is in OpenRouter's chat request schema and
    # in Groq's chat completions schema) and langchain-openai sends it with every request.
    # Disabling it is defence in depth: ToolNode would otherwise run one message's tool calls
    # concurrently, and the tools share one session and one non-thread-safe registry.
    with_tools = [
        (m.provider, m.model.bind_tools(tools, parallel_tool_calls=False)) for m in models
    ]
    # The last round is called without tools on purpose: it cannot answer by searching again.
    without_tools = [(m.provider, m.model) for m in models]

    def call_models(
        candidates: Sequence[tuple[str, _Model]], messages: list[AnyMessage]
    ) -> AIMessage:
        # An explicit fallback loop, not `.with_fallbacks()`: every attempt is a request against
        # the free tier, so the guard has to see each one and know which provider answered it.
        failure: Exception | None = None
        for provider, model in candidates:
            guard.before_call()
            try:
                message = model.invoke(messages)
            except (DailyCapReached, BudgetExceeded):
                # Decisions about this answer, not provider trouble: another provider would
                # spend the same budget or bypass the same cap.
                raise
            except Exception as exc:  # noqa: BLE001 - any provider error moves to the next model
                # The attempt reached the provider, so it counts against the daily cap (Task 13
                # Step 11). A store that cannot record it is logged, never raised: the provider
                # error is what the caller needs to see.
                try:
                    guard.after_failure(provider)
                except Exception:  # recording a failure must not hide it
                    log.warning("chat_failure_record_failed", provider=provider, exc_info=True)
                log.warning("chat_model_failed", provider=provider, error=str(exc))
                failure = exc
                continue
            guard.after_call(provider, message)
            # Every chat model returns an AIMessage; the cast only keeps mypy strict honest
            # about the generic `_Model` above.
            return cast(AIMessage, message)
        raise ModelUnavailable(f"every chat model failed: {failure}") from failure

    def call_agent(state: AgentState) -> dict[str, Any]:
        # The system prompt is prepended at every call and never stored, so the transcript a
        # client sees holds only the question, the tool calls and the answer.
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        if state["rounds"] >= MAX_TOOL_ROUNDS:
            # The cap works by taking the tools away, not by asking the model to stop: the note
            # says why, and a reply without a tool call is the only way out of this node.
            note = HumanMessage(FINAL_ROUND_NOTE)
            return {"messages": [note, call_models(without_tools, [*messages, note])]}
        message = call_models(with_tools, messages)
        # Only a tool call spends a round; a plain answer ends the loop at once.
        return {
            "messages": [message],
            "rounds": state["rounds"] + (1 if message.tool_calls else 0),
        }

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check"

    def check(state: AgentState) -> dict[str, Any]:
        # Deterministic and last: an answer with no valid citation becomes NO_SUPPORT here,
        # before any caller can read it.
        last = state["messages"][-1]
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


def stream_answer(graph: AgentGraph, question: str) -> Iterator[AgentEvent]:
    """Yield tool calls and answer tokens as they happen, then exactly one `AnswerEvent`."""
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
    """Run the graph to the end: the checked answer, and the tools it called, in order."""
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
