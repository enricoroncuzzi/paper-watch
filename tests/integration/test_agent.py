"""Trajectory tests for the agent: which tools ran, in what order, and what the reader gets.

Every test drives the real graph and the real tools over the real test database; only the chat
model is scripted, so CI needs no key, no network and no live model. The script is also the
assertion: each step is consumed by exactly one model call, so a graph that asks too often (or
not at all) fails instead of quietly passing.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from structlog.testing import capture_logs

from paperwatch.agent.citations import NO_SUPPORT
from paperwatch.agent.graph import (
    AgentState,
    AnswerEvent,
    TokenEvent,
    ToolEvent,
    _may_run_tools,
    answer_question,
    build_agent,
    stream_answer,
)
from paperwatch.agent.prompts import FINAL_ROUND_NOTE, SYSTEM_PROMPT
from paperwatch.agent.sources import SourceRegistry
from paperwatch.agent.tools import make_tools
from paperwatch.config import MAX_TOOL_ROUNDS
from paperwatch.llm.chat import (
    CallGuard,
    ChatModel,
    DailyCapReached,
    ModelUnavailable,
    ScriptedChatModel,
)
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.store.models import ChatCall
from paperwatch.store.repository import Repository
from tests.conftest import AGENT_ROWS, SeedRow

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
USAGE = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
Seed = Callable[[list[SeedRow]], dict[str, UUID]]


def call(name: str, n: int = 1, **args: Any) -> AIMessage:
    return AIMessage(
        "", tool_calls=[{"name": name, "args": args, "id": f"c{n}"}], usage_metadata=USAGE
    )


def final(text: str) -> AIMessage:
    return AIMessage(text, usage_metadata=USAGE)


def agent(
    session: Session, *models: ScriptedChatModel, cap: int = 800, budget: int = 10_000
) -> Any:
    reg = SourceRegistry()
    guard = CallGuard(Repository(session), Ledger(budget), cap, lambda: NOW)
    chat = [ChatModel(f"p{i}", m) for i, m in enumerate(models)]
    return build_agent(chat, make_tools(session, FakeEmbedder(), reg), reg, guard)


def calls(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(ChatCall)) or 0


class _FailingStore:
    """A `chat_calls` store that cannot write, to prove that failure cannot hide a model error."""

    def record_chat_call(self, **kwargs: Any) -> None:
        raise RuntimeError("chat_calls is unwritable")

    def chat_calls_since(self, at: datetime) -> int:
        return 0


class _StatusError(RuntimeError):
    """A provider error that carries the HTTP status an SDK error exposes."""

    status_code = 429


def test_search_then_answer(db_session: Session, seed_items: Seed) -> None:
    ids = seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench coding agent harness"),
            final(
                "SWE-bench Verified measures coding agent harnesses on real repository bugs [S1]."
            ),
        ]
    )
    answer, tools = answer_question(agent(db_session, model), "how are coding agents evaluated?")
    assert tools == ["search_corpus"]
    assert answer.supported and answer.citations[0].item_id == ids["swe"]
    assert calls(db_session) == 2


def test_search_get_item_then_answer(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", 1, query="agent memory"),
            call("get_item", 2, source_id="S1"),
            final("Agents that remember across sessions plan better over long tasks [S1]."),
        ]
    )
    _, tools = answer_question(agent(db_session, model), "does memory help agents?")
    assert tools == ["search_corpus", "get_item"]
    assert "Ada Lovelace" in str(model.seen[2][-1].content)  # the get_item result reached the model


def test_opening_an_unseen_alias_is_recoverable(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("get_item", 1, source_id="S4"),
            call("search_corpus", 2, query="agent memory"),
            final("Agents that remember across sessions plan better over long tasks [S1]."),
        ]
    )
    answer, tools = answer_question(agent(db_session, model), "does memory help agents?")
    assert tools == ["get_item", "search_corpus"] and answer.supported


def test_tool_rounds_are_capped(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    script = [call("search_corpus", n, query=f"agents {n}") for n in range(1, 7)]
    model = ScriptedChatModel(
        script=[*script[:4], final("Agents plan better with memory across sessions [S1].")]
    )
    answer, tools = answer_question(agent(db_session, model), "tell me about agents")
    assert len(tools) == 4
    # The answer round keeps the tool definitions and forbids calls with tool_choice="none": a
    # request carrying tool history must include `tools` (OpenRouter's guide), and a strict
    # upstream would reject history without them.
    assert model.call_kwargs[4] == {"parallel_tool_calls": False, "tool_choice": "none"}
    assert sorted(model.bound_tools[4]) == ["compare_items", "get_item", "search_corpus"]
    assert model.seen[4][-1].content == FINAL_ROUND_NOTE
    assert answer.supported


def test_a_model_that_ignores_the_final_note_cannot_run_more_tools(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    # The fifth reply is the answer round, and it still asks for a tool. Nothing may run it, and
    # no sixth model call may happen: the old cap re-served a model that kept asking forever.
    model = ScriptedChatModel(
        script=[
            *[call("search_corpus", n, query=f"agents {n}") for n in range(1, 5)],
            call("search_corpus", 5, query="agents 5"),
        ]
    )
    answer, tools = answer_question(agent(db_session, model), "q")
    assert tools == ["search_corpus"] * 4
    assert answer.text == NO_SUPPORT and not answer.supported
    assert calls(db_session) == 5


def test_the_stream_does_not_echo_an_ignored_tool_call(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            *[call("search_corpus", n, query=f"agents {n}") for n in range(1, 5)],
            call("search_corpus", 5, query="agents 5"),
        ]
    )
    events = list(stream_answer(agent(db_session, model), "q"))
    # An event for a tool call that never ran would tell a client work happened that did not.
    assert [e.name for e in events if isinstance(e, ToolEvent)] == ["search_corpus"] * 4
    assert sum(isinstance(e, AnswerEvent) for e in events) == 1


def test_answer_citing_an_unretrieved_alias_is_flagged(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final(
                "SWE-bench Verified measures coding agents on real bugs [S1]. "
                "Another lab beat it last week [S7]."
            ),
        ]
    )
    answer, _ = answer_question(agent(db_session, model), "q")
    assert [f.kind for f in answer.flags] == ["unknown_source"]


def test_empty_corpus_gives_the_no_support_answer(db_session: Session) -> None:
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="quantum agents"),
            final("Quantum agents are widely deployed in production systems today."),
        ]
    )
    answer, _ = answer_question(agent(db_session, model), "q")
    assert answer.text == NO_SUPPORT and not answer.supported


def test_daily_cap_stops_before_any_model_call(db_session: Session) -> None:
    repo = Repository(db_session)
    repo.record_chat_call(at=NOW, provider="p0", model="m", input_tokens=1, output_tokens=1)
    model = ScriptedChatModel(script=[final("never sent")])
    with pytest.raises(DailyCapReached):
        answer_question(agent(db_session, model, cap=1), "q")
    assert model.seen == [] and calls(db_session) == 1


def test_budget_exceeded_mid_answer(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(script=[call("search_corpus", query="agents"), final("unused")])
    with pytest.raises(BudgetExceeded):
        answer_question(agent(db_session, model, budget=100), "q")  # 60 + 60 > 100
    assert calls(db_session) == 2


def test_fallback_attempt_passes_the_daily_cap(db_session: Session) -> None:
    # Every attempt is checked against the cap, not one check per model call: the primary's
    # failure spends the day's last request, so the fallback must never be reached.
    primary = ScriptedChatModel(script=[RuntimeError("429")])
    fallback = ScriptedChatModel(script=[final("never sent [S1].")])
    with pytest.raises(DailyCapReached):
        answer_question(agent(db_session, primary, fallback, cap=1), "q")
    assert fallback.seen == []
    assert calls(db_session) == 1


def test_fallback_model_answers_when_primary_fails(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    primary = ScriptedChatModel(script=[RuntimeError("429")])
    fallback = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    answer, _ = answer_question(agent(db_session, primary, fallback), "q")
    assert answer.supported
    # The failed attempt is a request against the free tier and is on record (Task 13 Step 11);
    # every call that produced an answer came from the fallback.
    rows = db_session.execute(select(ChatCall.provider, ChatCall.model)).all()
    assert ("p0", "failed") in rows
    assert {provider for provider, model in rows if model != "failed"} == {"p1"}


def test_a_failed_usage_write_does_not_hide_the_model_error(db_session: Session) -> None:
    # A store that raises while recording the failed attempt must not replace the 429 the caller
    # needs to see; the write failure is logged and the fallback loop carries on.
    reg = SourceRegistry()
    guard = CallGuard(_FailingStore(), Ledger(10_000), 800, lambda: NOW)
    model = ScriptedChatModel(script=[RuntimeError("429 from the provider")])
    graph = build_agent([ChatModel("p0", model)], [], reg, guard)
    with pytest.raises(ModelUnavailable) as raised:
        answer_question(graph, "q")
    # The client-facing message names the provider and the exception class only; the provider's
    # own text stays on the chained cause, for a debugger.
    assert str(raised.value) == "every chat model failed: p0 (RuntimeError)"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "429 from the provider"


def test_an_answer_with_no_tool_call_is_refused(db_session: Session) -> None:
    # Review Focus 1: a citation before any search. No tool issued S1, so the model's words about
    # it never become an answer, however confident they sound.
    model = ScriptedChatModel(
        script=[final("Agents plan better with memory across sessions [S1].")]
    )
    answer, tools = answer_question(agent(db_session, model), "q")
    assert tools == []
    assert [f.kind for f in answer.flags] == ["unknown_source"]
    assert answer.text == NO_SUPPORT and not answer.supported


def test_every_model_call_starts_with_the_citation_prompt(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    answer_question(agent(db_session, model), "q")
    assert model.seen[0][0].content == SYSTEM_PROMPT
    assert model.seen[1][0].content == SYSTEM_PROMPT  # prepended again after the tool round
    # The Task 15 review found real answers leaving topic and summary sentences uncited, so the
    # prompt names them instead of relying on "factual sentence" alone.
    assert "topic or summary sentence" in SYSTEM_PROMPT


def test_every_call_binds_tools_without_parallel_calls(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    answer_question(agent(db_session, model), "q")
    # ToolNode would otherwise run one message's tool calls on a thread pool, and the tools share
    # one session and one registry.
    assert model.call_kwargs == [
        {"parallel_tool_calls": False},
        {"parallel_tool_calls": False},
    ]


def test_stream_answer_yields_tools_tokens_then_one_answer(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    text = "SWE-bench Verified measures coding agents on real bugs [S1]."
    model = ScriptedChatModel(script=[call("search_corpus", query="SWE-bench"), final(text)])
    events = list(stream_answer(agent(db_session, model), "q"))
    kinds = [type(e) for e in events]
    assert kinds[0] is ToolEvent and kinds[-1] is AnswerEvent and kinds.count(AnswerEvent) == 1
    assert [e.name for e in events if isinstance(e, ToolEvent)] == ["search_corpus"]
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == text


def test_streaming_charges_the_aggregated_usage(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    reg = SourceRegistry()
    ledger = Ledger(10_000)
    guard = CallGuard(Repository(db_session), ledger, 800, lambda: NOW)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    graph = build_agent(
        [ChatModel("p0", model)], make_tools(db_session, FakeEmbedder(), reg), reg, guard
    )
    list(stream_answer(graph, "q"))
    # Usage arrives on the stream's last chunk, not its first: charging the message each call
    # aggregated keeps a streamed answer honest without double-counting chunks.
    assert ledger.tokens_used == 2 * USAGE["total_tokens"]


def test_a_graph_answers_only_one_question(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    graph = agent(db_session, model)
    answer_question(graph, "q1")
    # The registry (and the ledger) belong to that answer: a second question could cite S1
    # without any tool having returned it for that question, so it is refused outright.
    with pytest.raises(RuntimeError, match="one question"):
        answer_question(graph, "q2")


def test_a_second_question_is_refused_even_when_the_first_found_nothing(
    db_session: Session,
) -> None:
    # The rule is one graph, one run: the ledger is spent even when no tool ran, so a second
    # question on the same graph is refused rather than sharing that ledger.
    model = ScriptedChatModel(script=[final("Quantum agents ship today [S1].")])
    graph = agent(db_session, model)
    answer_question(graph, "q1")
    with pytest.raises(RuntimeError, match="one question"):
        answer_question(graph, "q2")


def test_build_agent_rejects_an_empty_model_list(db_session: Session) -> None:
    # No models cannot answer; refuse with a message that names the mistake, not an IndexError
    # when the fallback loop looks for the last failure.
    reg = SourceRegistry()
    guard = CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW)
    with pytest.raises(ValueError, match="at least one chat model"):
        build_agent([], [], reg, guard)


def test_provider_error_text_stays_out_of_logs_and_errors(db_session: Session) -> None:
    # An SDK status error's string is the whole response body, and OpenRouter's moderation
    # errors echo the flagged question, so neither the log nor the error may carry it.
    reg = SourceRegistry()
    guard = CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW)
    model = ScriptedChatModel(script=[RuntimeError("secret question text")])
    graph = build_agent([ChatModel("p0", model)], [], reg, guard)
    with capture_logs() as logs, pytest.raises(ModelUnavailable) as raised:
        answer_question(graph, "q")
    assert str(raised.value) == "every chat model failed: p0 (RuntimeError)"
    assert "secret question text" not in str(logs)
    failures = [entry for entry in logs if entry["event"] == "chat_model_failed"]
    assert [entry["error_type"] for entry in failures] == ["RuntimeError"]

    # A store failure while recording the attempt must not log a traceback either: structlog
    # renders the chain, and the chain holds the provider error above it.
    store_graph = build_agent(
        [ChatModel("p0", ScriptedChatModel(script=[RuntimeError("secret question text")]))],
        [],
        reg,
        CallGuard(_FailingStore(), Ledger(10_000), 800, lambda: NOW),
    )
    with capture_logs() as store_logs, pytest.raises(ModelUnavailable):
        answer_question(store_graph, "q")
    assert "secret question text" not in str(store_logs)
    assert all("exc_info" not in entry for entry in store_logs)
    record_failures = [e for e in store_logs if e["event"] == "chat_failure_record_failed"]
    assert [entry["error_type"] for entry in record_failures] == ["RuntimeError"]


def test_every_failed_model_is_named_in_order(db_session: Session) -> None:
    primary = ScriptedChatModel(script=[RuntimeError("primary down")])
    fallback = ScriptedChatModel(script=[ValueError("bad model id")])
    with pytest.raises(ModelUnavailable) as raised:
        answer_question(agent(db_session, primary, fallback), "q")
    assert str(raised.value) == "every chat model failed: p0 (RuntimeError), p1 (ValueError)"
    assert isinstance(raised.value.__cause__, ValueError)


def test_a_provider_status_is_logged(db_session: Session) -> None:
    # The HTTP status is the part of an SDK error worth keeping in the log: it says whether to
    # retry later or change the model, without the response body.
    reg = SourceRegistry()
    guard = CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW)
    graph = build_agent(
        [ChatModel("p0", ScriptedChatModel(script=[_StatusError("limited")]))], [], reg, guard
    )
    with capture_logs() as logs, pytest.raises(ModelUnavailable):
        answer_question(graph, "q")
    failure = next(entry for entry in logs if entry["event"] == "chat_model_failed")
    assert (failure["error_type"], failure["status"]) == ("_StatusError", 429)


def test_a_wrong_tool_argument_comes_back_as_an_error_message(
    db_session: Session, seed_items: Seed
) -> None:
    # A small model misspells an argument. ToolNode(handle_tool_errors=True) turns the schema
    # error into a message the model can read and recover from, instead of ending the answer.
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", 1, qery="SWE-bench"),
            call("search_corpus", 2, query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    answer, tools = answer_question(agent(db_session, model), "q")
    assert tools == ["search_corpus", "search_corpus"] and answer.supported
    errors = [m for m in model.seen[1] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errors) == 1


def test_one_message_with_two_tool_calls_is_one_round(
    db_session: Session, seed_items: Seed
) -> None:
    # A round is one model message, however many calls it carries: counting calls instead would
    # have cut the fourth message's call here, so four messages must run five tool calls.
    seed_items(AGENT_ROWS)
    two_calls = AIMessage(
        "",
        tool_calls=[
            {"name": "search_corpus", "args": {"query": "SWE-bench"}, "id": "c1"},
            {"name": "search_corpus", "args": {"query": "agent memory"}, "id": "c2"},
        ],
        usage_metadata=USAGE,
    )
    model = ScriptedChatModel(
        script=[
            two_calls,
            call("search_corpus", 3, query="agents 3"),
            call("search_corpus", 4, query="agents 4"),
            call("search_corpus", 5, query="agents 5"),
            final("Agents plan better with memory across sessions [S1]."),
        ]
    )
    _, tools = answer_question(agent(db_session, model), "q")
    assert tools == ["search_corpus"] * 5
    assert len(model.call_kwargs) == 5


def test_a_tool_call_never_routes_after_the_answer_round() -> None:
    # The provider-independent half of the cap, tested without a model: a reply that still
    # carries a call once the cap is spent goes to the checker, never to the tools.
    message = AIMessage(
        "", tool_calls=[{"name": "search_corpus", "args": {"query": "q"}, "id": "c1"}]
    )
    state: AgentState = {
        "messages": [HumanMessage("q"), message],
        "rounds": MAX_TOOL_ROUNDS + 1,
        "answer": None,
    }
    assert not _may_run_tools(state)
    assert _may_run_tools({**state, "rounds": MAX_TOOL_ROUNDS})
