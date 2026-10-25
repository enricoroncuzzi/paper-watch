"""Trajectory tests for the agent: which tools ran, in what order, and what the reader gets.

Every test drives the real graph and the real tools over the real test database; only the chat
model is scripted, so CI needs no key, no network and no live model. The script is also the
assertion: each step is consumed by exactly one model call, so a graph that asks too often (or
not at all) fails instead of quietly passing.
"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from structlog.testing import capture_logs

from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer
from paperwatch.agent.graph import (
    AgentState,
    AnswerAborted,
    AnswerEvent,
    SourceEvent,
    StepEvent,
    TokenEvent,
    ToolEvent,
    _may_run_tools,
    answer_question,
    bound_history,
    build_agent,
    stream_answer,
)
from paperwatch.agent.prompts import FINAL_ROUND_NOTE, HISTORY_NOTE, SYSTEM_PROMPT
from paperwatch.agent.sources import (
    SourceRef,
    SourceRegistry,
    ref_from_json,
    ref_to_json,
)
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


def agent_with_registry(
    session: Session, *models: ScriptedChatModel, cap: int = 800, budget: int = 10_000
) -> tuple[Any, SourceRegistry]:
    # The stream only carries source and step events when it can see the answer's registry.
    reg = SourceRegistry()
    guard = CallGuard(Repository(session), Ledger(budget), cap, lambda: NOW)
    chat = [ChatModel(f"p{i}", m) for i, m in enumerate(models)]
    return build_agent(chat, make_tools(session, FakeEmbedder(), reg), reg, guard), reg


def calls(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(ChatCall)) or 0


class _FailingStore:
    """A `chat_calls` store that reserves but cannot complete a row.

    Reservation has to succeed, or the model would never be called; the failure under test is the
    completion write, which must not hide the provider error or the abort.
    """

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int, int]:
        return 1, 1

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        raise RuntimeError("chat_calls is unwritable")

    def fail_chat_call(self, call_id: int) -> None:
        raise RuntimeError("chat_calls is unwritable")


class _AbortingModel(ScriptedChatModel):
    """A model whose request is aborted, as `_AbortOnDisconnect` aborts a real one.

    The abort is a `BaseException`, which `ScriptedChatModel`'s script deliberately does not
    accept (its union is `Exception | AIMessage`), so it is raised from `_next` instead.
    """

    def _next(self, messages: list[Any], kwargs: dict[str, Any]) -> AIMessage:
        self.seen.append(list(messages))
        raise AnswerAborted


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
    answer, tools, _ = answer_question(agent(db_session, model), "how are coding agents evaluated?")
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
    _, tools, _ = answer_question(agent(db_session, model), "does memory help agents?")
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
    answer, tools, _ = answer_question(agent(db_session, model), "does memory help agents?")
    assert tools == ["get_item", "search_corpus"] and answer.supported


def test_tool_rounds_are_capped(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    script = [call("search_corpus", n, query=f"agents {n}") for n in range(1, 7)]
    model = ScriptedChatModel(
        script=[*script[:4], final("Agents plan better with memory across sessions [S1].")]
    )
    answer, tools, _ = answer_question(agent(db_session, model), "tell me about agents")
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
    answer, tools, _ = answer_question(agent(db_session, model), "q")
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
    answer, _, _ = answer_question(agent(db_session, model), "q")
    assert [f.kind for f in answer.flags] == ["unknown_source"]


def test_empty_corpus_gives_the_no_support_answer(db_session: Session) -> None:
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="quantum agents"),
            final("Quantum agents are widely deployed in production systems today."),
        ]
    )
    answer, _, _ = answer_question(agent(db_session, model), "q")
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
    answer, _, _ = answer_question(agent(db_session, primary, fallback), "q")
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
    answer, tools, _ = answer_question(agent(db_session, model), "q")
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


class _ConfigSpy:
    """A stand-in graph that records the config each surface hands it."""

    def __init__(self) -> None:
        self.config: Any = None

    def invoke(self, state: Any, config: Any = None) -> Any:
        self.config = config
        # The final message is what `answer_question` returns as the model's own text.
        return {
            "answer": CheckedAnswer(NO_SUPPORT, (), (), False),
            "messages": [AIMessage("q")],
        }

    def stream(self, state: Any, config: Any = None, stream_mode: Any = None) -> Any:
        self.config = config
        # LangGraph's stream is a generator, which the caller closes.
        yield from ()


def test_both_surfaces_derive_the_recursion_limit_from_the_round_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The limit is a function of the cap, not a number: a cap change must move it too.
    import paperwatch.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "MAX_TOOL_ROUNDS", 7)
    spy = _ConfigSpy()
    answer_question(spy, "q")
    assert spy.config["recursion_limit"] == 2 * 7 + 5
    list(stream_answer(spy, "q"))
    assert spy.config["recursion_limit"] == 2 * 7 + 5


@pytest.mark.parametrize("mode", ["invoke", "stream"])
def test_a_raised_round_cap_does_not_hit_the_recursion_limit(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # A run spends about two supersteps per tool round plus the answer round and the check, and
    # LangGraph aborts a run that passes its `recursion_limit`. The documented default is 25, so a
    # cap-12 run would end in GraphRecursionError before the answer round unless the agent passes
    # its own limit. (The installed version defaults to 10007, which hides the bug; 25 is what the
    # framework promises, so the test uses it.)
    from langgraph._internal import _config as langgraph_config

    import paperwatch.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "MAX_TOOL_ROUNDS", 12)
    monkeypatch.setattr(langgraph_config, "DEFAULT_RECURSION_LIMIT", 25)
    model = ScriptedChatModel(
        script=[call("search_corpus", n, query="agents") for n in range(1, 13)]
        + [final("SWE-bench Verified measures coding agents on real bugs [S1].")]
    )
    graph = agent(db_session, model)
    if mode == "invoke":
        _, tools, _ = answer_question(graph, "q")
        assert len(tools) == 12
    else:
        events = list(stream_answer(graph, "q"))
        assert len([event for event in events if isinstance(event, ToolEvent)]) == 12
        assert isinstance(events[-1], AnswerEvent)


# Resolved from this file, not the working directory, so a run started elsewhere still finds it;
# the same pattern as `tests/unit/test_retrieval_eval.py`'s data path.
LESSON = Path(__file__).resolve().parents[2] / "docs" / "learn" / "12-langgraph-agents.md"


def test_the_lessons_graph_diagram_matches_the_built_graph(db_session: Session) -> None:
    # The lesson draws the graph with LangGraph's own renderer. The committed block must be what a
    # real build produces, or the picture drifts from the code; regenerate it by pasting
    # `graph.get_graph().draw_mermaid()` for a graph from `build_agent`.
    lesson = LESSON.read_text()
    assert lesson.count("```mermaid\n") == 1, "the lesson must hold exactly one mermaid block"
    committed = lesson.split("```mermaid\n", 1)[1].split("\n```", 1)[0] + "\n"
    drawn = agent(db_session, ScriptedChatModel(script=[])).get_graph().draw_mermaid()
    assert committed == drawn, "the lesson's diagram is stale: paste draw_mermaid()'s output"


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


def test_an_abort_never_tries_the_fallback_model(db_session: Session) -> None:
    # The client is gone, so another provider would spend another request for nobody. The abort is
    # a BaseException precisely so the fallback loop's `except Exception` cannot catch it.
    primary = _AbortingModel(script=[])
    fallback = ScriptedChatModel(script=[final("Quantum agents ship today [S1].")])
    graph = agent(db_session, primary, fallback)
    with pytest.raises(AnswerAborted):
        answer_question(graph, "q")
    assert fallback.seen == []


def test_a_failing_store_never_masks_the_abort(db_session: Session) -> None:
    # Recording the aborted attempt can fail; the abort is what the caller must see, and the
    # store's failure is only logged.
    reg = SourceRegistry()
    graph = build_agent(
        [ChatModel("p0", _AbortingModel(script=[]))],
        [],
        reg,
        CallGuard(_FailingStore(), Ledger(10_000), 800, lambda: NOW),
    )
    with capture_logs() as logs, pytest.raises(AnswerAborted):
        answer_question(graph, "q")
    assert [e["error_type"] for e in logs if e["event"] == "chat_failure_record_failed"] == [
        "RuntimeError"
    ]


def test_a_set_cancel_event_stops_the_first_request_from_being_made(
    db_session: Session,
) -> None:
    # The flag is checked before `guard.before_call()`, so an aborted request is not counted
    # against the daily cap and records nothing at all.
    model = ScriptedChatModel(script=[final("Quantum agents ship today [S1].")])
    cancel = threading.Event()
    cancel.set()
    reg = SourceRegistry()
    graph = build_agent(
        [ChatModel("p0", model)],
        [],
        reg,
        CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW),
        cancel,
    )
    before = calls(db_session)
    with pytest.raises(AnswerAborted):
        answer_question(graph, "q")
    assert model.seen == [] and calls(db_session) == before


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
    answer, tools, _ = answer_question(agent(db_session, model), "q")
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
    _, tools, _ = answer_question(agent(db_session, model), "q")
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
        "turn_start": 0,
    }
    assert not _may_run_tools(state)
    assert _may_run_tools({**state, "rounds": MAX_TOOL_ROUNDS})


def test_a_streamed_model_id_repeated_on_every_chunk_is_recorded_once(
    db_session: Session, seed_items: Seed
) -> None:
    # OpenRouter names the model on more than one streamed chunk, and langchain-core's merge
    # concatenates string metadata, so the old guard stored "idid" (and a longer answer could
    # overflow the column and kill the answer). The streamed path is what makes it happen.
    seed_items(AGENT_ROWS)
    name = "vendor/model:free"
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ],
        stream_model_name=name,
    )
    reg = SourceRegistry()
    guard = CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW)
    graph = build_agent(
        [ChatModel("p0", model)], make_tools(db_session, FakeEmbedder(), reg), reg, guard
    )
    list(stream_answer(graph, "q"))
    assert list(db_session.scalars(select(ChatCall.model))) == [name, name]


def test_sources_stream_once_as_they_are_found(db_session: Session, seed_items: Seed) -> None:
    # The page shows each source as soon as a tool returns it, and never twice: the registry keeps
    # the alias, so an overlapping second search adds only the items it found first.
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench", tiers=["peer_reviewed"]),
            call("search_corpus", query="agent harness memory", n=2),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    graph, registry = agent_with_registry(db_session, model)
    events = list(stream_answer(graph, "q", registry=registry))
    sources = [e for e in events if isinstance(e, SourceEvent)]
    assert [e.ref.alias for e in sources] == ["S1", "S2", "S3"]
    assert len({e.ref.item_id for e in sources}) == 3


def test_a_tool_round_ends_with_a_step_event(db_session: Session, seed_items: Seed) -> None:
    # One step per tool round, after that round's sources, so the page can time the round and count
    # what it added without reading the sources themselves.
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench", tiers=["peer_reviewed"]),
            call("search_corpus", query="agent harness memory", n=2),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    graph, registry = agent_with_registry(db_session, model)
    events = list(stream_answer(graph, "q", registry=registry))
    starts = [i for i, e in enumerate(events) if isinstance(e, ToolEvent)]
    steps = [(i, e) for i, e in enumerate(events) if isinstance(e, StepEvent)]
    assert [step.round for _, step in steps] == [1, 2]
    assert [step.new_sources for _, step in steps] == [1, 2]
    assert all(isinstance(step.ms, int) and step.ms >= 0 for _, step in steps)
    # A step comes after the tool call that opened its round, and after that round's sources.
    for index, step in steps:
        assert any(start < index for start in starts)
        between = [e for e in events[starts[step.round - 1] : index] if isinstance(e, SourceEvent)]
        assert len(between) == step.new_sources


def refs_for(session: Session, ids: dict[str, UUID], *keys: str) -> list[SourceRef]:
    """The refs a saved thread carries, taken from the stored items a search would return."""
    details = Repository(session).admitted_details([ids[key] for key in keys])
    registry = SourceRegistry()
    for key in keys:
        registry.register(details[ids[key]])
    return registry.refs()


def answer_from_stream(events: list[Any]) -> CheckedAnswer:
    answers = [event.answer for event in events if isinstance(event, AnswerEvent)]
    assert len(answers) == 1, "an answer stream ends with exactly one checked answer"
    return answers[0]


def test_a_graph_with_history_still_answers_once(db_session: Session) -> None:
    # The guard is about the graph, not the message count: a second run on a reused graph must be
    # refused even when both runs carry earlier turns.
    model = ScriptedChatModel(script=[final("An answer with no citation.")])
    graph = agent(db_session, model)
    history = [("first question?", "first answer [S1].")]
    list(stream_answer(graph, "second question?", history=history))
    with pytest.raises(RuntimeError, match="one question"):
        list(stream_answer(graph, "third question?", history=history))


def test_a_follow_up_may_cite_an_alias_from_an_earlier_turn(
    db_session: Session, seed_items: Seed
) -> None:
    # The thread's registry is loaded before the graph starts, so S2 from the first turn is a
    # source this turn may cite without any tool returning it again.
    ids = seed_items(AGENT_ROWS)
    registry = SourceRegistry.from_refs(refs_for(db_session, ids, "swe", "mem"))
    model = ScriptedChatModel(
        script=[final("Agents that remember across sessions plan better [S2].")]
    )
    guard = CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW)
    graph = build_agent(
        [ChatModel("p0", model)], make_tools(db_session, FakeEmbedder(), registry), registry, guard
    )
    answer = answer_from_stream(
        list(
            stream_answer(
                graph,
                "does memory help?",
                registry=registry,
                history=[("what did you find?", "Coding agents are scored on real bugs [S1].")],
            )
        )
    )
    assert answer.supported
    assert [citation.alias for citation in answer.citations] == ["S2"]
    assert answer.citations[0].item_id == ids["mem"]


def test_an_alias_the_thread_never_issued_is_still_flagged(
    db_session: Session, seed_items: Seed
) -> None:
    # A registry restored from a thread is checked exactly like a fresh one: a thread that never
    # issued S7 cannot make the model's S7 a citation.
    ids = seed_items(AGENT_ROWS)
    registry = SourceRegistry.from_refs(refs_for(db_session, ids, "swe"))
    model = ScriptedChatModel(
        script=[
            final("Coding agents are scored on real bugs [S1]. Another lab beat it last week [S7].")
        ]
    )
    guard = CallGuard(Repository(db_session), Ledger(10_000), 800, lambda: NOW)
    graph = build_agent(
        [ChatModel("p0", model)], make_tools(db_session, FakeEmbedder(), registry), registry, guard
    )
    answer = answer_from_stream(
        list(stream_answer(graph, "q", registry=registry, history=[("q1", "an answer [S1].")]))
    )
    assert [flag.kind for flag in answer.flags] == ["unknown_source"]


def test_history_reaches_the_model_before_the_question(db_session: Session) -> None:
    model = ScriptedChatModel(script=[final("An answer with no citation.")])
    list(
        stream_answer(
            agent(db_session, model),
            "and then?",
            history=[("first?", "first answer [S1].")],
        )
    )
    sent = model.seen[0]
    assert [type(message).__name__ for message in sent] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert HISTORY_NOTE in str(sent[0].content)
    assert [str(message.content) for message in sent[1:]] == [
        "first?",
        "first answer [S1].",
        "and then?",
    ]


def test_a_single_turn_does_not_get_the_history_note(db_session: Session) -> None:
    model = ScriptedChatModel(script=[final("An answer with no citation.")])
    list(stream_answer(agent(db_session, model), "q"))
    assert HISTORY_NOTE not in str(model.seen[0][0].content)


def test_history_keeps_the_last_six_turns() -> None:
    turns = [(f"q{n}", f"a{n}") for n in range(1, 9)]
    assert bound_history(turns) == turns[-6:]


def test_history_drops_the_oldest_answers_past_the_character_budget() -> None:
    # The budget counts earlier answers only; the newest turns are the ones kept.
    turns = [("q1", "x" * 4000), ("q2", "y" * 4000), ("q3", "z" * 100)]
    assert bound_history(turns) == [("q2", "y" * 4000), ("q3", "z" * 100)]


def test_an_answer_over_the_budget_leaves_no_history() -> None:
    assert bound_history([("q1", "x" * 6001)]) == []


def test_an_answer_of_exactly_the_budget_is_kept() -> None:
    # The boundary is inclusive: 6,000 characters fit and 6,001 do not.
    assert bound_history([("q1", "x" * 6000)]) == [("q1", "x" * 6000)]
    assert bound_history([("q1", "x" * 6001)]) == []


def test_an_oversized_middle_turn_takes_the_older_ones_with_it() -> None:
    # Everything older than a turn that does not fit is dropped, not skipped over: a gap would
    # replay an answer without the turn it followed.
    turns = [("q1", "a" * 100), ("q2", "b" * 7000), ("q3", "c" * 100)]
    assert bound_history(turns) == [("q3", "c" * 100)]


def test_a_registry_survives_the_json_a_thread_stores(
    db_session: Session, seed_items: Seed
) -> None:
    ids = seed_items(AGENT_ROWS)
    refs = refs_for(db_session, ids, "swe", "mem")
    restored = SourceRegistry.from_refs([ref_from_json(ref_to_json(ref)) for ref in refs])
    # Aliases are kept exactly, so turn 5 can cite S2 from turn 1.
    assert [ref.alias for ref in restored.refs()] == ["S1", "S2"]
    assert restored.refs() == refs
    assert restored.get("S2") == refs[1]
