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
from langchain_core.messages import AIMessage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperwatch.agent.citations import NO_SUPPORT
from paperwatch.agent.graph import (
    AnswerEvent,
    TokenEvent,
    ToolEvent,
    answer_question,
    build_agent,
    stream_answer,
)
from paperwatch.agent.prompts import FINAL_ROUND_NOTE, SYSTEM_PROMPT
from paperwatch.agent.sources import SourceRegistry
from paperwatch.agent.tools import make_tools
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
    assert model.bound_tools[4] == [] and model.seen[4][-1].content == FINAL_ROUND_NOTE
    assert answer.supported


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
    assert "429 from the provider" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)


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


def test_tools_are_bound_without_parallel_calls(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench"),
            final("SWE-bench Verified measures coding agents on real bugs [S1]."),
        ]
    )
    answer_question(agent(db_session, model), "q")
    # One bind at build time. ToolNode runs one message's tool calls on a thread pool otherwise,
    # and the tools share one session and one registry.
    assert model.bound_kwargs == [{"parallel_tool_calls": False}]


def test_stream_answer_yields_tools_tokens_then_one_answer(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    text = "SWE-bench Verified measures coding agents on real bugs [S1]."
    model = ScriptedChatModel(script=[call("search_corpus", query="SWE-bench"), final(text)])
    events = list(stream_answer(agent(db_session, model), "q"))
    kinds = [type(e) for e in events]
    assert kinds[0] is ToolEvent and kinds[-1] is AnswerEvent and kinds.count(AnswerEvent) == 1
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
