"""The summary writer against a scripted chat model (Task 35)."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from paperwatch.llm.chat import CallGuard, ChatModel, DailyCapReached, ScriptedChatModel
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.suggestions.summary import SummaryWriter
from tests.unit.test_summary_check import TEXT, TITLE, good


class Store:
    """A chat-call store that only needs to reserve, fill and fail one row."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.filled = 0
        self.failed = 0

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        return (None, cap) if self.refuse else (1, 1)

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        self.filled += 1

    def fail_chat_call(self, call_id: int) -> None:
        self.failed += 1


def writer(script: list[Any], *, refuse: bool = False) -> tuple[SummaryWriter, ScriptedChatModel]:
    model = ScriptedChatModel(script=script)
    store = Store(refuse=refuse)
    guard = CallGuard(
        store=store,
        ledger=Ledger(max_tokens=10_000),
        max_calls_per_day=10,
        clock=lambda: datetime(2026, 11, 2, 9, tzinfo=UTC),
    )
    return SummaryWriter([ChatModel("openrouter", model)], lambda: guard), model


def answer(raw: dict[str, Any]) -> AIMessage:
    # Usage is reported, as a real provider does, so the guard's ledger is actually charged.
    return AIMessage(
        content=json.dumps(raw),
        usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    )


def test_a_good_first_answer_is_checked_with_one_call() -> None:
    summary_writer, model = writer([answer(good())])

    summary = summary_writer.write(title=TITLE, text=TEXT)

    assert summary.status == "checked"
    assert summary.what is not None and summary.what.quote is not None
    assert summary.why is not None and summary.why.interpretation is True
    assert summary.catch is not None and summary.catch.interpretation is True
    assert len(model.seen) == 1


def test_a_failed_check_retries_once_naming_the_check() -> None:
    broken = good()
    broken["result"]["text"] = "Llama-3.1-8B agents solve 45% of the tasks against 28%."
    summary_writer, model = writer([answer(broken), answer(good())])

    summary = summary_writer.write(title=TITLE, text=TEXT)

    assert summary.status == "checked"
    assert len(model.seen) == 2
    retry = str(model.seen[1][-1].content)
    assert "number:result:45%" in retry


def test_two_failures_fall_back_to_the_excerpt() -> None:
    broken = good()
    broken["what"]["text"] = "The authors compare AgentBench-Lite with GPT-5."
    summary_writer, _ = writer([answer(broken), answer(broken)])

    summary = summary_writer.write(title=TITLE, text=TEXT)

    assert summary.status == "excerpt"
    assert summary.what is None
    assert summary.excerpt.startswith("We introduce AgentBench-Lite")
    assert "name:what:GPT-5" in summary.failed_checks


def test_unparseable_json_twice_falls_back_to_the_excerpt() -> None:
    summary_writer, _ = writer([AIMessage(content="not json"), AIMessage(content="{oops")])

    summary = summary_writer.write(title=TITLE, text=TEXT)

    assert summary.status == "excerpt"
    assert summary.failed_checks == ("shape",)


def test_daily_cap_propagates() -> None:
    summary_writer, _ = writer([answer(good())], refuse=True)

    with pytest.raises(DailyCapReached):
        summary_writer.write(title=TITLE, text=TEXT)


def test_budget_exceeded_propagates() -> None:
    model = ScriptedChatModel(script=[answer(good())])
    guard = CallGuard(
        store=Store(),
        # A ledger already over its cap: the charge after the call raises, as it does for an answer.
        ledger=Ledger(max_tokens=0),
        max_calls_per_day=10,
        clock=lambda: datetime(2026, 11, 2, 9, tzinfo=UTC),
    )
    summary_writer = SummaryWriter([ChatModel("openrouter", model)], lambda: guard)

    with pytest.raises(BudgetExceeded):
        summary_writer.write(title=TITLE, text=TEXT)
