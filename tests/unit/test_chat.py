"""The chat model layer: settings, provider wiring, the call guard, and the test double."""

from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from paperwatch.config import CHAT_MODEL, GROQ_BASE_URL, OPENROUTER_BASE_URL, Settings
from paperwatch.llm.chat import CallGuard, ChatNotConfigured, ScriptedChatModel, make_chat_models
from paperwatch.llm.ledger import BudgetExceeded, Ledger

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


class FakeRepo:
    """Stands in for Repository: counts and keeps recorded calls in memory."""

    def __init__(self, calls_today: int = 0) -> None:
        self.calls_today = calls_today
        self.recorded: list[dict[str, object]] = []

    def chat_calls_since(self, at: datetime) -> int:
        return self.calls_today

    def record_chat_call(self, **row: object) -> None:
        self.recorded.append(row)


def usage(i: int, o: int) -> dict[str, int]:
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def keyed(**extra: object) -> Settings:
    return Settings(openrouter_api_key="or-key", **extra)  # type: ignore[arg-type]


def test_settings_defaults() -> None:
    s = Settings()
    assert s.chat_model == CHAT_MODEL
    assert (s.chat_max_calls_per_day, s.chat_max_tokens_per_answer) == (800, 60_000)
    assert s.openrouter_api_key is None and s.groq_api_key is None


def test_missing_openrouter_key_is_refused() -> None:
    with pytest.raises(ChatNotConfigured, match="PW_OPENROUTER_API_KEY"):
        make_chat_models(Settings())


def test_openrouter_is_primary() -> None:
    [primary] = make_chat_models(keyed())
    assert primary.provider == "openrouter"
    assert primary.model.openai_api_base == OPENROUTER_BASE_URL
    assert primary.model.model_name == CHAT_MODEL


def test_groq_fallback_needs_key_and_model() -> None:
    assert len(make_chat_models(keyed(groq_api_key="g"))) == 1
    models = make_chat_models(keyed(groq_api_key="g", groq_chat_model="some-model"))
    assert [m.provider for m in models] == ["openrouter", "groq"]
    assert models[1].model.openai_api_base == GROQ_BASE_URL


def test_chat_model_has_timeout_and_retries() -> None:
    for m in make_chat_models(keyed(groq_api_key="g", groq_chat_model="some-model")):
        assert (m.model.request_timeout, m.model.max_retries) == (60.0, 2)


def test_call_guard_charges_real_usage() -> None:
    repo, ledger = FakeRepo(), Ledger(1000)
    message = AIMessage("x", usage_metadata=usage(100, 20), response_metadata={"model_name": "m"})
    CallGuard(repo, ledger, 800, lambda: NOW).after_call("openrouter", message)
    assert ledger.tokens_used == 120
    assert repo.recorded == [
        {
            "at": NOW,
            "provider": "openrouter",
            "model": "m",
            "input_tokens": 100,
            "output_tokens": 20,
        }
    ]


def test_call_guard_raises_past_the_answer_budget() -> None:
    repo, ledger = FakeRepo(), Ledger(100)
    with pytest.raises(BudgetExceeded):
        CallGuard(repo, ledger, 800, lambda: NOW).after_call(
            "openrouter", AIMessage("x", usage_metadata=usage(100, 20))
        )
    assert len(repo.recorded) == 1  # the tokens were spent, so the call is on record


def test_message_without_usage_is_charged_zero_and_recorded() -> None:
    repo, ledger = FakeRepo(), Ledger(100)
    CallGuard(repo, ledger, 800, lambda: NOW).after_call("openrouter", AIMessage("x"))
    assert ledger.tokens_used == 0
    assert repo.recorded[0]["input_tokens"] == 0 and repo.recorded[0]["model"] == "unknown"


def test_before_call_refuses_at_the_daily_cap() -> None:
    from paperwatch.llm.chat import DailyCapReached

    with pytest.raises(DailyCapReached):
        CallGuard(FakeRepo(calls_today=800), Ledger(100), 800, lambda: NOW).before_call()
    CallGuard(FakeRepo(calls_today=799), Ledger(100), 800, lambda: NOW).before_call()


def test_scripted_model_replays_records_and_raises() -> None:
    fake = ScriptedChatModel(script=[AIMessage("one"), RuntimeError("429")])
    bound = fake.bind_tools([])
    assert bound.invoke([HumanMessage("q")]).content == "one"
    with pytest.raises(RuntimeError, match="429"):
        fake.invoke([HumanMessage("q2")])
    assert [m[-1].content for m in fake.seen] == ["q", "q2"]
    with pytest.raises(AssertionError, match="script exhausted"):
        fake.invoke([HumanMessage("q3")])


def test_scripted_model_streams_words_then_usage() -> None:
    fake = ScriptedChatModel(script=[AIMessage("a b c", usage_metadata=usage(5, 3))])
    chunks = list(fake.stream([HumanMessage("q")]))
    assert "".join(str(c.content) for c in chunks) == "a b c"
    assert chunks[-1].usage_metadata == usage(5, 3)
