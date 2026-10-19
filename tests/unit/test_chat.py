"""The chat model layer: settings, provider wiring, the call guard, and the test double."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from paperwatch.config import CHAT_MODEL, GROQ_BASE_URL, OPENROUTER_BASE_URL, Settings
from paperwatch.llm.chat import (
    CallGuard,
    ChatNotConfigured,
    DailyCapReached,
    ScriptedChatModel,
    make_chat_models,
)
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
        # A recorded call is a call: the real repository would count it from now on.
        self.calls_today += 1


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


def test_chat_model_has_timeout_and_no_sdk_retries() -> None:
    # 0 retries: the SDK retrying inside a call would spend requests the daily cap never counts.
    for m in make_chat_models(keyed(groq_api_key="g", groq_chat_model="some-model")):
        assert (m.model.request_timeout, m.model.max_retries) == (60.0, 0)


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


def test_settings_ignore_a_developer_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A .env in the working directory is not test input: the captain keeps a real key there, and
    # the suite must stay green with it (review S1).
    (tmp_path / ".env").write_text("PW_CHAT_MODEL=other/model:free\nPW_GROQ_API_KEY=fake\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().chat_model == CHAT_MODEL


def test_settings_repr_hides_every_key() -> None:
    # A settings object lands in logs and tracebacks; repr must not be the way a key leaks.
    s = Settings(
        openrouter_api_key="sk-or-secret",
        groq_api_key="gsk-secret",
        azure_openai_api_key="az-secret",
        semantic_scholar_api_key="s2-secret",
    )
    shown = repr(s)
    assert "sk-or-secret" not in shown and "gsk-secret" not in shown
    assert "az-secret" not in shown and "s2-secret" not in shown


def test_failed_attempt_is_recorded_with_zero_tokens() -> None:
    # A 429 or a timeout is exactly when the next request hurts, so a failed attempt must be
    # counted against the cap too (review S3).
    repo, ledger = FakeRepo(), Ledger(100)
    CallGuard(repo, ledger, 800, lambda: NOW).after_failure("openrouter")
    assert repo.recorded == [
        {
            "at": NOW,
            "provider": "openrouter",
            "model": "failed",
            "input_tokens": 0,
            "output_tokens": 0,
        }
    ]
    assert ledger.tokens_used == 0  # a failed attempt spent no tokens


def test_every_attempt_passes_the_cap() -> None:
    # Two failed attempts against a cap of 2 use the day up, so the cap counts attempts, not
    # successes.
    repo = FakeRepo()
    guard = CallGuard(repo, Ledger(100), 2, lambda: NOW)
    guard.before_call()
    guard.after_failure("openrouter")
    guard.before_call()
    guard.after_failure("groq")
    with pytest.raises(DailyCapReached):
        guard.before_call()


def test_groq_model_id_without_a_key_builds_one_model() -> None:
    # ChatOpenAI(api_key=None) silently falls back to OPENAI_API_KEY from the environment, so a
    # Groq model built without its key would send the wrong key to Groq (review N1).
    assert [m.provider for m in make_chat_models(keyed(groq_chat_model="some-model"))] == [
        "openrouter"
    ]


def test_streamed_answers_ask_for_usage() -> None:
    # langchain-openai leaves stream_usage off for a custom base URL, so a streamed answer would
    # report no tokens and the ledger would charge nothing for it (review N1).
    models = make_chat_models(keyed(groq_api_key="g", groq_chat_model="some-model"))
    assert all(m.model.stream_usage for m in models)


def test_openrouter_sends_its_attribution_headers() -> None:
    # OpenRouter's free tier is rate-limited per app, and these headers are how it knows us.
    [primary] = make_chat_models(keyed())
    headers = primary.model.default_headers
    assert headers["HTTP-Referer"] == "https://github.com/enricoroncuzzi/paper-watch"
    assert headers["X-Title"] == "Paper Watch"


def test_cap_resets_at_utc_midnight_on_a_non_utc_clock() -> None:
    # A local clock just after local midnight is still the previous UTC day, so the cap must not
    # reset yet; astimezone(UTC) is what makes that true (review N2).
    berlin = timezone(timedelta(hours=2))
    now = datetime(2026, 10, 17, 0, 30, tzinfo=berlin)  # 2026-10-16T22:30Z
    seen: list[datetime] = []

    class RecordingRepo(FakeRepo):
        def chat_calls_since(self, at: datetime) -> int:
            seen.append(at)
            return 0

    CallGuard(RecordingRepo(), Ledger(100), 800, lambda: now).before_call()
    assert seen == [datetime(2026, 10, 16, tzinfo=UTC)]


def test_before_call_refuses_a_naive_clock() -> None:
    # A naive datetime would be read as local time by astimezone and shift the day silently. A
    # ValueError, not an assert: assertions vanish under `python -O` and this guard must not.
    naive = datetime(2026, 10, 16, 12)  # no timezone: that is the point of the test
    with pytest.raises(ValueError, match="aware"):
        CallGuard(FakeRepo(), Ledger(100), 800, lambda: naive).before_call()


def test_scripted_model_streams_a_tool_call() -> None:
    # Task 16's graph reads tool calls off the stream, so the double has to stream them the way a
    # provider does: as chunks that reassemble into the same call (review N3).
    call = {"name": "search_corpus", "args": {"q": "x"}, "id": "c1"}
    fake = ScriptedChatModel(script=[AIMessage("", tool_calls=[call])])
    chunks = list(fake.stream([HumanMessage("q")]))
    # Assert the fields, not the whole dict: pydantic adds its own `type` discriminator.
    [chunk] = chunks[0].tool_call_chunks or []
    assert (chunk["name"], chunk["args"], chunk["id"], chunk["index"]) == (
        "search_corpus",
        '{"q": "x"}',
        "c1",
        0,
    )
    # The chunk reassembles into the same call the model returned.
    assert chunks[0].tool_calls[0]["name"] == "search_corpus"
    assert chunks[0].tool_calls[0]["args"] == {"q": "x"}


def test_scripted_model_records_the_tools_it_was_bound() -> None:
    # Trajectory tests assert on bound_tools, so an unbound call must record [] and a bound one
    # the names, or a test could not tell a tool-less final round from a tool round (review N3).
    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    fake = ScriptedChatModel(script=[AIMessage("a"), AIMessage("b")])
    fake.bind_tools([_Named("search_corpus")]).invoke([HumanMessage("q")])
    fake.invoke([HumanMessage("q")])
    assert fake.bound_tools == [["search_corpus"], []]


def test_settings_defaults_hold_against_an_inherited_pw_variable() -> None:
    # The autouse fixture must clear inherited PW_* variables, not just hide .env. Running one test
    # in a subprocess with PW_CHAT_MODEL exported is the only way to prove it: in-process, the
    # fixture would clear the variable the test itself just set (review N1).
    # Run in the repo root: an IDE or script may start this from anywhere, and the relative node
    # id would otherwise be looked up under the parent's working directory.
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PW_CHAT_MODEL": "other/model:free"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "-q",
            "tests/unit/test_chat.py::test_settings_defaults",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
