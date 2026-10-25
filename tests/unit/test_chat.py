"""The chat model layer: settings, provider wiring, the call guard, and the test double."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from paperwatch.config import (
    CHAT_CALL_MODEL_MAX_CHARS,
    CHAT_MODEL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    Settings,
)
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
    """Stands in for Repository: counts, reserves and keeps completed rows in memory."""

    def __init__(self, calls_today: int = 0) -> None:
        self.calls_today = calls_today
        self.recorded: list[dict[str, object]] = []
        self.reservations: list[dict[str, object]] = []
        self._next = 0

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        if self.calls_today >= cap:
            return None, self.calls_today
        self.calls_today += 1
        self._next += 1
        self.reservations.append(
            {
                "id": self._next,
                "at": at,
                "provider": provider,
                "model": "reserved",
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        return self._next, self.calls_today

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        row = next(row for row in self.reservations if row["id"] == call_id)
        row.update(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        self.recorded.append(
            {
                "at": row["at"],
                "provider": row["provider"],
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def fail_chat_call(self, call_id: int) -> None:
        row = next(row for row in self.reservations if row["id"] == call_id)
        row["model"] = "failed"
        self.recorded.append(
            {
                "at": row["at"],
                "provider": row["provider"],
                "model": "failed",
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )


def usage(i: int, o: int) -> dict[str, int]:
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def keyed(**extra: object) -> Settings:
    return Settings(openrouter_api_key="or-key", **extra)  # type: ignore[arg-type]


def test_settings_defaults() -> None:
    s = Settings()
    assert s.chat_model == CHAT_MODEL
    assert (s.chat_max_calls_per_day, s.chat_max_tokens_per_answer) == (800, 60_000)
    assert s.chat_max_output_tokens == 4096
    assert s.openrouter_api_key is None and s.groq_api_key is None
    # Unset, so a draft uses the chat model; the server may name a different one.
    assert s.draft_model is None


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


def test_chat_output_cap_reaches_both_models() -> None:
    # A stream is not bounded by the read timeout, so one call needs its own output limit, or a
    # runaway generation can pass the per-answer budget in a single response.
    models = make_chat_models(keyed(groq_api_key="g", groq_chat_model="some-model"))
    assert [m.model.max_tokens for m in models] == [4096, 4096]


def test_chat_output_cap_is_sent_as_max_completion_tokens() -> None:
    # langchain-openai renames max_tokens to max_completion_tokens in the request payload, and
    # both providers document that field; Groq deprecates the max_tokens spelling.
    [primary] = make_chat_models(keyed(chat_max_output_tokens=1234))
    payload = primary.model._get_request_payload([HumanMessage("q")])
    assert payload["max_completion_tokens"] == 1234
    assert "max_tokens" not in payload


def test_call_guard_charges_real_usage() -> None:
    repo, ledger = FakeRepo(), Ledger(1000)
    guard = CallGuard(repo, ledger, 800, lambda: NOW)
    call_id = guard.before_call("openrouter")
    message = AIMessage("x", usage_metadata=usage(100, 20), response_metadata={"model_name": "m"})
    guard.after_call("openrouter", message, call_id)
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
    guard = CallGuard(repo, ledger, 800, lambda: NOW)
    call_id = guard.before_call("openrouter")
    with pytest.raises(BudgetExceeded):
        guard.after_call("openrouter", AIMessage("x", usage_metadata=usage(100, 20)), call_id)
    assert len(repo.recorded) == 1  # the tokens were spent, so the call is on record


def test_message_without_usage_is_charged_zero_and_recorded() -> None:
    repo, ledger = FakeRepo(), Ledger(100)
    guard = CallGuard(repo, ledger, 800, lambda: NOW)
    guard.after_call("openrouter", AIMessage("x"), guard.before_call("openrouter"))
    assert ledger.tokens_used == 0
    assert repo.recorded[0]["input_tokens"] == 0 and repo.recorded[0]["model"] == "unknown"


def test_before_call_refuses_at_the_daily_cap() -> None:
    with pytest.raises(DailyCapReached):
        CallGuard(FakeRepo(calls_today=800), Ledger(100), 800, lambda: NOW).before_call(
            "openrouter"
        )
    guard = CallGuard(FakeRepo(calls_today=799), Ledger(100), 800, lambda: NOW)
    call_id = guard.before_call("openrouter")
    assert call_id == 1


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


def test_failed_attempt_keeps_its_reserved_row_with_zero_tokens() -> None:
    # A 429 or a timeout is exactly when the next request hurts, so a failed attempt must be
    # counted against the cap too (review S3).
    repo, ledger = FakeRepo(), Ledger(100)
    guard = CallGuard(repo, ledger, 800, lambda: NOW)
    guard.after_failure("openrouter", guard.before_call("openrouter"))
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
    guard.after_failure("openrouter", guard.before_call("openrouter"))
    guard.after_failure("groq", guard.before_call("groq"))
    with pytest.raises(DailyCapReached):
        guard.before_call("openrouter")


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


def test_the_guard_passes_its_clock_to_the_reservation() -> None:
    # The store, not the guard, owns the UTC-midnight boundary (the integration test covers it);
    # the guard hands over the aware instant its injected clock was set to.
    berlin = timezone(timedelta(hours=2))
    now = datetime(2026, 10, 17, 0, 30, tzinfo=berlin)  # 2026-10-16T22:30Z
    seen: list[datetime] = []

    class RecordingRepo(FakeRepo):
        def reserve_chat_call(
            self, *, at: datetime, provider: str, cap: int
        ) -> tuple[int | None, int]:
            seen.append(at)
            return super().reserve_chat_call(at=at, provider=provider, cap=cap)

    CallGuard(RecordingRepo(), Ledger(100), 800, lambda: now).before_call("openrouter")
    assert seen == [now]


def test_before_call_refuses_a_naive_clock() -> None:
    # A naive datetime would be read as local time by astimezone and shift the day silently. A
    # ValueError, not an assert: assertions vanish under `python -O` and this guard must not.
    naive = datetime(2026, 10, 16, 12)  # no timezone: that is the point of the test
    with pytest.raises(ValueError, match="aware"):
        CallGuard(FakeRepo(), Ledger(100), 800, lambda: naive).before_call("openrouter")


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


def test_provider_base_url_and_key_beat_the_openai_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # openai/_client.py reads OPENAI_BASE_URL and OPENAI_API_KEY only when base_url and api_key are
    # None, and make_chat_models always passes both; the comment there rests on that.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://stray.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stray")
    models = make_chat_models(keyed(groq_api_key="g", groq_chat_model="some-model"))
    assert [m.model.openai_api_base for m in models] == [OPENROUTER_BASE_URL, GROQ_BASE_URL]
    assert [m.model.openai_api_key.get_secret_value() for m in models] == ["or-key", "g"]


def test_scripted_model_streams_text_before_a_tool_call() -> None:
    # A provider streams text it emits before the call, and invoke() already keeps that text; the
    # stream must agree, or a streamed answer loses the model's own words (post-acceptance fix).
    call = {"name": "search_corpus", "args": {"q": "x"}, "id": "c1"}
    fake = ScriptedChatModel(script=[AIMessage("let me search", tool_calls=[call])])
    chunks = list(fake.stream([HumanMessage("q")]))
    aggregate = chunks[0]
    for chunk in chunks[1:]:
        aggregate += chunk
    assert aggregate.content == "let me search"
    assert [(c["name"], c["args"], c["id"]) for c in aggregate.tool_calls] == [
        ("search_corpus", {"q": "x"}, "c1")
    ]


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


def test_a_streamed_model_id_repeated_by_the_merge_is_recorded_once() -> None:
    # langchain-core merges streamed chunks by concatenating string metadata, so a provider that
    # names the model on several chunks leaves "idid"; the guard stores the id once.
    name = "vendor/model:free"
    chunks = [AIMessageChunk(content="a", response_metadata={"model_name": name}) for _ in range(3)]
    merged = chunks[0] + chunks[1] + chunks[2]
    assert merged.response_metadata["model_name"] == name * 3  # langchain-core's own merge
    repo = FakeRepo()
    guard = CallGuard(repo, Ledger(1000), 10, lambda: NOW)
    guard.after_call(
        "p",
        AIMessage("x", usage_metadata=usage(1, 1), response_metadata=merged.response_metadata),
        guard.before_call("p"),
    )
    assert repo.recorded[0]["model"] == name


def test_a_model_id_longer_than_the_column_is_clipped() -> None:
    # The id is written to ChatCall.model (String(128)); a provider-left id longer than the
    # column must not turn a usage write into an error that ends the answer.
    name = "vendor/" + "a" * 193 + "/end"
    assert len(name) > CHAT_CALL_MODEL_MAX_CHARS
    repo = FakeRepo()
    guard = CallGuard(repo, Ledger(1000), 10, lambda: NOW)
    guard.after_call(
        "p",
        AIMessage("x", usage_metadata=usage(1, 1), response_metadata={"model_name": name}),
        guard.before_call("p"),
    )
    assert len(str(repo.recorded[0]["model"])) == CHAT_CALL_MODEL_MAX_CHARS


def test_draft_models_are_one_openrouter_model_and_skip_groq() -> None:
    # A draft carries the captain's own point of view, so it goes to one provider only: a second
    # provider configured in the environment for answers must not be reached at all.
    models = make_chat_models(
        keyed(groq_api_key="g", groq_chat_model="some-model"), purpose="draft"
    )
    [draft] = models
    assert draft.provider == "openrouter"
    assert draft.model.openai_api_base == OPENROUTER_BASE_URL


def test_draft_and_answer_models_send_no_provider_block() -> None:
    # The owner accepted that the free provider may train on the draft prompt (ADR 0015), so a
    # draft's wire body is an answer's: no `provider` block, no data-collection rule.
    purposes: tuple[Literal["answer", "draft"], ...] = ("answer", "draft")
    for purpose in purposes:
        [model] = make_chat_models(keyed(), purpose=purpose)
        assert model.model.extra_body is None
        payload = model.model._get_request_payload([HumanMessage("q")])
        assert "provider" not in payload


def test_draft_models_use_the_draft_model_when_set() -> None:
    [draft] = make_chat_models(keyed(draft_model="vendor/draft:free"), purpose="draft")
    assert draft.model.model_name == "vendor/draft:free"
    payload = draft.model._get_request_payload([HumanMessage("q")])
    assert payload["model"] == "vendor/draft:free"


def test_draft_models_use_the_chat_model_when_unset() -> None:
    [draft] = make_chat_models(keyed(), purpose="draft")
    assert draft.model.model_name == CHAT_MODEL


def test_answer_models_ignore_the_draft_model() -> None:
    # The draft setting is the draft's alone: an answer keeps the model the QA eval measured.
    [primary] = make_chat_models(keyed(draft_model="vendor/draft:free"))
    assert primary.model.model_name == CHAT_MODEL
    assert primary.model.extra_body is None
