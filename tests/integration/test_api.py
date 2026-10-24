"""The streaming API: the SSE contract, the error events, and the session it releases.

Every test drives the real graph and the real tools over the real test database; only the chat
model is scripted, so the API's behaviour is pinned without a key, a network call or a live model.
The disconnect tests drive the app through raw ASGI, because a `TestClient` collects the whole
stream and never hangs up.
"""

import asyncio
import contextlib
import json
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import anyio
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from sqlalchemy import Engine, delete, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.orm import Session
from starlette.responses import ClientDisconnect
from structlog.testing import capture_logs

from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer
from paperwatch.agent.graph import AnswerAborted, AnswerEvent, ToolEvent
from paperwatch.agent.sources import SourceRegistry
from paperwatch.agent.tools import make_tools
from paperwatch.api import app as api
from paperwatch.api.app import (
    AgentDeps,
    AnswerData,
    ErrorData,
    StreamItem,
    TokenData,
    ToolData,
    create_app,
    run_question,
)
from paperwatch.api.auth import HmacVerifier, mint_hmac_token
from paperwatch.api.limits import Admitted, RequestLimiter
from paperwatch.api.search_mode import ModeDecision
from paperwatch.config import (
    ANSWER_POOL_OVERFLOW,
    ANSWER_POOL_SIZE,
    ANSWER_POOL_TIMEOUT_S,
    CONCURRENT_ANSWERS,
    CONNECT_TIMEOUT_S,
    LOOPBACK_HOSTS,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    USAGE_POOL_SIZE,
    Settings,
)
from paperwatch.domain import load_profiles
from paperwatch.llm.chat import ChatModel, ScriptedChatModel
from paperwatch.llm.embedder import FakeEmbedder, WorkerEmbedder
from paperwatch.retrieval.hybrid import Filters
from paperwatch.store.db import make_engine, make_session_factory
from paperwatch.store.models import ChatCall
from paperwatch.store.repository import ChatCallLog
from tests.conftest import AGENT_ROWS, SeedRow

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
ANSWER = "SWE-bench Verified measures coding agents on real repository bugs [S1]."
SEED_USAGE = {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
PROFILES = frozenset({"agentic-engineering", "ai-forensics", "ml-architectures"})
Seed = Callable[[list[SeedRow]], dict[str, UUID]]
# The model id a `ChatCall` row gets when the provider answered without naming one.
UNKNOWN_MODEL = "unknown"
# The gate's credential for the whole module: a real HMAC verifier, so the tests exercise the same
# check production runs instead of a stub that would accept anything.
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret


class AuthedClient(TestClient):
    """A client that mints a fresh single-use token for every request, as a real caller does."""

    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", None) or {})
        if "authorization" not in {name.lower() for name in headers}:
            headers["Authorization"] = f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}"
        return super().request(method, url, headers=headers, **kwargs)


def client(app: Any) -> AuthedClient:
    # localhost, not TestClient's default testserver, so the host allow-list accepts the request.
    return AuthedClient(app, base_url="http://localhost")


class FixedMode:
    """A search-mode gate with one answer, so an API test does not need a database or /proc."""

    def __init__(self, decision: ModeDecision) -> None:
        self.decision = decision

    def current(self) -> ModeDecision:
        return self.decision


class MemoryChatCalls:
    """The `ChatCallStore` the API tests need: a count in memory, never a database row.

    The tests below share one database session, and `run_question` marks it read-only, so a real
    `ChatCallLog` (which commits on its own session) has no place here. The production store is
    covered by `tests/integration/test_chat_calls.py`.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[datetime, str]] = []

    def record_chat_call(
        self,
        *,
        at: datetime,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.rows.append((at, model))

    def chat_calls_since(self, at: datetime) -> int:
        return sum(1 for call_at, _ in self.rows if call_at >= at)

    @property
    def models(self) -> list[str]:
        return [model for _, model in self.rows]


def scripted(*steps: AIMessage | Exception) -> list[ChatModel]:
    return [ChatModel("openrouter", ScriptedChatModel(script=list(steps)))]


def search_then(text: str = ANSWER, **args: Any) -> list[ChatModel]:
    tool_args = {"query": "SWE-bench coding agents", **args}
    return scripted(
        AIMessage("", tool_calls=[{"name": "search_corpus", "args": tool_args, "id": "c1"}]),
        AIMessage(text),
    )


def deps(
    db_session: Session,
    models: list[ChatModel],
    store: MemoryChatCalls | None = None,
    embedder: Any = None,
    search_mode: Any = None,
    **settings: object,
) -> AgentDeps:
    # Every request reuses the test session so seeded rows are visible. run_question closes it;
    # with the fixture's savepoint join, the outer test transaction still rolls back.
    return AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=embedder or FakeEmbedder(),
        models=models,
        settings=Settings(**settings),  # type: ignore[arg-type]
        clock=lambda: NOW,
        chat_calls=store or MemoryChatCalls(),
        profile_names=PROFILES,
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=search_mode or FixedMode(ModeDecision("hybrid", "ok")),
    )


def events(body: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ask(app_deps: AgentDeps, payload: dict[str, Any]) -> Any:
    return client(create_app(app_deps)).post("/ask", json=payload)


def test_healthz(db_session: Session) -> None:
    assert client(create_app(deps(db_session, []))).get("/healthz").json() == {"status": "ok"}


def test_readyz_checks_the_database(db_session: Session) -> None:
    assert client(create_app(deps(db_session, []))).get("/readyz").status_code == 200
    broken = deps(db_session, [])
    broken.session_factory = lambda: (_ for _ in ()).throw(RuntimeError("db down"))  # type: ignore[assignment,return-value]
    response = client(create_app(broken)).get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


class _FullLimiter:
    """A capacity limiter with no free slot: every readiness check is answered 503 at once."""

    def acquire_nowait(self) -> None:
        raise anyio.WouldBlock

    def release(self) -> None:
        raise AssertionError("a slot that was never taken must not be released")


def test_readyz_answers_every_time(db_session: Session) -> None:
    # The limiter slot is released in a finally, so a check that timed out cannot leave /readyz
    # answering 503 forever: the server would look permanently unready.
    client_ = client(create_app(deps(db_session, [])))
    assert [client_.get("/readyz").status_code for _ in range(5)] == [200] * 5


def test_readyz_uses_the_readiness_factory(db_session: Session, engine: Engine) -> None:
    # Readiness checks the database through its own small engine in production, so a saturated
    # answer pool does not read as "the database is down".
    seen: list[int] = []
    factory = make_session_factory(engine)

    def ready() -> Session:
        seen.append(1)
        return factory()

    app_deps = deps(db_session, [])
    app_deps.ready_session_factory = ready
    assert client(create_app(app_deps)).get("/readyz").status_code == 200
    assert seen == [1]


def test_readyz_is_503_when_the_check_slots_are_full(db_session: Session) -> None:
    app = create_app(deps(db_session, []))
    app.state.ready_limiter = _FullLimiter()
    response = client(app).get("/readyz")
    assert response.status_code == 503 and response.json() == {"status": "unavailable"}


def test_readyz_times_out_while_healthz_still_answers(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A paused database leaves a check blocked with a connection open. The probe must answer 503
    # within its bound, and /healthz must still answer: it never touches the database.
    monkeypatch.setattr(api, "READY_TIMEOUT_S", 0.05)
    entered, release = threading.Event(), threading.Event()

    def blocking_factory() -> Session:
        # The check thread is abandoned by the timeout, so it must never touch the shared test
        # session; it only has to stay blocked until the test releases it.
        entered.set()
        release.wait(10)
        raise RuntimeError("the database is paused")

    app_deps = deps(db_session, [])
    app_deps.session_factory = blocking_factory  # type: ignore[assignment]
    app = create_app(app_deps)
    results: dict[str, Any] = {}

    def call_readyz() -> None:
        results["ready"] = client(app).get("/readyz")

    worker = threading.Thread(target=call_readyz)
    try:
        worker.start()
        assert entered.wait(10)
        started = time.monotonic()
        health = client(app).get("/healthz")
        assert health.status_code == 200
        assert time.monotonic() - started < 1.0
        worker.join(10)
    finally:
        release.set()
    assert results["ready"].status_code == 503


def test_ask_streams_tool_token_answer_in_order(db_session: Session, seed_items: Seed) -> None:
    ids = seed_items(AGENT_ROWS)
    response = ask(
        deps(db_session, search_then()), {"question": "how are coding agents evaluated?"}
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    parsed = events(response.text)
    names = [name for name, _ in parsed]
    assert names[0] == "tool" and names[-1] == "answer" and names.count("answer") == 1
    assert "token" in names
    answer = parsed[-1][1]
    assert answer["supported"] is True
    citation = answer["citations"][0]
    assert citation["id"] == "S1"
    assert citation["tier"] == "peer_reviewed"
    assert citation["url"] == "https://example.org/swe"
    assert citation["item_id"] == str(ids["swe"])  # the stored id, as text
    assert answer["flags"] == []


def test_every_streamed_payload_validates_against_its_model(
    db_session: Session, seed_items: Seed
) -> None:
    # The Pydantic models are the one definition of the stream's shape: the OpenAPI contract
    # documents them and the route builds every payload from them.
    seed_items(AGENT_ROWS)
    response = ask(deps(db_session, search_then()), {"question": "q"})
    from paperwatch.api.app import NoticeData, SourceData, StepData

    models: dict[str, Any] = {
        "tool": ToolData,
        "step": StepData,
        "source": SourceData,
        "token": TokenData,
        "notice": NoticeData,
        "answer": AnswerData,
        "error": ErrorData,
    }
    for name, data in events(response.text):
        assert isinstance(models[name].model_validate(data), models[name])
    assert StreamItem.model_validate({"event": "tool", "data": {"name": "x", "args": {}}})


def test_blank_question_is_refused(db_session: Session) -> None:
    assert ask(deps(db_session, []), {"question": "   "}).status_code == 422


def test_the_question_is_stripped_before_it_reaches_the_agent(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def record(
        deps_: AgentDeps,
        question: str,
        filters: Filters,
        cancel: Any = None,
        search_log: Any = None,
    ) -> Any:
        seen.append(question)
        yield AnswerEvent(CheckedAnswer(NO_SUPPORT, (), (), False))

    monkeypatch.setattr(api, "run_question", record)
    ask(deps(db_session, []), {"question": "  how are agents evaluated?  "})
    assert seen == ["how are agents evaluated?"]


def test_request_filters_reach_the_answer(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Filters] = []

    def record(
        deps_: AgentDeps,
        question: str,
        filters: Filters,
        cancel: Any = None,
        search_log: Any = None,
    ) -> Any:
        seen.append(filters)
        yield AnswerEvent(CheckedAnswer(NO_SUPPORT, (), (), False))

    monkeypatch.setattr(api, "run_question", record)
    ask(
        deps(db_session, []),
        {
            "question": "q",
            "tiers": ["peer_reviewed"],
            "profiles": ["ai-forensics"],
            "since": "2026-10-01",
        },
    )
    assert seen == [
        Filters(
            tiers=frozenset({"peer_reviewed"}),
            profiles=frozenset({"ai-forensics"}),
            since=datetime(2026, 10, 1, tzinfo=UTC),
        )
    ]


def test_request_filters_constrain_every_search(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    models = search_then(
        "Agents plan better with memory over long tasks [S1].",
        query="agents",
        tiers=["vetted_preprint", "lab_blog", "peer_reviewed"],
    )
    response = ask(deps(db_session, models), {"question": "agents?", "tiers": ["peer_reviewed"]})
    answer = events(response.text)[-1][1]
    assert {c["tier"] for c in answer["citations"]} == {"peer_reviewed"}


def test_an_unknown_profile_is_refused(db_session: Session) -> None:
    response = ask(deps(db_session, []), {"question": "q", "profiles": ["not-a-profile"]})
    assert response.status_code == 422
    assert "not-a-profile" in response.text
    # The contract documents 422 as an array of error objects, so a typed client can parse it.
    [error] = response.json()["detail"]
    assert error["msg"] == "unknown profiles: not-a-profile"
    assert error["loc"] == ["body", "profiles"]


def test_a_future_since_is_refused(db_session: Session) -> None:
    tomorrow = (NOW + timedelta(days=1)).date().isoformat()
    response = ask(deps(db_session, []), {"question": "q", "since": tomorrow})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["msg"] == "since must not be in the future"
    assert error["loc"] == ["body", "since"]


def test_today_is_an_acceptable_since_boundary(db_session: Session) -> None:
    # `since` is a date, so "today" is the last value that is not in the future: the whole of
    # today is searched, not refused.
    today = NOW.date().isoformat()
    response = ask(deps(db_session, search_then()), {"question": "q", "since": today})
    assert response.status_code == 200


def test_profile_lists_are_bounded(db_session: Session) -> None:
    too_many = [f"p{i}" for i in range(50)]
    assert ask(deps(db_session, []), {"question": "q", "profiles": too_many}).status_code == 422
    long_item = "p" * 200
    assert ask(deps(db_session, []), {"question": "q", "profiles": [long_item]}).status_code == 422


def test_model_failure_streams_an_error_event(db_session: Session) -> None:
    response = ask(deps(db_session, scripted(RuntimeError("503"))), {"question": "q"})
    assert response.status_code == 200  # the stream had started, so the error travels inside it
    name, data = events(response.text)[-1]
    assert (name, data["code"]) == ("error", "model_unavailable") and data["message"]


def test_provider_text_stays_out_of_the_stream_and_the_logs(
    db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    # An SDK error's body can echo the flagged question, and ModelUnavailable keeps it as its
    # cause on purpose. run_question must not log the cause or put it in the event, through
    # structlog or through a library's stdlib logger.
    models = scripted(RuntimeError("provider body: the captain's secret question"))
    with capture_logs() as logs:
        response = ask(deps(db_session, models), {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["code"]) == ("error", "model_unavailable")
    assert "provider body" not in data["message"]
    assert "provider body" not in str(logs)
    assert "provider body" not in caplog.text


def test_an_internal_failure_logs_the_site_and_no_text(
    db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    # The internal branch serves an unforeseen failure. Its traceback (and any chained cause)
    # could hold provider or question text, so messages are never logged, but the code site is:
    # an operator needs to know which stage failed.
    def broken_factory() -> Session:
        raise RuntimeError("provider body: the captain's secret question")

    app_deps = deps(db_session, [])
    app_deps.session_factory = broken_factory
    with capture_logs() as logs:
        response = ask(app_deps, {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["code"]) == ("error", "internal")
    failures = [entry for entry in logs if entry["event"] == "ask_failed"]
    assert [entry["error_type"] for entry in failures] == ["RuntimeError"]
    assert any("broken_factory" in frame for frame in failures[0]["site"])
    assert all("exc_info" not in entry for entry in logs)
    assert "secret question" not in str(logs)
    assert "secret question" not in caplog.text


def test_an_internal_failure_sends_only_the_fixed_message(db_session: Session) -> None:
    # The event body is a fixed sentence: a failure's text can hold a provider body or the question.
    def broken_factory() -> Session:
        raise RuntimeError("provider body: the captain's secret question")

    app_deps = deps(db_session, [])
    app_deps.session_factory = broken_factory
    response = ask(app_deps, {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["code"], data["message"]) == ("error", "internal", api.INTERNAL_MESSAGE)
    assert "secret question" not in response.text


def test_the_internal_log_names_the_cause_type(db_session: Session) -> None:
    # `cause_type` is how an operator tells a driver error from a bug without reading its text.
    def broken_factory() -> Session:
        try:
            raise ValueError("the real cause")
        except ValueError as exc:
            raise RuntimeError("provider body") from exc

    app_deps = deps(db_session, [])
    app_deps.session_factory = broken_factory
    with capture_logs() as logs:
        ask(app_deps, {"question": "q"})
    [failure] = [entry for entry in logs if entry["event"] == "ask_failed"]
    assert (failure["error_type"], failure["cause_type"]) == ("RuntimeError", "ValueError")

    # A failure with no chained cause logs None rather than a missing key.
    def plain_factory() -> Session:
        raise RuntimeError("no cause")

    app_deps.session_factory = plain_factory
    with capture_logs() as logs:
        ask(app_deps, {"question": "q"})
    [failure] = [entry for entry in logs if entry["event"] == "ask_failed"]
    assert failure["cause_type"] is None


def test_a_library_under_a_paperwatch_directory_is_not_application_code(tmp_path: Path) -> None:
    # A checkout (or a virtualenv) under a directory named `paperwatch` used to match every frame,
    # so a library frame was logged as the application's and the real one was not. The filter
    # compares against the package's own directory instead.
    lib = tmp_path / "paperwatch"
    lib.mkdir()
    (lib / "fakelib.py").write_text(
        "from paperwatch.api import app\n\n\ndef boom():\n    app._select_one(None)\n"
    )
    sys.path.insert(0, str(lib))
    try:
        import fakelib  # a module that only exists for this test

        with pytest.raises(TypeError) as raised:
            fakelib.boom()
    finally:
        sys.path.remove(str(lib))
        sys.modules.pop("fakelib", None)
    # The only application frame is the one that called the failing factory, and it is the whole
    # site: the `paperwatch/fakelib.py` frame is a stranger's code despite its directory name.
    site = api._failure_site(raised.value)
    assert len(site) == 1 and site[0].startswith("app.py:"), site


def test_daily_cap_streams_an_error_event(db_session: Session) -> None:
    response = ask(deps(db_session, search_then(), chat_max_calls_per_day=1), {"question": "q"})
    # The first call fits the cap of 1; the second is refused.
    assert events(response.text)[-1][0] == "error"
    assert events(response.text)[-1][1]["code"] == "daily_cap"


def test_budget_exhaustion_streams_an_error_event(db_session: Session) -> None:
    # The first model call's reported usage already exceeds the per-answer budget of 1 token, so
    # the ledger refuses the answer after recording the call.
    model = AIMessage(
        "",
        tool_calls=[{"name": "search_corpus", "args": {"query": "x"}, "id": "c1"}],
        usage_metadata=SEED_USAGE,
    )
    response = ask(
        deps(db_session, scripted(model, AIMessage(ANSWER)), chat_max_tokens_per_answer=1),
        {"question": "q"},
    )
    assert events(response.text)[-1][1]["code"] == "budget"


def test_a_saturated_pool_is_a_busy_server_not_a_bug(db_session: Session) -> None:
    # The answer pool is small and its wait short, so a full pool arrives as a pool timeout. The
    # client is told to retry, and the pool's own text never reaches the stream.
    def exhausted() -> Session:
        raise PoolTimeout("QueuePool limit of size 6 overflow 2 reached, connection timed out")

    app_deps = deps(db_session, [])
    app_deps.session_factory = exhausted
    response = ask(app_deps, {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["code"], data["message"]) == (
        "error",
        "internal",
        api.BUSY_MESSAGE,
    )
    assert "QueuePool" not in response.text


def test_closing_the_stream_closes_the_session(db_session: Session, seed_items: Seed) -> None:
    seed_items(AGENT_ROWS)
    closed: list[bool] = []

    class Spy:
        def __init__(self, inner: Session) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def close(self) -> None:
            closed.append(True)

    app_deps = deps(db_session, search_then())
    app_deps.session_factory = lambda: Spy(db_session)  # type: ignore[assignment,return-value]
    stream: Any = run_question(app_deps, "q", Filters())
    next(stream)
    stream.close()  # what Starlette does when the client disconnects
    assert closed == [True]


def test_closing_the_sse_stream_closes_the_answer(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The SSE generator closes the answer explicitly. The test keeps its own reference to the
    # answer's generator, which is what a reference-counted runtime would otherwise finalise only
    # when the frame dies, so the explicit close is observable.
    closed: list[bool] = []
    created: list[Any] = []
    handed: list[Any] = []

    def answer(
        deps: AgentDeps,
        question: str,
        filters: Filters,
        cancel: Any = None,
        search_log: Any = None,
    ) -> Any:
        def events() -> Any:
            try:
                yield ToolEvent("search_corpus", {"query": "x"})
            finally:
                closed.append(True)

        handed.append(cancel)
        created.append(events())
        return created[-1]

    monkeypatch.setattr(api, "run_question", answer)
    cancel = threading.Event()
    stream = api._stream(_dummy_deps(), api.AskRequest(question="q"), cancel)
    next(stream)
    stream.close()
    assert closed == [True] and len(created) == 1
    # The response's own event is the answer's abort signal, so a hang-up reaches the model call
    # without waiting for this generator to be closed.
    assert handed == [cancel]


def test_the_response_sets_the_cancel_event_when_the_client_hangs_up() -> None:
    # Starlette's listener is the only thing that sees a hang-up while the response task waits for
    # the next chunk; without setting the flag there, a tool round would start one more request.
    cancel = threading.Event()
    response = api._ClosingStream((chunk for chunk in ()), cancel, media_type="text/event-stream")

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    asyncio.run(response.listen_for_disconnect(receive))
    assert cancel.is_set()


def _scope(spec_version: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "path": "/ask",
        "raw_path": b"/ask",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"localhost"),
            (
                b"authorization",
                f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}".encode(),
            ),
        ],
        "server": ("testserver", 80),
        "client": ("testclient", 1),
        "scheme": "http",
        "root_path": "",
    }


async def _hang_up_mid_stream(
    app: Any,
    payload: dict[str, Any],
    *,
    spec_version: str,
    after_chunks: int = 1,
) -> None:
    """Drive the app through raw ASGI and hang up after `after_chunks` body chunks.

    A `TestClient` reads the whole stream, so only this can show what a real server does with a
    client that disconnects. Under ASGI 2.3 Starlette listens for `http.disconnect`; under 2.4 it
    notices on send instead, which is why send raises once the client is gone.
    """
    body = json.dumps(payload).encode()
    state = {"request_sent": False, "chunks": 0}
    hung_up = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if not state["request_sent"]:
            state["request_sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await hung_up.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            state["chunks"] += 1
            if state["chunks"] >= after_chunks:
                hung_up.set()
                if spec_version >= "2.4":
                    raise OSError("connection lost")

    with contextlib.suppress(ClientDisconnect):
        await app(_scope(spec_version), receive, send)


class _ClosableSpy:
    """A session that records when it is closed, so a released answer is observable."""

    def __init__(self, inner: Session, closed: list[float]) -> None:
        self._inner = inner
        self._closed = closed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def close(self) -> None:
        self._closed.append(time.monotonic())
        self._inner.close()


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_mid_stream_closes_the_session(
    db_session: Session, seed_items: Seed, spec_version: str
) -> None:
    # No gc.collect(): a server cannot rely on the collector. This is red on 4edd307 because the
    # generator sits suspended at a yield until a later (and possibly blocking) collection.
    seed_items(AGENT_ROWS)
    closed: list[float] = []
    app_deps = deps(db_session, search_then())
    app_deps.session_factory = lambda: _ClosableSpy(db_session, closed)  # type: ignore[assignment,return-value]
    asyncio.run(
        _hang_up_mid_stream(create_app(app_deps), {"question": "q"}, spec_version=spec_version)
    )
    assert closed, "the session was not released after the client hung up"


class _SlowModel(ScriptedChatModel):
    """A model that spends real time on every chunk, so a hang-up lands mid-answer."""

    emitted: ClassVar[list[str]] = []

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        for chunk in super()._stream(*args, **kwargs):
            self.emitted.append(str(chunk.message.content))
            time.sleep(0.01)
            yield chunk


def test_an_aborted_answer_ends_quietly(db_session: Session) -> None:
    # The consumer left, so there is nobody to send an error event to: `run_question` returns with
    # no events at all and still releases its session.
    closed: list[float] = []
    app_deps = deps(db_session, [ChatModel("openrouter", _AbortingModel(script=[]))])
    app_deps.session_factory = lambda: _ClosableSpy(db_session, closed)  # type: ignore[assignment,return-value]
    assert list(run_question(app_deps, "q", Filters())) == []
    assert closed


def test_disconnect_stops_the_in_flight_model_call_and_records_it(db_session: Session) -> None:
    # The client is gone, so the answer must stop producing tokens, and the request that reached
    # the provider must still be recorded: the daily cap counts what was spent.
    text = " ".join(["Coding agents are scored on real bugs [S1]."] * 12)
    slow = _SlowModel(
        script=[
            AIMessage(
                "", tool_calls=[{"name": "search_corpus", "args": {"query": "x"}, "id": "c1"}]
            ),
            AIMessage(text),
        ]
    )
    store = MemoryChatCalls()
    app_deps = deps(db_session, [ChatModel("openrouter", slow)], store)
    asyncio.run(
        _hang_up_mid_stream(
            create_app(app_deps), {"question": "q"}, spec_version="2.3", after_chunks=6
        )
    )
    # The tool event plus a handful of tokens, not the whole 90-chunk answer.
    assert len(slow.emitted) < 20, len(slow.emitted)
    assert store.models == [UNKNOWN_MODEL, "failed"]


def test_the_abort_writes_its_usage_row_before_the_session_is_released(
    db_session: Session,
) -> None:
    # Closing the stream joins the graph's worker thread, which records the aborted call. The row
    # must not be lost to a session release that happens first.
    order: list[str] = []

    class OrderedStore(MemoryChatCalls):
        def record_chat_call(self, **kwargs: Any) -> None:
            order.append("usage")
            super().record_chat_call(**kwargs)

    class OrderedSession:
        def __init__(self, inner: Session) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def close(self) -> None:
            order.append("session.close")
            self._inner.close()

    text = " ".join(["Coding agents are scored on real bugs [S1]."] * 12)
    slow = _SlowModel(
        script=[
            AIMessage(
                "", tool_calls=[{"name": "search_corpus", "args": {"query": "x"}, "id": "c1"}]
            ),
            AIMessage(text),
        ]
    )
    app_deps = deps(db_session, [ChatModel("openrouter", slow)], OrderedStore())
    app_deps.session_factory = lambda: OrderedSession(db_session)  # type: ignore[assignment,return-value]
    asyncio.run(
        _hang_up_mid_stream(
            create_app(app_deps), {"question": "q"}, spec_version="2.3", after_chunks=6
        )
    )
    # Every usage row is written before the one session close: a close first would lose it.
    assert order.count("session.close") == 1
    assert order.index("session.close") > max(i for i, step in enumerate(order) if step == "usage")


def test_the_answer_is_closed_on_a_worker_thread_not_the_event_loop() -> None:
    # Closing joins the graph's worker thread, so running it on the event loop would freeze the
    # server for the length of a model call.
    threads: list[int] = []

    def events() -> Any:
        try:
            yield _sse_line()
        finally:
            threads.append(threading.get_ident())

    async def run() -> int:
        loop_thread = threading.get_ident()
        response = api._ClosingStream(events(), threading.Event(), media_type="text/event-stream")

        async def send(message: dict[str, Any]) -> None:
            # Let the response start and one chunk through, then the client is gone.
            if message["type"] == "http.response.body" and message.get("body"):
                raise OSError("connection lost")

        with contextlib.suppress(Exception):
            await response.stream_response(send)
        return loop_thread

    loop_thread = asyncio.run(run())
    assert threads and threads[0] != loop_thread


def _sse_line() -> str:
    return api._sse("token", {"text": "a"})


class _AbortingModel(ScriptedChatModel):
    """A model whose request is aborted, as the callback aborts a real one mid-stream."""

    def _next(self, messages: list[Any], kwargs: dict[str, Any]) -> AIMessage:
        raise AnswerAborted


class _CountingModel(_SlowModel):
    """A slow scripted model that counts the provider requests it was asked for.

    `ScriptedChatModel.seen` already holds one entry per request; `requests` names that count so
    a test reads as what it means.
    """

    @property
    def requests(self) -> int:
        return len(self.seen)


def test_no_new_request_starts_after_the_client_left_during_a_tool(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The client leaves while a tool runs. The tool itself cannot be interrupted, but the round
    # after it must not be: one more request would be spent out of the free daily cap for an
    # answer nobody is reading. The flag is what `listen_for_disconnect` sets; this drives the
    # answer directly so the tool is provably running when it is set.
    from paperwatch.agent import tools as tools_mod

    started = threading.Event()

    def search(*args: Any, **kwargs: Any) -> list[Any]:
        started.set()
        return []

    monkeypatch.setattr(tools_mod, "hybrid_search", search)
    tool_call = AIMessage(
        "", tool_calls=[{"name": "search_corpus", "args": {"query": "x"}, "id": "c1"}]
    )
    model = _CountingModel(script=[tool_call, AIMessage(ANSWER)])
    store = MemoryChatCalls()
    cancel = threading.Event()
    stream = run_question(
        deps(db_session, [ChatModel("openrouter", model)], store), "q", Filters(), cancel
    )
    received: list[Any] = []

    def consume() -> None:
        for event in stream:
            received.append(event)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert started.wait(timeout=10), "the tool never ran"
    cancel.set()
    consumer.join(timeout=10)
    assert not consumer.is_alive()
    # Exactly the one request that made the tool call: the round after the tool was refused
    # before it could reach the provider, so nothing was spent and nothing was recorded.
    assert model.requests == 1
    assert store.models == [UNKNOWN_MODEL]


def test_disconnect_during_a_tool_call_round_starts_no_further_request(
    db_session: Session,
) -> None:
    # The model is mid-round when the client leaves: it has streamed words and is about to call a
    # tool. The round may finish, but the round after it must not start, so no second request is
    # made and nothing more is spent.
    tool_call = AIMessage(
        "let me search",
        tool_calls=[{"name": "search_corpus", "args": {"query": "x"}, "id": "c1"}],
    )
    model = _CountingModel(script=[tool_call, AIMessage(ANSWER)])
    app_deps = deps(db_session, [ChatModel("openrouter", model)])
    asyncio.run(
        _hang_up_mid_stream(
            create_app(app_deps), {"question": "q"}, spec_version="2.3", after_chunks=2
        )
    )
    assert model.requests == 1


def test_a_no_support_answer_sends_no_flags_and_logs_them(db_session: Session) -> None:
    # The model cites an alias no tool returned and has no other evidence, so the checker returns
    # the fixed no-support message. Its flag describes words the reader never sees: it goes to the
    # log, and the answer event carries none.
    models = scripted(AIMessage("Coding agents are scored on real bugs [S9]."))
    with capture_logs() as logs:
        response = ask(deps(db_session, models), {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["text"]) == ("answer", NO_SUPPORT)
    assert data["flags"] == []
    logged = [entry for entry in logs if entry["event"] == "answer_unsupported"]
    assert logged and logged[0]["flags"][0]["kind"] == "unknown_source"
    assert logged[0]["flags"][0]["sentence"] == "Coding agents are scored on real bugs"
    assert "S9" in logged[0]["flags"][0]["detail"]


def test_tool_args_with_non_finite_numbers_become_null(db_session: Session) -> None:
    # json.dumps writes NaN for a float nan, which a browser's JSON.parse rejects. A model can
    # emit it in a tool call's arguments, so the event payload coerces it to null.
    tool_call = AIMessage(
        "",
        tool_calls=[
            {
                "name": "search_corpus",
                "args": {"query": "SWE-bench", "tier": float("nan")},
                "id": "c1",
            }
        ],
    )
    response = ask(deps(db_session, scripted(tool_call, AIMessage(ANSWER))), {"question": "q"})
    assert "NaN" not in response.text
    tool = events(response.text)[0][1]
    assert tool["name"] == "search_corpus" and tool["args"]["tier"] is None


def test_json_safe_cleans_non_finite_numbers_at_any_depth() -> None:
    # A model can nest a runaway number anywhere in its tool arguments, and a client's JSON.parse
    # rejects NaN or Infinity wherever it appears.
    assert api._json_safe(
        {"a": [1.0, float("inf"), {"b": float("nan")}], "c": 2, "d": (float("-inf"),)}
    ) == {"a": [1.0, None, {"b": None}], "c": 2, "d": [None]}


def test_a_profile_item_is_stripped_and_bounded() -> None:
    # A name with surrounding spaces is the same profile; a blank or over-long one is a bug.
    assert api.AskRequest(question="q", profiles=[" ai-forensics "]).profiles == ["ai-forensics"]
    for bad in ("", "   ", "p" * (api.PROFILE_MAX_CHARS + 1)):
        with pytest.raises(ValidationError):
            api.AskRequest(question="q", profiles=[bad])


def test_sse_escapes_every_non_ascii_character() -> None:
    # ensure_ascii=True (json.dumps' default) is what keeps a line separator or a lone surrogate
    # from breaking the frame or the encode; the client decodes the \uXXXX escapes itself.
    line = api._sse("token", {"text": "a\u202eb\ud800"})
    assert "\\u202e" in line and "\\ud800" in line
    assert "\u202e" not in line
    assert line.startswith("event: token\ndata: ") and line.endswith("\n\n")


def test_live_deps_wires_a_usage_log_on_its_own_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Production has one wiring path: the answer engine and the usage engine are separate, so a
    # usage write never queues behind an answer that is holding its connection for a model call.
    monkeypatch.setattr(api, "make_embedder", lambda settings, ledger: FakeEmbedder())
    monkeypatch.setattr(api, "make_chat_models", lambda settings: [])
    settings = Settings(  # type: ignore[arg-type]
        openrouter_api_key="sk-test", api_auth="hmac", api_hmac_secret=TEST_SECRET
    )
    app_deps = api.live_deps(settings)
    assert isinstance(app_deps.chat_calls, ChatCallLog)
    answer_engine = app_deps.session_factory.kw["bind"]
    usage_engine = app_deps.chat_calls._session_factory.kw["bind"]  # type: ignore[attr-defined]
    assert usage_engine is not answer_engine
    assert answer_engine.pool.size() == ANSWER_POOL_SIZE
    assert answer_engine.pool._max_overflow == ANSWER_POOL_OVERFLOW
    assert answer_engine.pool._timeout == ANSWER_POOL_TIMEOUT_S
    assert usage_engine.pool.size() == USAGE_POOL_SIZE < ANSWER_POOL_SIZE
    assert usage_engine.pool._max_overflow == 0
    assert usage_engine.pool._timeout == ANSWER_POOL_TIMEOUT_S
    # The profile names the API validates against come from the repository's file, and readiness
    # checks the database through this small engine rather than the answer pool.
    assert app_deps.profile_names == {profile.name for profile in load_profiles(api.PROFILES_PATH)}
    assert app_deps.ready_session_factory is not None
    assert app_deps.ready_session_factory.kw["bind"] is usage_engine
    # The gate's host allow-list starts from the loopback names; PW_API_PUBLIC_HOST adds the
    # tunnel's own name when the server has one.
    assert app_deps.allowed_hosts == LOOPBACK_HOSTS
    answer_engine.dispose()
    usage_engine.dispose()


def test_a_full_pool_fails_fast(engine: Engine) -> None:
    # The answer pool is deliberately small and its wait short: a full pool must surface as an
    # error event soon, not after the driver's 30 s default.
    small = make_engine(
        engine.url.render_as_string(hide_password=False),
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
    )
    from sqlalchemy.exc import TimeoutError as PoolTimeout
    from sqlalchemy.orm import sessionmaker

    session = sessionmaker(small)()
    try:
        session.execute(text("SELECT 1"))
        started = time.monotonic()
        with pytest.raises(PoolTimeout):
            sessionmaker(small)().execute(text("SELECT 1"))
        assert time.monotonic() - started < 2
    finally:
        session.close()
        small.dispose()


def test_make_engine_bounds_a_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    # A paused or unreachable server must not hang a probe or a request on connect. The argument
    # goes to libpq, which answers after this many seconds instead of waiting on the socket.
    from paperwatch.store import db

    captured: dict[str, Any] = {}

    def fake_create_engine(url: str, **kwargs: Any) -> Any:
        captured.update(url=url, kwargs=kwargs)
        return object()

    monkeypatch.setattr(db, "create_engine", fake_create_engine)
    db.make_engine("postgresql+psycopg://x", pool_size=2)
    assert captured["kwargs"]["connect_args"] == {"connect_timeout": CONNECT_TIMEOUT_S}
    assert captured["kwargs"]["pool_size"] == 2


def test_the_openapi_contract_is_committed(db_session: Session) -> None:
    """Rewrite after an intended API change with `python -m paperwatch.api.contract`."""
    from paperwatch.api.contract import build_contract

    committed = json.loads(Path("docs/api/openapi.json").read_text())
    assert build_contract() == committed


def test_the_openapi_contract_never_claims_verification(db_session: Session) -> None:
    # supported means "every visible claim points at a retrieved source", not "verified": numbers
    # and sentences under six words are not checked (lesson 11). No document may say otherwise.
    document = json.dumps(create_app(deps(db_session, [])).openapi()).lower()
    assert "verified" not in document and "fact-check" not in document
    assert "every visible claim points at a retrieved source" in json.dumps(
        create_app(deps(db_session, [])).openapi()
    )


def test_the_openapi_contract_types_the_event_payloads(db_session: Session) -> None:
    # M3 generates its client from the contract, so the stream's payloads are schemas, not prose.
    schemas = create_app(deps(db_session, [])).openapi()["components"]["schemas"]
    for name in (
        "ToolData",
        "StepData",
        "SourceData",
        "TokenData",
        "NoticeData",
        "AnswerData",
        "CitationData",
        "FlagData",
        "ErrorData",
    ):
        assert name in schemas, name


def test_the_contract_describes_one_parsed_event_not_the_body(db_session: Session) -> None:
    # The media type's schema is read as the shape of one message, which for SSE is the `event`
    # name plus the parsed JSON of the `data` line. An envelope with `event` and `data` keys in
    # the same object does not exist on the wire (Task 17 review round 2, R5).
    spec = create_app(deps(db_session, [])).openapi()
    media = spec["paths"]["/ask"]["post"]["responses"]["200"]["content"]["text/event-stream"]
    assert "schema" not in media
    item = media["itemSchema"]
    assert set(item["required"]) == {"event", "data"}
    assert item["properties"]["event"]["enum"] == [
        "tool",
        "step",
        "source",
        "token",
        "notice",
        "answer",
        "error",
    ]
    data = item["properties"]["data"]
    assert data["contentMediaType"] == "application/json"
    assert {ref["$ref"] for ref in data["contentSchema"]["anyOf"]} == {
        "#/components/schemas/ToolData",
        "#/components/schemas/StepData",
        "#/components/schemas/SourceData",
        "#/components/schemas/TokenData",
        "#/components/schemas/NoticeData",
        "#/components/schemas/AnswerData",
        "#/components/schemas/ErrorData",
    }
    # The registration model is not part of the contract.
    assert "StreamItem" not in spec["components"]["schemas"]


def _dummy_deps() -> AgentDeps:
    # The health routes never touch the database or the models, so they need no real dependency.
    return AgentDeps(
        session_factory=lambda: None,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=[],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=MemoryChatCalls(),
        profile_names=PROFILES,
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=FixedMode(ModeDecision("hybrid", "ok")),
    )


def test_healthz_is_async_and_readyz_is_async() -> None:
    # A sync route runs on the shared thread pool; /healthz must not queue behind a stalled
    # /readyz. Both are async, and only /readyz hands work to a thread.
    import inspect

    app = create_app(_dummy_deps())
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    assert inspect.iscoroutinefunction(routes["/healthz"])
    assert inspect.iscoroutinefunction(routes["/readyz"])


def test_run_question_marks_its_session_read_only(db_session: Session, seed_items: Seed) -> None:
    # The guarantee lives in run_question, not in the caller: the session it opens cannot write.
    seed_items(AGENT_ROWS)
    statements: list[str] = []

    class SpySession:
        def __init__(self, inner: Session) -> None:
            self._inner = inner

        def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            statements.append(str(statement))
            return self._inner.execute(statement, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    app_deps = deps(db_session, search_then())
    app_deps.session_factory = lambda: SpySession(db_session)  # type: ignore[assignment,return-value]
    received = list(run_question(app_deps, "q", Filters()))
    assert any("SET TRANSACTION READ ONLY" in statement for statement in statements)
    # And the answer still worked: the tools ran inside the read-only transaction.
    assert isinstance(received[-1], AnswerEvent) and received[-1].answer.supported


def test_all_three_tools_work_in_a_read_only_transaction(
    db_session: Session, engine: Engine, seed_items: Seed
) -> None:
    # Task 14 N5: the tools only read, so the answer's session cannot write even through a bug.
    seed_items(AGENT_ROWS)
    db_session.execute(text("SET TRANSACTION READ ONLY"))
    registry = SourceRegistry()
    tools = {tool.name: tool for tool in make_tools(db_session, FakeEmbedder(), registry)}
    # All three tools still work under the read-only transaction, and none of them ends it with
    # a rollback or a commit (which would silently make the session writable again).
    calls: list[str] = []

    class SpySession:
        def __init__(self, inner: Session) -> None:
            self._inner = inner

        def rollback(self) -> None:
            calls.append("rollback")
            self._inner.rollback()

        def commit(self) -> None:
            calls.append("commit")
            self._inner.commit()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    tools = {
        tool.name: tool for tool in make_tools(SpySession(db_session), FakeEmbedder(), registry)
    }
    tools["search_corpus"].invoke({"query": "SWE-bench coding agents"})
    tools["get_item"].invoke({"source_id": "S1"})
    tools["compare_items"].invoke({"source_ids": ["S1", "S2"]})
    assert calls == []

    # Usage rows belong to ChatCallLog's own sessions, on their own connections, so the read-only
    # transaction cannot block them: that is what keeps the daily cap honest.
    at = datetime(2026, 10, 16, 5, tzinfo=UTC)
    factory = make_session_factory(engine)
    try:
        ChatCallLog(factory).record_chat_call(
            at=at, provider="openrouter", model="m", input_tokens=1, output_tokens=2
        )
    finally:
        with factory() as cleanup:
            cleanup.execute(delete(ChatCall).where(ChatCall.at == at))
            cleanup.commit()

    with pytest.raises(SQLAlchemyError):
        db_session.execute(text("CREATE TABLE probe_should_fail (a int)"))


def test_the_seventh_question_in_a_minute_is_refused_before_any_model_call(
    db_session: Session,
) -> None:
    # The limit is spent before the graph starts, so a refused question costs no model call.
    model = ScriptedChatModel(script=[AIMessage(ANSWER) for _ in range(6)])
    app_deps = deps(db_session, [ChatModel("openrouter", model)])
    for index in range(6):
        assert (
            client(create_app(app_deps)).post("/ask", json={"question": f"q{index}"}).status_code
            == 200
        )
    refused = client(create_app(app_deps)).post("/ask", json={"question": "q7"})
    assert refused.status_code == 429
    assert refused.json() == {"detail": "rate_limited", "retry_after_s": 60}
    assert refused.headers["retry-after"] == "60"
    assert len(model.seen) == 6


def test_a_third_concurrent_answer_is_busy(db_session: Session) -> None:
    # Two answers already hold the slots; the third is refused at once rather than queued behind
    # two slow free models.
    app_deps = deps(db_session, [])
    first = app_deps.limiter.admit()
    second = app_deps.limiter.admit()
    assert isinstance(first, Admitted) and isinstance(second, Admitted)
    refused = client(create_app(app_deps)).post("/ask", json={"question": "q"})
    assert refused.status_code == 429
    assert refused.json()["detail"] == "busy"
    first.release()
    second.release()


def test_a_released_answer_frees_its_slot(db_session: Session) -> None:
    # The slot belongs to the stream, so a finished answer releases it without an operator.
    app_deps = deps(db_session, scripted(AIMessage(ANSWER)))
    for _ in range(3):
        assert client(create_app(app_deps)).post("/ask", json={"question": "q"}).status_code == 200
    assert app_deps.limiter.admit().__class__ is Admitted


def test_daily_cap_streams_matching_sources_then_the_error(
    db_session: Session, seed_items: Seed
) -> None:
    # With the cap spent, the sheet keeps its search results: one keyword search runs, its sources
    # stream, and only then the error with the reset time.
    seed_items(AGENT_ROWS)
    # A cap of 1 with one call already recorded is the same state as a spent cap; the settings
    # refuse 0 because a server with no model calls would be a misconfiguration, not a limit.
    store = MemoryChatCalls()
    store.rows.append((NOW, UNKNOWN_MODEL))
    response = ask(
        deps(db_session, scripted(AIMessage(ANSWER)), store=store, chat_max_calls_per_day=1),
        {"question": "coding agents"},
    )
    received = events(response.text)
    assert [name for name, _ in received][-1] == "error"
    assert all(name == "source" for name, _ in received[:-1]) and received[:-1]
    assert received[-1][1]["code"] == "daily_cap"
    assert received[-1][1]["retry_at"] == "2026-10-17T00:00:00Z"


def test_no_token_or_question_text_in_info_logs(db_session: Session, seed_items: Seed) -> None:
    # A token and the question are the two things a log must never carry at info: the token is a
    # credential, and the question is the captain's private text.
    seed_items(AGENT_ROWS)
    phrase = "distinctive phrase from the captain's private question"
    app_deps = deps(db_session, search_then())
    token = mint_hmac_token(TEST_SECRET, "test", time.time())
    with capture_logs() as logs:
        authed = TestClient(create_app(app_deps), base_url="http://localhost")
        authed.post(
            "/ask",
            json={"question": phrase},
            headers={"Authorization": f"Bearer {token}"},
        )
        # A 401 and a 429 are logged with a reason code only.
        TestClient(create_app(app_deps), base_url="http://localhost").post(
            "/ask", json={"question": phrase}
        )
        for _ in range(6):
            authed.post(
                "/ask",
                json={"question": phrase},
                headers={
                    "Authorization": f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}"
                },
            )
    text = str(logs)
    assert token not in text
    assert phrase not in text


def test_every_streamed_event_has_a_payload_model(db_session: Session, seed_items: Seed) -> None:
    # Every event the stream can emit is one of these models, and the contract's itemSchema lists
    # the same set. A new event without a payload model would fail here.
    from paperwatch.api.app import STREAM_ITEM_SCHEMA, NoticeData, SourceData, StepData

    seed_items(AGENT_ROWS)
    response = ask(deps(db_session, search_then()), {"question": "q"})
    models: dict[str, Any] = {
        "tool": ToolData,
        "step": StepData,
        "source": SourceData,
        "token": TokenData,
        "notice": NoticeData,
        "answer": AnswerData,
        "error": ErrorData,
    }
    for name, data in events(response.text):
        assert isinstance(models[name].model_validate(data), models[name])
    referenced = {
        ref["$ref"].rsplit("/", 1)[-1]
        for ref in STREAM_ITEM_SCHEMA["properties"]["data"]["contentSchema"]["anyOf"]
    }
    assert referenced == {model.__name__ for model in models.values()}


def test_status_reports_the_mode(db_session: Session) -> None:
    # The page reads this on load, so it needs the mode, the reason and when an ingestion started.
    since = NOW - timedelta(minutes=20)
    decision = ModeDecision("text_only", "ingestion", since)
    response = client(create_app(deps(db_session, [], search_mode=FixedMode(decision)))).get(
        "/status"
    )
    assert response.status_code == 200
    assert response.json() == {
        "mode": "text_only",
        "reason": "ingestion",
        "since": since.isoformat().replace("+00:00", "Z"),
    }


def test_text_only_answers_send_a_notice_and_label_the_answer(db_session: Session) -> None:
    # The page shows the notice before the first token, and the answer itself says which mode
    # produced it, so a reduced answer is never mistaken for a hybrid one.
    decision = ModeDecision("text_only", "ingestion", NOW - timedelta(minutes=5))
    response = ask(
        deps(db_session, scripted(AIMessage(ANSWER)), search_mode=FixedMode(decision)),
        {"question": "q"},
    )
    received = events(response.text)
    assert received[0][0] == "notice"
    assert received[0][1]["code"] == "text_only"
    answer = next(data for name, data in received if name == "answer")
    assert answer["search_mode"] == "text_only"


class _OrderModel(ScriptedChatModel):
    """A scripted model that adds its calls to a shared order log."""

    # `Any`, not `list[str]`: pydantic copies a validated list, and the test reads its own object.
    order: Any

    def _next(self, messages: Any, kwargs: Any) -> AIMessage:
        self.order.append("model")
        return super()._next(messages, kwargs)


class _RecordingWorker(WorkerEmbedder):
    """A WorkerEmbedder that records the prewarm and the answer window without spawning a child."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order

    def prewarm(self) -> None:
        self.order.append("prewarm")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.order.append("embed")
        return FakeEmbedder().embed(texts)

    @contextlib.contextmanager
    def answer_open(self) -> Any:
        self.order.append("open")
        yield
        self.order.append("closed")


def test_hybrid_answers_prewarm_before_the_first_model_call(db_session: Session) -> None:
    # Starting the child while the model thinks hides the model load; prewarming after the first
    # token would not.
    order: list[str] = []
    worker = _RecordingWorker(order)
    model = _OrderModel(script=[AIMessage(ANSWER)], order=order)
    ask(deps(db_session, [ChatModel("openrouter", model)], embedder=worker), {"question": "q"})
    assert order[0] == "prewarm"
    assert order.index("prewarm") < order.index("model")
    assert "open" in order and "closed" in order


def test_a_running_child_is_stopped_when_the_mode_turns_keyword_only() -> None:
    # While the ingestion needs the memory the child exits at the next reap, so the API gives the
    # memory back instead of waiting for the ten idle minutes.
    from paperwatch.api.search_mode import SearchModeGate

    class _NoRun:
        def __enter__(self) -> "_NoRun":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def scalar(self, statement: Any) -> None:
            return None

    state = {"mode": "hybrid"}
    gate = SearchModeGate(
        lambda: _NoRun(),  # type: ignore[arg-type,return-value]
        clock=lambda: NOW,
        read_meminfo=lambda: None,
        read_psi=lambda: None,
    )
    worker = WorkerEmbedder(
        [sys.executable, str(FAKE_WORKER)], should_stop=lambda: state["mode"] == "text_only"
    )
    try:
        worker.embed(["hello"])
        assert worker.alive()
        state["mode"] = "text_only"
        worker.reap_once()
        assert not worker.alive()
    finally:
        worker.stop()
    # The gate itself is exactly what `should_stop` asks; this asserts the two agree.
    assert gate.current().mode == "hybrid"


FAKE_WORKER = Path(__file__).resolve().parents[1] / "helpers" / "fake_embed_worker.py"
