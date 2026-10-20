"""The streaming API: the SSE contract, the error events, and the session it releases.

Every test drives the real graph and the real tools over the real test database; only the chat
model is scripted, so the API's behaviour is pinned without a key, a network call or a live model.
"""

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import Engine, delete, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from structlog.testing import capture_logs

from paperwatch.agent.citations import NO_SUPPORT
from paperwatch.agent.graph import AnswerEvent
from paperwatch.agent.sources import SourceRegistry
from paperwatch.agent.tools import make_tools
from paperwatch.api.app import AgentDeps, create_app, run_question
from paperwatch.config import Settings
from paperwatch.llm.chat import ChatModel, ScriptedChatModel
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval.hybrid import Filters
from paperwatch.store.db import make_session_factory
from paperwatch.store.models import ChatCall
from paperwatch.store.repository import ChatCallLog
from tests.conftest import AGENT_ROWS, SeedRow

pytestmark = pytest.mark.integration
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
ANSWER = "SWE-bench Verified measures coding agents on real repository bugs [S1]."
Seed = Callable[[list[SeedRow]], dict[str, UUID]]


class MemoryChatCalls:
    """The `ChatCallStore` the API tests need: a count in memory, never a database row.

    The tests below share one database session, and `run_question` marks it read-only, so a real
    `ChatCallLog` (which commits on its own session) has no place here. The production store is
    covered by `tests/integration/test_chat_calls.py`.
    """

    def __init__(self) -> None:
        self.calls: list[datetime] = []

    def record_chat_call(
        self,
        *,
        at: datetime,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.calls.append(at)

    def chat_calls_since(self, at: datetime) -> int:
        return sum(1 for call_at in self.calls if call_at >= at)


def scripted(*steps: AIMessage | Exception) -> list[ChatModel]:
    return [ChatModel("openrouter", ScriptedChatModel(script=list(steps)))]


def search_then(text: str = ANSWER, **args: Any) -> list[ChatModel]:
    tool_args = {"query": "SWE-bench coding agents", **args}
    return scripted(
        AIMessage("", tool_calls=[{"name": "search_corpus", "args": tool_args, "id": "c1"}]),
        AIMessage(text),
    )


def deps(db_session: Session, models: list[ChatModel], **settings: object) -> AgentDeps:
    # Every request reuses the test session so seeded rows are visible. run_question closes it;
    # with the fixture's savepoint join, the outer test transaction still rolls back.
    return AgentDeps(  # type: ignore[arg-type]
        lambda: db_session,
        FakeEmbedder(),
        models,
        Settings(**settings),  # type: ignore[arg-type]
        lambda: NOW,
        MemoryChatCalls(),
    )


def events(body: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ask(app_deps: AgentDeps, payload: dict[str, Any]) -> Any:
    return TestClient(create_app(app_deps)).post("/ask", json=payload)


def test_healthz(db_session: Session) -> None:
    assert TestClient(create_app(deps(db_session, []))).get("/healthz").json() == {"status": "ok"}


def test_readyz_checks_the_database(db_session: Session) -> None:
    assert TestClient(create_app(deps(db_session, []))).get("/readyz").status_code == 200
    broken = deps(db_session, [])
    broken.session_factory = lambda: (_ for _ in ()).throw(RuntimeError("db down"))  # type: ignore[assignment,return-value]
    response = TestClient(create_app(broken)).get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


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


def test_blank_question_is_refused(db_session: Session) -> None:
    assert ask(deps(db_session, []), {"question": "   "}).status_code == 422


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


def test_model_failure_streams_an_error_event(db_session: Session) -> None:
    response = ask(deps(db_session, scripted(RuntimeError("503"))), {"question": "q"})
    assert response.status_code == 200  # the stream had started, so the error travels inside it
    name, data = events(response.text)[-1]
    assert (name, data["code"]) == ("error", "model_unavailable") and data["message"]


def test_provider_text_stays_out_of_the_stream_and_the_logs(db_session: Session) -> None:
    # An SDK error's body can echo the flagged question, and ModelUnavailable keeps it as its
    # cause on purpose. run_question must not log the cause or put it in the event.
    models = scripted(RuntimeError("provider body: the captain's secret question"))
    with capture_logs() as logs:
        response = ask(deps(db_session, models), {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["code"]) == ("error", "model_unavailable")
    assert "provider body" not in data["message"]
    assert "provider body" not in str(logs)


def test_an_internal_failure_logs_only_the_error_type(db_session: Session) -> None:
    # The internal branch serves an unforeseen failure. Its traceback (and any chained cause)
    # could hold provider or question text, so only the exception's type is logged.
    app_deps = deps(db_session, [])
    app_deps.session_factory = lambda: (_ for _ in ()).throw(  # type: ignore[assignment,return-value]
        RuntimeError("provider body: the captain's secret question")
    )
    with capture_logs() as logs:
        response = ask(app_deps, {"question": "q"})
    name, data = events(response.text)[-1]
    assert (name, data["code"]) == ("error", "internal")
    failures = [entry for entry in logs if entry["event"] == "ask_failed"]
    assert [entry["error_type"] for entry in failures] == ["RuntimeError"]
    assert all("exc_info" not in entry for entry in logs)
    assert "secret question" not in str(logs)


def test_daily_cap_streams_an_error_event(db_session: Session) -> None:
    response = ask(deps(db_session, search_then(), chat_max_calls_per_day=1), {"question": "q"})
    # The first call fits the cap of 1; the second is refused.
    assert events(response.text)[-1][0] == "error"
    assert events(response.text)[-1][1]["code"] == "daily_cap"


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
    stream: Iterator[Any] = run_question(app_deps, "q", Filters())
    next(stream)
    stream.close()  # what Starlette does when the client disconnects
    assert closed == [True]


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


def test_the_openapi_contract_is_committed(db_session: Session) -> None:
    """Rewrite after an intended API change with the one-liner in docs/api/README.md."""
    committed = json.loads(Path("docs/api/openapi.json").read_text())
    assert create_app(deps(db_session, [])).openapi() == committed


def test_the_openapi_contract_never_claims_verification(db_session: Session) -> None:
    # supported means "every visible claim points at a retrieved source", not "verified": numbers
    # and sentences under six words are not checked (lesson 11). No document may say otherwise.
    document = json.dumps(create_app(deps(db_session, [])).openapi()).lower()
    assert "verified" not in document and "fact-check" not in document
    assert "every visible claim points at a retrieved source" in json.dumps(
        create_app(deps(db_session, [])).openapi()
    )


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
    # All three tools still work under the read-only transaction.
    tools["search_corpus"].invoke({"query": "SWE-bench coding agents"})
    tools["get_item"].invoke({"source_id": "S1"})
    tools["compare_items"].invoke({"source_ids": ["S1", "S2"]})

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
