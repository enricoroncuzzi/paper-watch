"""The streaming HTTP face of the agent: health, readiness and one server-sent-events answer.

`run_question` is the one place where an agent failure becomes an `ErrorEvent` and where the
database session is released, so the CLI and the HTTP API cannot drift apart. The route only
formats events; every rule about answers (the citation check, the alias registry, the call cap)
lives below it in `agent` and `llm`.

The answer arrives as server-sent events (`tool`, `token`, `answer`, then optionally `error`).
Only the `answer` event is authoritative: the model's streamed words can differ from the checked
text (an unknown alias is masked, or the whole answer is replaced with `NO_SUPPORT`), so a client
replaces its streamed text with `answer.text` instead of appending to it.
"""

import json
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer
from paperwatch.agent.graph import (
    AgentEvent,
    AnswerEvent,
    TokenEvent,
    ToolEvent,
    build_agent,
    stream_answer,
)
from paperwatch.agent.sources import SourceRegistry
from paperwatch.agent.tools import make_tools
from paperwatch.config import Settings
from paperwatch.domain import Tier
from paperwatch.llm.chat import (
    CallGuard,
    ChatCallStore,
    ChatModel,
    DailyCapReached,
    ModelUnavailable,
    make_chat_models,
)
from paperwatch.llm.embedder import Embedder, make_embedder
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.retrieval.hybrid import Filters
from paperwatch.store.db import make_engine, make_session_factory
from paperwatch.store.repository import ChatCallLog

log = structlog.get_logger()

# The API contract's own version, not the package version: release-please bumps the package on
# every release, and the committed docs/api/openapi.json must change only when the API does.
API_VERSION = "0.1"

# The fixed text of the `internal` event: the client gets something it can show, and the detail
# (which could carry a provider body or stored text) stays in the server log.
INTERNAL_MESSAGE = "the answer failed; see the server log"


@dataclass
class AgentDeps:
    """Everything one answer needs, built once per process (or per test)."""

    session_factory: sessionmaker[Session]
    embedder: Embedder
    models: list[ChatModel]
    settings: Settings
    clock: Callable[[], datetime]
    # The usage store is separate from the tools' session on purpose: the tools read through a
    # read-only transaction, while a usage row needs its own committed transaction (ADR 0006).
    # Production leaves this None and gets `ChatCallLog(session_factory)`; a test that shares one
    # session passes a store of its own instead.
    chat_calls: ChatCallStore | None = None


def live_deps(settings: Settings) -> AgentDeps:
    """Build the production dependencies: one engine, the local embedder, the provider chain."""
    engine = make_engine(settings.database_url)
    factory = make_session_factory(engine)
    return AgentDeps(
        session_factory=factory,
        # Local embeddings need no token budget of their own; the ledger is here because
        # `make_embedder` takes one, and it charges nothing on the local path.
        embedder=make_embedder(settings, Ledger(settings.max_tokens_per_run)),
        models=make_chat_models(settings),
        settings=settings,
        clock=lambda: datetime.now(UTC),
        chat_calls=ChatCallLog(factory),
    )


@dataclass(frozen=True)
class ErrorEvent:
    """A failure the stream reports instead of ending in a traceback."""

    code: Literal["daily_cap", "budget", "model_unavailable", "internal"]
    message: str


def run_question(
    deps: AgentDeps, question: str, filters: Filters
) -> Generator[AgentEvent | ErrorEvent, None, None]:
    """Drive one answer, yielding its events and closing its database session whatever happens.

    A new `SourceRegistry`, ledger and tool set is built per call: aliases and spending belong to
    one answer. This is a generator, so a client that never starts reading spends nothing.
    """
    session: Session | None = None
    try:
        session = deps.session_factory()
        # The tools only read, so the database enforces it: a bug, or a tool added later, cannot
        # write to the corpus during an answer. SET TRANSACTION has to be the transaction's first
        # statement, which is why nothing runs before it on this fresh session.
        session.execute(text("SET TRANSACTION READ ONLY"))
        registry = SourceRegistry()
        # ChatCallLog commits each usage row on its own session, so the cap holds even when this
        # read-only session is rolled back or the client hangs up.
        store = (
            deps.chat_calls if deps.chat_calls is not None else ChatCallLog(deps.session_factory)
        )
        guard = CallGuard(
            store,
            Ledger(deps.settings.chat_max_tokens_per_answer),
            deps.settings.chat_max_calls_per_day,
            deps.clock,
        )
        tools = make_tools(session, deps.embedder, registry, base_filters=filters)
        yield from stream_answer(build_agent(deps.models, tools, registry, guard), question)
    except DailyCapReached as exc:
        yield ErrorEvent("daily_cap", str(exc))
    except BudgetExceeded as exc:
        yield ErrorEvent("budget", str(exc))
    except ModelUnavailable as exc:
        # `str(exc)` names providers and exception classes only. The cause holds the provider's
        # response text (Task 16 Step 8), which is never logged here nor sent to a client.
        yield ErrorEvent("model_unavailable", str(exc))
    except Exception as exc:  # noqa: BLE001 - the stream has started, so the error travels inside it
        # Only the type: `log.exception` would render the traceback and any chained cause, and a
        # provider or SQL error's text can echo the question.
        log.error("ask_failed", error_type=type(exc).__name__)
        yield ErrorEvent("internal", INTERNAL_MESSAGE)
    finally:
        # Runs on a normal end, on an error, and on generator close (a client disconnect). The
        # session is read-only, so there is nothing to commit: the usage rows were already
        # committed by ChatCallLog on its own session. rollback() also clears a failed statement.
        if session is not None:
            session.rollback()
            session.close()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    tiers: list[Tier] | None = None
    profiles: list[str] | None = None
    since: date | None = None

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        # A question of spaces passes min_length but would search for nothing; the client can
        # catch that before a request is sent, and the API refuses it too.
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class _EventStream(StreamingResponse):
    """Marker class: the OpenAPI contract documents this route as `text/event-stream`."""

    media_type = "text/event-stream"


_ASK_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "Server-sent events, one JSON object per `data:` line. `tool` carries "
            '{"name", "args"}; `token` carries {"text"}; `answer` carries {"text", "supported", '
            '"citations", "flags"} and is the last event unless an `error` follows; `error` '
            'carries {"code", "message"}. `supported` is true when every visible claim points '
            "at a retrieved source. That is not a verification: numbers and sentences under six "
            "words are not checked. A client replaces its streamed text with `answer.text`, "
            "never appends it, and takes every link from `citations[].url`."
        ),
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


def _filters(body: AskRequest) -> Filters:
    return Filters(
        # A request that sends no tiers asks for no tier filter. Only a filter built by code can
        # mean "match nothing" (an empty frozenset), so an empty list is treated as absent.
        tiers=frozenset(t.value for t in body.tiers) if body.tiers else None,
        profiles=frozenset(body.profiles) if body.profiles else None,
        # A date means midnight UTC of that day, like the `search_corpus` tool's `since`.
        since=(
            datetime(body.since.year, body.since.month, body.since.day, tzinfo=UTC)
            if body.since is not None
            else None
        ),
    )


def _answer_payload(answer: CheckedAnswer) -> dict[str, Any]:
    flags = answer.flags
    if answer.text == NO_SUPPORT:
        # The fixed message replaces the model's text, so the flags describe words the reader
        # never saw. They go to the log for the operator and never into the event.
        log.info(
            "answer_unsupported",
            flags=[{"kind": f.kind, "sentence": f.sentence, "detail": f.detail} for f in flags],
        )
        flags = ()
    return {
        "text": answer.text,
        "supported": answer.supported,
        "citations": [
            {
                "id": c.alias,
                "item_id": str(c.item_id),
                "title": c.title,
                "url": c.url,
                "tier": c.tier,
                "basis": c.basis,
            }
            for c in answer.citations
        ],
        "flags": [{"kind": f.kind, "sentence": f.sentence, "detail": f.detail} for f in flags],
    }


def _sse(name: str, payload: dict[str, Any]) -> str:
    # json.dumps escapes control characters, so a model-written ESC or newline cannot break the
    # framing. Clients still render these values as text, never as HTML.
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _stream(deps: AgentDeps, body: AskRequest) -> Iterator[str]:
    # closing() releases the database session as soon as this generator is closed (a client
    # disconnect) instead of waiting for garbage collection.
    with closing(run_question(deps, body.question, _filters(body))) as events:
        for event in events:
            if isinstance(event, ToolEvent):
                yield _sse("tool", {"name": event.name, "args": event.args})
            elif isinstance(event, TokenEvent):
                yield _sse("token", {"text": event.text})
            elif isinstance(event, AnswerEvent):
                yield _sse("answer", _answer_payload(event.answer))
            else:
                yield _sse("error", {"code": event.code, "message": event.message})


def create_app(deps: AgentDeps) -> FastAPI:
    app = FastAPI(
        title="Paper Watch API",
        version=API_VERSION,
        summary="Ask questions about the stored AI-research corpus and stream cited answers.",
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness: the process is up. It deliberately does not touch the database."""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        """Readiness: an answer needs the database, so readiness means one query succeeds."""
        try:
            with deps.session_factory() as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
            # Only the type: a connection error's text can hold a driver or host detail that a
            # probe should not report to an anonymous caller.
            log.warning("readiness_failed", error_type=type(exc).__name__)
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return JSONResponse(content={"status": "ready"})

    @app.post("/ask", response_class=_EventStream, responses=_ASK_RESPONSES)
    def ask(body: AskRequest) -> StreamingResponse:
        """Answer one question as server-sent events. See the response description."""
        # A plain StreamingResponse, not the response_class: the generator is formatted here, and
        # the class only tells the OpenAPI document which media type the route returns. Starlette
        # runs a sync generator in its thread pool and closes it when the client disconnects.
        return StreamingResponse(
            _stream(deps, body),
            media_type="text/event-stream",
            # A cached or proxy-buffered answer would arrive all at once, so the stream is
            # uncacheable and unbuffered end to end.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
