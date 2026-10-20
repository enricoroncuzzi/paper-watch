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
import math
import threading
import traceback
from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import anyio
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker
from starlette.types import Send

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
from paperwatch.config import (
    ANSWER_POOL_OVERFLOW,
    ANSWER_POOL_SIZE,
    ANSWER_POOL_TIMEOUT_S,
    READY_CONCURRENCY,
    READY_TIMEOUT_S,
    USAGE_POOL_SIZE,
    Settings,
)
from paperwatch.domain import Tier, load_profiles
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

QUESTION_MAX_CHARS = 1000
# The corpus has a handful of profiles; a longer list is a client bug, and any unknown name would
# only buy an empty search after a model call was already paid for.
PROFILES_MAX = 8
PROFILE_MAX_CHARS = 64


@dataclass
class AgentDeps:
    """Everything one answer needs, built once per process (or per test)."""

    session_factory: sessionmaker[Session]
    embedder: Embedder
    models: list[ChatModel]
    settings: Settings
    clock: Callable[[], datetime]
    # The usage store is separate from the tools' session on purpose: the tools read through a
    # read-only transaction, while a usage row needs its own committed transaction (ADR 0006). It
    # also gets its own small engine, so a usage write never queues behind an answer that holds a
    # connection while it waits for the model.
    chat_calls: ChatCallStore
    # The profile names the corpus knows, so a typo is refused before a model call is spent.
    profile_names: frozenset[str]


def live_deps(settings: Settings) -> AgentDeps:
    """Build the production dependencies: the provider chain, then the two engines."""
    # The chat chain is built first: without a key it refuses here, before the local embedding
    # model is loaded or its ~180 MB file is downloaded.
    models = make_chat_models(settings)
    engine = make_engine(
        settings.database_url,
        pool_size=ANSWER_POOL_SIZE,
        max_overflow=ANSWER_POOL_OVERFLOW,
        pool_timeout=ANSWER_POOL_TIMEOUT_S,
    )
    factory = make_session_factory(engine)
    # A stuck usage write must not hold the answer's close for the driver's 30 s default, so the
    # usage pool gets the same short wait.
    usage_engine = make_engine(
        settings.database_url,
        pool_size=USAGE_POOL_SIZE,
        max_overflow=0,
        pool_timeout=ANSWER_POOL_TIMEOUT_S,
    )
    profiles = load_profiles(Path("config/profiles.yaml"))
    return AgentDeps(
        session_factory=factory,
        # Local embeddings need no token budget of their own; the ledger is here because
        # `make_embedder` takes one, and it charges nothing on the local path.
        embedder=make_embedder(settings, Ledger(settings.max_tokens_per_run)),
        models=models,
        settings=settings,
        clock=lambda: datetime.now(UTC),
        chat_calls=ChatCallLog(make_session_factory(usage_engine)),
        profile_names=frozenset(p.name for p in profiles),
    )


@dataclass(frozen=True)
class ErrorEvent:
    """A failure the stream reports instead of ending in a traceback."""

    code: Literal["daily_cap", "budget", "model_unavailable", "internal"]
    message: str


class ToolData(BaseModel):
    """The `tool` event: the model called one of the corpus tools."""

    name: str
    # Model-written JSON arguments. Values are rendered, so they are JSON-safe (no NaN) and a
    # client treats every string as text.
    args: dict[str, Any]


class TokenData(BaseModel):
    """The `token` event: a piece of the model's answer, before the citation check."""

    text: str


class CitationData(BaseModel):
    id: str = Field(description="The per-answer source alias, such as S1.")
    item_id: str
    title: str
    url: str
    tier: str
    basis: str


class FlagData(BaseModel):
    kind: str
    sentence: str
    detail: str


class AnswerData(BaseModel):
    text: str
    supported: bool = Field(
        description=(
            "True when every visible claim points at a retrieved source. That is not a "
            "verification: sentences under six words and numbers are not checked."
        )
    )
    citations: list[CitationData]
    flags: list[FlagData]


class ErrorData(BaseModel):
    code: Literal["daily_cap", "budget", "model_unavailable", "internal"]
    message: str


class StreamItem(BaseModel):
    """One server-sent event: the `event` name and the JSON payload of its `data` line.

    On the wire each event is one `event: <event>` line and one `data: <json>` line, with no
    blank line inside the payload; a client replaces its streamed text with `answer.data.text`.
    """

    event: Literal["tool", "token", "answer", "error"]
    data: ToolData | TokenData | AnswerData | ErrorData


def run_question(
    deps: AgentDeps,
    question: str,
    filters: Filters,
    cancel: threading.Event | None = None,
) -> Generator[AgentEvent | ErrorEvent, None, None]:
    """Drive one answer, yielding its events and closing its database session whatever happens.

    A new `SourceRegistry`, ledger and tool set is built per call: aliases and spending belong to
    one answer. This is a generator, so a client that never starts reading spends nothing.

    `cancel` is set when the consumer leaves: a model call still in flight then stops at its next
    token instead of writing a whole answer for nobody. A caller that can be interrupted from
    outside (the CLI's Ctrl-C) passes its own event, because the interrupt unwinds the stream
    before this function's `finally` could set it.
    """
    session: Session | None = None
    stream: Generator[AgentEvent, None, None] | None = None
    cancel = cancel if cancel is not None else threading.Event()
    try:
        session = deps.session_factory()
        # The tools only read, so the database enforces it: a bug, or a tool added later, cannot
        # write to the corpus during an answer. SET TRANSACTION has to be the transaction's first
        # statement, which is why nothing runs before it on this fresh session.
        session.execute(text("SET TRANSACTION READ ONLY"))
        registry = SourceRegistry()
        guard = CallGuard(
            deps.chat_calls,
            Ledger(deps.settings.chat_max_tokens_per_answer),
            deps.settings.chat_max_calls_per_day,
            deps.clock,
        )
        tools = make_tools(session, deps.embedder, registry, base_filters=filters)
        stream = stream_answer(build_agent(deps.models, tools, registry, guard), question, cancel)
        # Not `yield from`: closing a generator suspended in a yield-from closes the sub-generator
        # before this frame can set the flag, and the flag has to be set first.
        for event in stream:  # noqa: UP028 - a yield-from would close the stream before the flag
            yield event
    except DailyCapReached as exc:
        yield ErrorEvent("daily_cap", str(exc))
    except BudgetExceeded as exc:
        yield ErrorEvent("budget", str(exc))
    except ModelUnavailable as exc:
        # `str(exc)` names providers and exception classes only. The cause holds the provider's
        # response text (Task 16 Step 8), which is never logged here nor sent to a client.
        yield ErrorEvent("model_unavailable", str(exc))
    except Exception as exc:  # noqa: BLE001 - the stream has started, so the error travels inside it
        log.error(
            "ask_failed",
            error_type=type(exc).__name__,
            cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
            site=_failure_site(exc),
        )
        yield ErrorEvent("internal", INTERNAL_MESSAGE)
    finally:
        # Runs on a normal end, on an error, on generator close (a client disconnect, a
        # GeneratorExit) and on Ctrl-C. Setting the flag is a no-op when nothing is in flight.
        cancel.set()
        if stream is not None:
            # Joins the graph's worker thread, so the aborted call records its usage row before
            # the read-only session below is released.
            stream.close()
        # The session is read-only, so there is nothing to commit: the usage rows were already
        # committed by ChatCallLog on its own session. rollback() also clears a failed statement.
        if session is not None:
            session.rollback()
            session.close()


def _failure_site(exc: BaseException, limit: int = 4) -> list[str]:
    """Where a failure happened, as `file:line:function`, and nothing more.

    Never a message, a variable value or a chained cause: an exception's text can hold a provider
    body or the question. The application's own frames come first; the deepest frame is appended
    because it names the library call that actually failed.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    inside = [f for f in frames if "paperwatch" in Path(f.filename).parts]
    chosen = inside[-limit:] if inside else frames[-limit:]
    if frames and frames[-1] not in chosen:
        chosen = [*chosen, frames[-1]]
    return [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in chosen]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    tiers: list[Tier] | None = Field(default=None, max_length=3)
    profiles: list[str] | None = Field(default=None, max_length=PROFILES_MAX)
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

    @field_validator("profiles")
    @classmethod
    def _clean_profiles(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = []
        for item in value:
            stripped = item.strip()
            if not stripped or len(stripped) > PROFILE_MAX_CHARS:
                raise ValueError(f"each profile must be 1 to {PROFILE_MAX_CHARS} characters")
            cleaned.append(stripped)
        return cleaned


class _EventStream(JSONResponse):
    """OpenAPI marker for the SSE route: media type plus the typed response model.

    The route returns its own `_ClosingStream`, so this class is never instantiated. Being a
    `JSONResponse` subclass is what makes FastAPI register the response model (and the payload
    models inside it) in `components.schemas` while the media type says `text/event-stream`.
    """

    media_type = "text/event-stream"


_ASK_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "Server-sent events, one JSON object per `data:` line. The `event` line names the "
            "payload: `tool`, `token`, `answer` or `error`. `answer` is the last event unless an "
            "`error` follows; a client replaces its streamed text with the `answer` event's "
            "`text`, never appends it, and takes every link from its `citations[].url`."
        ),
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


def _json_safe(value: Any) -> Any:
    """Replace a non-finite float with null, everywhere in model-written tool arguments.

    JSON has no NaN or Infinity, and a browser's `JSON.parse` rejects them, so one runaway number
    in a tool call must not corrupt the stream.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _answer_payload(answer: CheckedAnswer) -> AnswerData:
    flags = answer.flags
    if answer.text == NO_SUPPORT:
        # The fixed message replaces the model's text, so the flags describe words the reader
        # never saw. They go to the log for the operator and never into the event.
        log.info(
            "answer_unsupported",
            flags=[{"kind": f.kind, "sentence": f.sentence, "detail": f.detail} for f in flags],
        )
        flags = ()
    return AnswerData(
        text=answer.text,
        supported=answer.supported,
        citations=[
            CitationData(
                id=c.alias,
                item_id=str(c.item_id),
                title=c.title,
                url=c.url,
                tier=c.tier,
                basis=c.basis,
            )
            for c in answer.citations
        ],
        flags=[FlagData(kind=f.kind, sentence=f.sentence, detail=f.detail) for f in flags],
    )


def _sse(name: str, payload: dict[str, Any]) -> str:
    # json.dumps escapes control characters, and its default ensure_ascii=True turns every
    # non-ASCII character into \uXXXX, so a line separator or even a lone surrogate cannot break
    # the frame or fail the later encode. allow_nan=False: NaN is not JSON, and a client's parser
    # would reject it.
    return f"event: {name}\ndata: {json.dumps(payload, allow_nan=False)}\n\n"


def _stream(deps: AgentDeps, body: AskRequest) -> Generator[str, None, None]:
    # closing() releases the database session as soon as this generator is closed (a client
    # disconnect) instead of waiting for garbage collection.
    with closing(run_question(deps, body.question, _filters(body))) as events:
        for event in events:
            if isinstance(event, ToolEvent):
                yield _sse(
                    "tool", ToolData(name=event.name, args=_json_safe(event.args)).model_dump()
                )
            elif isinstance(event, TokenEvent):
                yield _sse("token", TokenData(text=event.text).model_dump())
            elif isinstance(event, AnswerEvent):
                yield _sse("answer", _answer_payload(event.answer).model_dump())
            else:
                yield _sse("error", ErrorData(code=event.code, message=event.message).model_dump())


class _ClosingStream(StreamingResponse):
    """A streaming response that closes its event generator however the response ends.

    Starlette wraps a sync generator for a thread pool and never closes it, so a client
    disconnect leaves it suspended at a yield; a later garbage-collection close runs on whatever
    thread triggers the pass, and closing the graph joins its worker thread, which freezes the
    event loop or deadlocks that thread. Closing here is deterministic, and on a worker thread
    with cancellation shielded, so it never races the `next()` still in flight.
    """

    def __init__(self, events: Generator[str, None, None], **kwargs: Any) -> None:
        super().__init__(events, **kwargs)
        self._events = events

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self._events.close)


def _select_one(session_factory: Callable[[], Session]) -> None:
    with session_factory() as session:
        session.execute(text("SELECT 1"))


def _not_ready() -> JSONResponse:
    return JSONResponse(status_code=503, content={"status": "unavailable"})


def create_app(deps: AgentDeps) -> FastAPI:
    app = FastAPI(
        title="Paper Watch API",
        version=API_VERSION,
        summary="Ask questions about the stored AI-research corpus and stream cited answers.",
    )
    # The readiness gate lives on the app so a test can replace it, and so two apps never share
    # one limiter (a limiter belongs to one event loop's concurrency budget).
    app.state.ready_limiter = anyio.CapacityLimiter(READY_CONCURRENCY)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. It deliberately does not touch the database."""
        # async, so a liveness probe never queues behind a sync route waiting for a thread.
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        """Readiness: an answer needs the database, so readiness means one query succeeds."""
        limiter = request.app.state.ready_limiter
        try:
            limiter.acquire_nowait()
        except anyio.WouldBlock:
            # Checks are already stuck; answering 503 now is better than queueing another.
            return _not_ready()
        try:
            with anyio.fail_after(READY_TIMEOUT_S):
                # abandon_on_cancel: a check stuck on a paused database must not make the probe
                # wait for it; the limiter above bounds how many such checks can pile up.
                await anyio.to_thread.run_sync(
                    _select_one, deps.session_factory, abandon_on_cancel=True
                )
        except TimeoutError:
            log.warning("readiness_timeout", timeout_s=READY_TIMEOUT_S)
            return _not_ready()
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
            # Only the type: a connection error's text can hold a driver or host detail that an
            # anonymous probe should not be told.
            log.warning("readiness_failed", error_type=type(exc).__name__)
            return _not_ready()
        finally:
            limiter.release()
        return JSONResponse(content={"status": "ready"})

    @app.post(
        "/ask",
        response_class=_EventStream,
        response_model=StreamItem,
        responses=_ASK_RESPONSES,
    )
    async def ask(body: AskRequest) -> Response:
        """Answer one question as server-sent events, one `StreamItem` per event."""
        # Refused here rather than in the model: both would only buy an empty search after a
        # model call was paid for.
        unknown = sorted(set(body.profiles or ()) - deps.profile_names)
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown profiles: {', '.join(unknown)}")
        if body.since is not None and body.since > deps.clock().astimezone(UTC).date():
            raise HTTPException(status_code=422, detail="since must not be in the future")
        return _ClosingStream(
            _stream(deps, body),
            media_type="text/event-stream",
            # A cached or proxy-buffered answer would arrive all at once, so the stream is
            # uncacheable and unbuffered end to end.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
