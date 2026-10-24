"""The streaming HTTP face of the agent: health, readiness and one server-sent-events answer.

`run_question` is the one place where an agent failure becomes an `ErrorEvent` and where the
database session is released, so the CLI and the HTTP API cannot drift apart. The route only
formats events; every rule about answers (the citation check, the alias registry, the call cap)
lives below it in `agent` and `llm`.

The answer arrives as server-sent events (`tool`, `step`, `source`, `token`, `answer`, then
optionally `error`), paced with a `: ping` comment every 15 s of silence and a 120 s deadline.
Only the `answer` event is authoritative: the model's streamed words can differ from the checked
text (an unknown alias is masked, or the whole answer is replaced with `NO_SUPPORT`), so a client
replaces its streamed text with `answer.text` instead of appending to it.
"""

import json
import math
import threading
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import anyio
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.orm import Session, sessionmaker
from starlette.types import Receive, Send

import paperwatch
from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer
from paperwatch.agent.graph import (
    AgentEvent,
    AnswerAborted,
    AnswerEvent,
    SourceEvent,
    StepEvent,
    TokenEvent,
    ToolEvent,
    build_agent,
    stream_answer,
)
from paperwatch.agent.sources import SourceRef, SourceRegistry
from paperwatch.agent.tools import make_tools
from paperwatch.api.auth import Verifier, make_verifier
from paperwatch.api.gate import ApiGate
from paperwatch.api.limits import Admitted, Refused, RequestLimiter
from paperwatch.api.pacing import paced
from paperwatch.config import (
    ANSWER_DEADLINE_S,
    ANSWER_POOL_OVERFLOW,
    ANSWER_POOL_SIZE,
    ANSWER_POOL_TIMEOUT_S,
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    PROFILES_PATH,
    READY_CONCURRENCY,
    READY_TIMEOUT_S,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    SSE_HEARTBEAT_S,
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
from paperwatch.retrieval.hybrid import Filters, hybrid_search
from paperwatch.store.db import make_engine, make_session_factory
from paperwatch.store.repository import ChatCallLog, Repository

log = structlog.get_logger()

# The API contract's own version, not the package version: release-please bumps the package on
# every release, and the committed docs/api/openapi.json must change only when the API does.
API_VERSION = "0.2"

# The fixed text of the `internal` event: the client gets something it can show, and the detail
# (which could carry a provider body or stored text) stays in the server log.
INTERNAL_MESSAGE = "the answer failed; see the server log"
# A saturated connection pool is not the same failure as a bug: the client is told to come back,
# and the operator gets the code site in the log.
BUSY_MESSAGE = "the server is busy; try again shortly"
# The deadline's message is fixed for the same reason as the internal one: the client gets a
# sentence it can show, and the operator reads the reason code.
DEADLINE_MESSAGE = "the answer took too long; ask again"
# The SSE comment line, not a named event: it carries no data, and a client's watchdog resets on
# any received bytes.
HEARTBEAT_FRAME = ": ping\n\n"

# The package's own directory, so `_failure_site` can tell the application's frames from a
# library's by location. A checkout in a directory called `paperwatch` used to match every frame.
_APP_ROOT = Path(paperwatch.__file__).parent

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
    # The credential verifier and the host names the gate accepts. `create_app` always installs the
    # gate with them, so no code path can serve a route without one.
    verifier: Verifier
    allowed_hosts: frozenset[str]
    # `/readyz` queries the database through this factory. In production it is the usage engine,
    # not the answer pool: with every answer connection held by a live answer, a healthy database
    # would otherwise answer 503, which an orchestrator reads as "restart me".
    # The per-client limit in front of every model-spending request (Task 27).
    limiter: RequestLimiter
    ready_session_factory: Callable[[], Session] | None = None


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
    profiles = load_profiles(PROFILES_PATH)
    usage_factory = make_session_factory(usage_engine)
    # The tunnel's own host name is added to the loopback ones; unset means loopback only, and the
    # tunnel name arrives through the same 0600 env file as the credential.
    public_host = (settings.api_public_host or "").strip().lower()
    allowed_hosts = LOOPBACK_HOSTS | ({public_host} if public_host else set())
    return AgentDeps(
        session_factory=factory,
        # Local embeddings need no token budget of their own; the ledger is here because
        # `make_embedder` takes one, and it charges nothing on the local path.
        embedder=make_embedder(settings, Ledger(settings.max_tokens_per_run)),
        models=models,
        settings=settings,
        clock=lambda: datetime.now(UTC),
        chat_calls=ChatCallLog(usage_factory),
        profile_names=frozenset(p.name for p in profiles),
        verifier=make_verifier(settings),
        allowed_hosts=frozenset(allowed_hosts),
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: datetime.now(UTC),
        ),
        ready_session_factory=usage_factory,
    )


@dataclass(frozen=True)
class ErrorEvent:
    """A failure the stream reports instead of ending in a traceback."""

    code: Literal["daily_cap", "budget", "model_unavailable", "internal", "deadline"]
    message: str
    # Set for `daily_cap`: the UTC midnight when the free tier's call count resets.
    retry_at: datetime | None = None


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


class StepData(BaseModel):
    """The `step` event: one tool round finished."""

    round: int
    ms: int
    new_sources: int


class SourceData(BaseModel):
    """The `source` event: a source a tool just returned, in alias order."""

    id: str = Field(description="The per-answer source alias, such as S1.")
    item_id: str
    title: str
    url: str
    tier: str
    basis: str


class NoticeData(BaseModel):
    """The `notice` event: the answer runs in a reduced mode and says so."""

    code: Literal["text_only"]
    message: str


class FlagData(BaseModel):
    kind: str
    sentence: str
    detail: str


class AnswerData(BaseModel):
    text: str
    search_mode: Literal["hybrid", "text_only"] = Field(
        default="hybrid", description="Hybrid retrieval, or the keyword-only branch during a run."
    )
    supported: bool = Field(
        description=(
            "True when every visible claim points at a retrieved source. That is not a "
            "verification: sentences under six words and numbers are not checked."
        )
    )
    citations: list[CitationData]
    flags: list[FlagData]


class ErrorData(BaseModel):
    code: Literal["daily_cap", "budget", "model_unavailable", "internal", "deadline"]
    message: str
    retry_at: datetime | None = None


class StreamItem(BaseModel):
    """Registration of the payload models, not the shape of the wire.

    A real event is one `event: <name>` line and one `data: <json payload>` line, so its payload
    has no `event` key and no `data` wrapper. FastAPI needs a response model to register the
    payload models in the contract, and the app's `openapi` post-processing replaces this envelope
    with the honest `itemSchema`. A client replaces its streamed text with `answer.data.text`.

    The heartbeat is an SSE comment line (`: ping`), not a named event, so it is not here.
    """

    event: Literal["tool", "step", "source", "token", "notice", "answer", "error"]
    data: ToolData | StepData | SourceData | TokenData | NoticeData | AnswerData | ErrorData


# The daily cap's fallback search: enough to fill a reading sheet, and one query.
CAP_FALLBACK_K = 8

# The schema of one parsed server-sent event, in the OpenAPI 3.1 form FastAPI's native SSE
# response emits: `event` names the frame and `data` is a JSON string whose own schema is the
# payload. A client reads the media type as "one message", so the response body's shape would be
# the wrong answer.
STREAM_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event": {
            "type": "string",
            "enum": ["tool", "step", "source", "token", "notice", "answer", "error"],
        },
        "data": {
            "type": "string",
            "contentMediaType": "application/json",
            "contentSchema": {
                "anyOf": [
                    {"$ref": "#/components/schemas/ToolData"},
                    {"$ref": "#/components/schemas/StepData"},
                    {"$ref": "#/components/schemas/SourceData"},
                    {"$ref": "#/components/schemas/TokenData"},
                    {"$ref": "#/components/schemas/NoticeData"},
                    {"$ref": "#/components/schemas/AnswerData"},
                    {"$ref": "#/components/schemas/ErrorData"},
                ]
            },
        },
    },
    "required": ["event", "data"],
}


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
    token, and the round after a running tool is never sent. The API's response sets it the moment
    its disconnect listener fires; a caller that can be interrupted from outside (the CLI's
    Ctrl-C) passes its own event.
    """
    session: Session | None = None
    stream: Generator[AgentEvent, None, None] | None = None
    cancel = cancel if cancel is not None else threading.Event()
    # Created before the graph so the cap fallback below can see what the answer had already found.
    registry = SourceRegistry()
    try:
        session = deps.session_factory()
        # The tools only read, so the database enforces it: a bug, or a tool added later, cannot
        # write to the corpus during an answer. SET TRANSACTION has to be the transaction's first
        # statement, which is why nothing runs before it on this fresh session.
        session.execute(text("SET TRANSACTION READ ONLY"))
        guard = CallGuard(
            deps.chat_calls,
            Ledger(deps.settings.chat_max_tokens_per_answer),
            deps.settings.chat_max_calls_per_day,
            deps.clock,
        )
        tools = make_tools(session, deps.embedder, registry, base_filters=filters)
        graph = build_agent(deps.models, tools, registry, guard, cancel)
        stream = stream_answer(graph, question, cancel, registry)
        # Not `yield from`: closing a generator suspended in a yield-from closes the sub-generator
        # before this frame can set the flag, and the flag has to be set first.
        for event in stream:  # noqa: UP028 - a yield-from would close the stream before the flag
            yield event
    except AnswerAborted:
        # The consumer left (see `AnswerAborted`): there is nobody to report to. The `finally`
        # below still releases the session, and `ChatCallLog` has already committed any usage row.
        return
    except DailyCapReached as exc:
        # The sheet stays useful: one search still runs, so the reader sees candidate sources
        # beside the message about the reset time (section 15's call).
        yield from _cap_fallback(deps, session, question, filters, registry)
        yield ErrorEvent("daily_cap", str(exc), retry_at=_next_midnight(deps.clock()))
    except BudgetExceeded as exc:
        yield ErrorEvent("budget", str(exc))
    except ModelUnavailable as exc:
        # `str(exc)` names providers and exception classes only. The cause holds the provider's
        # response text (Task 16 Step 8), which is never logged here nor sent to a client.
        yield ErrorEvent("model_unavailable", str(exc))
    except PoolTimeout:
        # The answer pool is small and its wait short, so a full pool arrives here; it is a busy
        # server, not a bug, but the client gets a fixed sentence either way.
        log.warning("answer_pool_exhausted")
        yield ErrorEvent("internal", BUSY_MESSAGE)
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
        try:
            if stream is not None:
                # Joins the graph's worker thread, so the aborted call records its usage row
                # before the read-only session below is released.
                stream.close()
        finally:
            # The session is read-only, so there is nothing to commit: the usage rows were
            # already committed by ChatCallLog on its own session. rollback() also clears a
            # failed statement. In `finally` so a failing close cannot leak the connection.
            if session is not None:
                session.rollback()
                session.close()


def _next_midnight(now: datetime) -> datetime:
    """The UTC midnight when the free tier's daily call count starts again."""
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=1)


def _cap_fallback(
    deps: AgentDeps,
    session: Session | None,
    question: str,
    filters: Filters,
    registry: SourceRegistry,
) -> Generator[SourceEvent, None, None]:
    """One plain search when the daily cap refuses the answer before any source was found.

    A failure here is logged by type and skipped: the cap error is what the client needs, and a
    search that cannot run must not replace it with a traceback.
    """
    if session is None or registry.all():
        return
    try:
        hits = hybrid_search(session, deps.embedder, question, k=CAP_FALLBACK_K, filters=filters)
        details = Repository(session).admitted_details([hit.item_id for hit in hits])
        refs = [registry.register(details[hit.item_id]) for hit in hits if hit.item_id in details]
    except Exception as exc:  # noqa: BLE001 - the cap error must still reach the client
        log.warning("cap_fallback_search_failed", error_type=type(exc).__name__)
        return
    for ref in refs:
        yield SourceEvent(ref)


def _failure_site(exc: BaseException, limit: int = 4) -> list[str]:
    """Where a failure happened, as `file:line:function`, and nothing more.

    Never a message, a variable value or a chained cause: an exception's text can hold a provider
    body or the question. The application's own frames come first; the deepest frame is appended
    because it names the library call that actually failed.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    inside = [f for f in frames if Path(f.filename).is_relative_to(_APP_ROOT)]
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
            "payload: `tool`, `step`, `source`, `token`, `notice`, `answer` or `error`. A `: ping` "
            "comment line arrives after 15 s of silence; it has no `data:`. `answer` is the last "
            "event unless an `error` follows; a client replaces its streamed text with the "
            "`answer` event's `text`, never appends it, and takes every link from its "
            "`citations[].url`."
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


def _refused(message: str, loc: list[str]) -> HTTPException:
    """A route-level 422 in the shape the contract documents for a validation error.

    FastAPI's own validators answer with an array of error objects, so a plain string `detail`
    from `HTTPException` would break a typed client that parses the documented shape.
    """
    return HTTPException(
        status_code=422,
        detail=[{"type": "value_error", "loc": ["body", *loc], "msg": message}],
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


def _source_payload(ref: SourceRef) -> SourceData:
    # The `source` event carries the same facts as a citation, so a client can render a card
    # before the answer is checked and the alias it will cite later is already known.
    return SourceData(
        id=ref.alias,
        item_id=str(ref.item_id),
        title=ref.title,
        url=ref.url,
        tier=ref.tier,
        basis=ref.basis,
    )


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


def _stream(
    deps: AgentDeps, body: AskRequest, cancel: threading.Event
) -> Generator[str, None, None]:
    # The pacer owns the deadline and the heartbeat, and closing this generator closes both: it
    # sets the cancel event and joins the producer, on the worker thread `_ClosingStream` uses.
    yield from paced(
        _answer_frames(deps, body, cancel),
        cancel=cancel,
        clock=time.monotonic,
        heartbeat_s=SSE_HEARTBEAT_S,
        deadline_s=ANSWER_DEADLINE_S,
        heartbeat=lambda: HEARTBEAT_FRAME,
        deadline_error=lambda: _sse(
            "error", ErrorData(code="deadline", message=DEADLINE_MESSAGE).model_dump(mode="json")
        ),
    )


def _answer_frames(
    deps: AgentDeps, body: AskRequest, cancel: threading.Event
) -> Generator[str, None, None]:
    # closing() releases the database session as soon as this generator is closed (a client
    # disconnect) instead of waiting for garbage collection. The cancel event is the response's,
    # so the flag is set the moment the client leaves, not when this generator is closed.
    with closing(run_question(deps, body.question, _filters(body), cancel)) as events:
        for event in events:
            if isinstance(event, ToolEvent):
                yield _sse(
                    "tool", ToolData(name=event.name, args=_json_safe(event.args)).model_dump()
                )
            elif isinstance(event, SourceEvent):
                yield _sse("source", _source_payload(event.ref).model_dump())
            elif isinstance(event, StepEvent):
                yield _sse(
                    "step",
                    StepData(
                        round=event.round, ms=event.ms, new_sources=event.new_sources
                    ).model_dump(),
                )
            elif isinstance(event, TokenEvent):
                yield _sse("token", TokenData(text=event.text).model_dump())
            elif isinstance(event, AnswerEvent):
                yield _sse("answer", _answer_payload(event.answer).model_dump())
            else:
                yield _sse(
                    "error",
                    ErrorData(
                        code=event.code, message=event.message, retry_at=event.retry_at
                    ).model_dump(mode="json"),
                )


class _ClosingStream(StreamingResponse):
    """A streaming response that closes its event generator however the response ends.

    Starlette wraps a sync generator for a thread pool and never closes it, so a client
    disconnect leaves it suspended at a yield; a later garbage-collection close runs on whatever
    thread triggers the pass, and closing the graph joins its worker thread, which freezes the
    event loop or deadlocks that thread. Closing here is deterministic, and on a worker thread
    with cancellation shielded, so it never races the `next()` still in flight.

    The same response owns the answer's cancel event. Setting it here is what reaches a model
    call still in flight, or the round after a tool: closing the generator alone only happens
    once the current step has returned.
    """

    def __init__(
        self,
        events: Generator[str, None, None],
        cancel: threading.Event,
        release: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(events, **kwargs)
        self._events = events
        self._cancel = cancel
        # The request limiter's slot belongs to this response, not to the route: it must be freed
        # when the stream ends however it ends, including a client hang-up.
        self._release = release

    async def listen_for_disconnect(self, receive: Receive) -> None:
        # Starlette's listener is the only thing that learns about a hang-up while the response
        # task waits for the next chunk; the generator is closed later, when that wait ends.
        await super().listen_for_disconnect(receive)
        self._cancel.set()

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self._events.close)
                if self._release is not None:
                    self._release()


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
        # The served app publishes no docs and no schema: the credential gate would otherwise
        # decide who may read them, and the committed docs/api/openapi.json is the contract a
        # client builds from. `create_app(...).openapi()` still builds the document in process.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Every request, including the routes FastAPI adds itself, meets the gate first. Adding it here
    # rather than in `serve` means a test, a script or a future caller cannot build an open app.
    app.add_middleware(ApiGate, verifier=deps.verifier, allowed_hosts=deps.allowed_hosts)
    # The readiness gate lives on the app so a test can replace it, and so two apps never share
    # one limiter (a limiter belongs to one event loop's concurrency budget).
    app.state.ready_limiter = anyio.CapacityLimiter(READY_CONCURRENCY)

    # FastAPI types the SSE media type with the `StreamItem` envelope, which exists only to
    # register the payload models. Post-processing swaps in the shape a client parses and drops
    # the envelope, so the contract says what is on the wire (Task 17 review round 2, R5).
    generate_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        schema = generate_openapi()
        media = schema["paths"]["/ask"]["post"]["responses"]["200"]["content"]["text/event-stream"]
        media.pop("schema", None)
        media["itemSchema"] = STREAM_ITEM_SCHEMA
        schema["components"]["schemas"].pop("StreamItem", None)
        return schema

    app.openapi = openapi  # type: ignore[method-assign]

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
                # wait for it. The limiter bounds how many checks are *started*; what bounds the
                # abandoned ones is the connect timeout and the connection pool they queue on,
                # which is why readiness uses the small usage engine rather than the answer pool.
                await anyio.to_thread.run_sync(
                    _select_one,
                    deps.ready_session_factory or deps.session_factory,
                    abandon_on_cancel=True,
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
        """Answer one question as server-sent events, one `itemSchema` event per frame."""
        # Refused here rather than in `AskRequest`: both would only buy an empty search after a
        # model call was paid for, and the profile names are the process's, not the model's.
        unknown = sorted(set(body.profiles or ()) - deps.profile_names)
        if unknown:
            raise _refused(f"unknown profiles: {', '.join(unknown)}", ["profiles"])
        if body.since is not None and body.since > deps.clock().astimezone(UTC).date():
            raise _refused("since must not be in the future", ["since"])
        admitted = deps.limiter.admit()
        if isinstance(admitted, Refused):
            # Refused before the stream starts, so the client gets a status and a Retry-After it
            # can honour instead of a 200 that ends in an error event.
            return JSONResponse(
                status_code=429,
                content={"detail": admitted.reason, "retry_after_s": admitted.retry_after_s},
                headers={"Retry-After": str(admitted.retry_after_s)},
            )
        assert isinstance(admitted, Admitted)  # narrows the union for mypy
        cancel = threading.Event()
        return _ClosingStream(
            _stream(deps, body, cancel),
            cancel,
            release=admitted.release,
            media_type="text/event-stream",
            # A cached or proxy-buffered answer would arrive all at once, so the stream is
            # uncacheable and unbuffered end to end.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
