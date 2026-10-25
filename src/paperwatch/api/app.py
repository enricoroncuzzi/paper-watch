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
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import anyio
import structlog
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.orm import Session, sessionmaker
from starlette.types import Receive, Scope, Send

import paperwatch
from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer, Flag
from paperwatch.agent.graph import (
    AgentEvent,
    AnswerAborted,
    AnswerEvent,
    SourceEvent,
    StepEvent,
    TokenEvent,
    ToolEvent,
    bound_history,
    build_agent,
    stream_answer,
)
from paperwatch.agent.sources import (
    SourceRef,
    SourceRegistry,
    ref_from_json,
    ref_to_json,
)
from paperwatch.agent.tools import SearchLog, item_block, make_tools
from paperwatch.api.auth import Principal, Verifier
from paperwatch.api.gate import ApiGate
from paperwatch.api.limits import Admitted, Refused, RequestLimiter
from paperwatch.api.pacing import paced
from paperwatch.api.payloads import AnswerData, CitationData, FlagData
from paperwatch.api.responses import (
    ASK_GATE_RESPONSES,
    BEARER,
    GATE_RESPONSES,
    Refusal,
)
from paperwatch.api.routes.drafts import DraftService, make_drafts_router
from paperwatch.api.routes.items import make_items_router
from paperwatch.api.routes.runs import make_runs_router
from paperwatch.api.routes.suggestions import (
    SuggestionService,
    make_suggestions_router,
)
from paperwatch.api.routes.threads import make_threads_router
from paperwatch.api.search_mode import SearchModeGate
from paperwatch.api.summary_queue import SummaryQueue
from paperwatch.config import (
    ANSWER_DEADLINE_S,
    ANSWER_POOL_OVERFLOW,
    ANSWER_POOL_SIZE,
    ANSWER_POOL_TIMEOUT_S,
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    PROFILES_PATH,
    READ_POOL_SIZE,
    READY_CONCURRENCY,
    READY_TIMEOUT_S,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    SSE_HEARTBEAT_S,
    SUGGESTIONS_PATH,
    THREAD_TITLE_MAX_CHARS,
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
from paperwatch.llm.embedder import Embedder, EmbedderUnavailable, WorkerEmbedder, make_embedder
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.retrieval.hybrid import Filters, hybrid_search
from paperwatch.store.db import make_engine, make_session_factory
from paperwatch.store.repository import ChatCallLog, Repository, ThreadDetailRow
from paperwatch.suggestions.config import load_suggestions
from paperwatch.suggestions.summary import SummaryWriter

log = structlog.get_logger()

# The API contract's own version, not the package version: release-please bumps the package on
# every release, and the committed docs/api/openapi.json must change only when the API does.
API_VERSION = "0.7"

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
# The notice a keyword-only answer starts with, so the page can explain the reduced search.
TEXT_ONLY_MESSAGE = (
    "Semantic search is paused while the corpus updates, so this answer uses keyword search."
)


@dataclass(frozen=True)
class ThreadEvent:
    """A finished turn was stored: the thread it joined and that turn's number.

    It is the last event of the stream and exists only on success, so a client follows up with an
    id it can re-fetch; an answer that failed stores nothing and sends no thread event.
    """

    thread_id: UUID
    turn: int


@dataclass(frozen=True)
class NoticeEvent:
    """A mode notice sent before the answer: currently only `text_only`."""

    code: Literal["text_only"]
    message: str


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
    # The per-client limit in front of every model-spending request.
    limiter: RequestLimiter
    # The search mode: a running main pass or memory pressure switches answers to keyword-only.
    search_mode: SearchModeGate
    # The API opens one short read-only session per tool call through this factory, so an answer
    # holds no connection while it waits for the model. Unset in tests and the CLI, where the
    # tools use the caller's already read-only session.
    read_session_factory: Callable[[], Session] | None = None
    # Where a finished turn is written. The answer's own session is read-only by design, so the
    # thread write needs its own: in production this is the answer pool, which an answer holds no
    # connection on. None means the caller does not keep threads (a test, the contract build).
    write_session_factory: Callable[[], Session] | None = None
    # `/readyz` queries the database through this factory. In production it is the usage engine,
    # not the answer pool: with every answer connection held by a live answer, a healthy database
    # would otherwise answer 503, which an orchestrator reads as "restart me".
    ready_session_factory: Callable[[], Session] | None = None
    # The suggestions page's ranking cache and summary queue. Unset builds the default one from
    # this deps object; a test hands in its own so the queue never starts a real thread.
    suggestions: SuggestionService | None = None
    # The draft chain, one OpenRouter model and no fallback. Unset builds it lazily from
    # the settings, so a test or the contract build needs no key until a draft is asked for.
    draft_models: list[ChatModel] | None = None
    # The drafts service. Unset builds the default one; a test hands in its own scripted models.
    drafts: DraftService | None = None


def _default_suggestions(deps: AgentDeps) -> SuggestionService:
    """The page's suggestions service: the config file, the profiles and a background queue."""
    session_factory = deps.read_session_factory or deps.session_factory
    profiles = {profile.name: profile for profile in load_profiles(PROFILES_PATH)}

    def writer() -> SummaryWriter:
        # A guard per call: a summary runs later, on the queue's thread, so its ledger is its own.
        guard = CallGuard(
            deps.chat_calls,
            Ledger(deps.settings.chat_max_tokens_per_answer),
            deps.settings.chat_max_calls_per_day,
            deps.clock,
        )
        return SummaryWriter(deps.models, lambda: guard)

    queue = SummaryQueue(
        writer,
        session_factory,
        deps.clock,
        # The configured model id is what `item_summaries` records, so a model change on the server
        # regenerates the cards instead of leaving a page written by two models.
        lambda: deps.settings.chat_model,
    )
    return SuggestionService(
        session_factory,
        profiles,
        load_suggestions(SUGGESTIONS_PATH),
        queue,
        deps.clock,
    )


def _default_drafts(deps: AgentDeps, suggestions: SuggestionService) -> DraftService:
    """The drafts service: the stored item, the draft chain and the same limiter as an answer."""
    session_factory = deps.read_session_factory or deps.session_factory

    def models() -> list[ChatModel]:
        # Lazy on purpose: building the document or an app without drafts needs no key, while a
        # live app fails here the first time a draft is asked for if its chain cannot be built.
        if deps.draft_models is not None:
            return deps.draft_models
        return make_chat_models(deps.settings, purpose="draft")

    return DraftService(
        session_factory=session_factory,
        models_factory=models,
        settings=deps.settings,
        chat_calls=deps.chat_calls,
        limiter=deps.limiter,
        suggestions=suggestions,
        config=load_suggestions(SUGGESTIONS_PATH),
        clock=deps.clock,
    )


def live_deps(settings: Settings, verifier: Verifier) -> AgentDeps:
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
    # The tools' own pool: a connection per tool call, so the answer pool's long-lived
    # connections are not the API's normal state.
    read_engine = make_engine(
        settings.database_url,
        pool_size=READ_POOL_SIZE,
        max_overflow=0,
        pool_timeout=ANSWER_POOL_TIMEOUT_S,
    )
    read_factory = make_session_factory(read_engine)
    # The gate and the child's stop check need each other, so `worker` is bound after both exist.
    worker: WorkerEmbedder | None = None

    def child_rss_mb() -> float | None:
        return worker.child_rss_mb() if worker is not None else None

    search_mode = SearchModeGate(
        read_factory, clock=lambda: datetime.now(UTC), read_child_rss=child_rss_mb
    )
    # The model lives in a child process and exits when idle or when the ingestion needs the memory
    # (ADR 0013). Azure embeddings are the kept Azure proof, not the deployment: they run in
    # process because there is no local model to release.
    if settings.embedding_provider == "azure":
        embedder: Embedder = make_embedder(settings, Ledger(settings.max_tokens_per_run))
    else:
        worker = WorkerEmbedder(should_stop=lambda: search_mode.current().mode == "text_only")
        embedder = worker
    profiles = load_profiles(PROFILES_PATH)
    usage_factory = make_session_factory(usage_engine)
    # The tunnel's own host name is added to the loopback ones; unset means loopback only, and the
    # tunnel name arrives through the same 0600 env file as the credential.
    public_host = (settings.api_public_host or "").strip().lower()
    allowed_hosts = LOOPBACK_HOSTS | ({public_host} if public_host else set())
    return AgentDeps(
        session_factory=factory,
        embedder=embedder,
        models=models,
        settings=settings,
        clock=lambda: datetime.now(UTC),
        chat_calls=ChatCallLog(usage_factory),
        profile_names=frozenset(p.name for p in profiles),
        verifier=verifier,
        allowed_hosts=frozenset(allowed_hosts),
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: datetime.now(UTC),
        ),
        search_mode=search_mode,
        read_session_factory=read_factory,
        ready_session_factory=usage_factory,
        # The answer pool is writable and an answer holds no connection on it, so the thread write
        # takes one short connection there rather than a fourth pool's worth of memory.
        write_session_factory=factory,
        # Built here with the answer chain: without an OpenRouter key the app refuses to start,
        # and the draft chain is one OpenRouter model (PW_DRAFT_MODEL or the chat model).
        draft_models=make_chat_models(settings, purpose="draft"),
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


class ThreadData(BaseModel):
    """The `thread` event: the finished turn was stored, so the client can follow up."""

    id: UUID
    turn: int


class ErrorData(BaseModel):
    code: Literal["daily_cap", "budget", "model_unavailable", "internal", "deadline"]
    message: str
    retry_at: datetime | None = None


class StatusData(BaseModel):
    """`GET /status`: the search mode the page reads on load."""

    mode: Literal["hybrid", "text_only"]
    reason: Literal["ok", "ingestion", "memory", "database"]
    since: datetime | None = None


class StreamItem(BaseModel):
    """Registration of the payload models, not the shape of the wire.

    A real event is one `event: <name>` line and one `data: <json payload>` line, so its payload
    has no `event` key and no `data` wrapper. FastAPI needs a response model to register the
    payload models in the contract, and the app's `openapi` post-processing replaces this envelope
    with the honest `itemSchema`. A client replaces its streamed text with `answer.data.text`.

    The heartbeat is an SSE comment line (`: ping`), not a named event, so it is not here.
    """

    event: Literal["tool", "step", "source", "token", "notice", "answer", "thread", "error"]
    data: (
        ToolData
        | StepData
        | SourceData
        | TokenData
        | NoticeData
        | AnswerData
        | ThreadData
        | ErrorData
    )


# The daily cap's fallback search: enough to fill a reading sheet, and one query.
CAP_FALLBACK_K = 8

_GATE_RESPONSES = GATE_RESPONSES
_ASK_GATE_RESPONSES = ASK_GATE_RESPONSES
_BEARER = BEARER

# The schema of one parsed server-sent event, in the OpenAPI 3.1 form FastAPI's native SSE
# response emits: `event` names the frame and `data` is a JSON string whose own schema is the
# payload. A client reads the media type as "one message", so the response body's shape would be
# the wrong answer.
STREAM_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event": {
            "type": "string",
            "enum": [
                "tool",
                "step",
                "source",
                "token",
                "notice",
                "answer",
                "thread",
                "error",
            ],
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
                    {"$ref": "#/components/schemas/ThreadData"},
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
    search_log: SearchLog | None = None,
    item_id: UUID | None = None,
    thread_id: UUID | None = None,
) -> Generator[AgentEvent | ErrorEvent | NoticeEvent | ThreadEvent, None, None]:
    """Drive one answer, yielding its events and closing its database session whatever happens.

    A new `SourceRegistry`, ledger and tool set is built per call: aliases and spending belong to
    one answer. This is a generator, so a client that never starts reading spends nothing.

    `thread_id` continues a conversation: its registry and its last finished turns are loaded
    before the graph starts, so a follow-up can cite an earlier alias and knows what was asked.
    A finished answer is stored as the thread's next turn. Only the reader's `question` is
    stored, never the prompt the item block turned it into.

    `cancel` is set when the consumer leaves: a model call still in flight then stops at its next
    token, and the round after a running tool is never sent. The API's response sets it the moment
    its disconnect listener fires; a caller that can be interrupted from outside (the CLI's
    Ctrl-C) passes its own event.

    `search_log` records whether a search ran by keyword only, so the answer can carry
    `search_mode`; the API passes the one its stream will read.
    """
    session: Session | None = None
    stream: Generator[AgentEvent, None, None] | None = None
    cancel = cancel if cancel is not None else threading.Event()
    search_log = search_log if search_log is not None else SearchLog()
    # Created before the graph so the cap fallback below can see what the answer had already found.
    registry = SourceRegistry()
    history: list[tuple[str, str]] = []
    # How many aliases the thread already held, and how many turns it held, when this answer
    # loaded it: both are what tell this answer's own work apart from the conversation's history.
    restored_sources = 0
    expected_turns = 0
    try:
        if thread_id is not None:
            # The route checks the thread first, so an unknown id is a 404 before the stream; this
            # is the read the answer itself needs. A thread deleted between the two is an error
            # event rather than a silent new conversation under a different id.
            saved = _load_thread(deps, thread_id)
            if saved is None:
                yield ErrorEvent("internal", INTERNAL_MESSAGE)
                return
            registry = SourceRegistry.from_refs([ref_from_json(ref) for ref in saved.registry])
            restored_sources = len(registry.refs())
            expected_turns = len(saved.turns)
            history = bound_history(
                [(turn.question, str(turn.answer.get("text", ""))) for turn in saved.turns]
            )
        decision = deps.search_mode.current()
        if decision.mode == "text_only":
            # The answer still runs; the page is told why its search is keyword-only.
            search_log.text_only = True
            yield NoticeEvent("text_only", TEXT_ONLY_MESSAGE)
        # With a read factory the tools open one short read-only session per call and the answer
        # holds no connection; without one (CLI, tests) it uses this session, made read-only once.
        if deps.read_session_factory is None:
            session = deps.session_factory()
            # The tools only read, so the database enforces it: a bug, or a tool added later,
            # cannot write to the corpus during an answer. SET TRANSACTION has to be the
            # transaction's first statement, which is why nothing runs before it here.
            session.execute(text("SET TRANSACTION READ ONLY"))
        guard = CallGuard(
            deps.chat_calls,
            Ledger(deps.settings.chat_max_tokens_per_answer),
            deps.settings.chat_max_calls_per_day,
            deps.clock,
        )
        # What the model reads: the reader's own question, or that question under the item's
        # block. `question` stays the reader's words, so a stored turn never becomes a prompt.
        prompt = question
        # Only a new thread registers the item as S1: a follow-up already has its aliases, and a
        # second S1 would point the reader at two different papers.
        if item_id is not None and thread_id is None:
            # Registered before the graph starts, so S1 is this item whatever the search returns:
            # "Ask about it" means the question is about that paper, and the citation check reads
            # the same registry the answer cites from.
            opener = deps.read_session_factory or deps.session_factory
            with closing(opener()) as reader:
                detail = Repository(reader).admitted_details([item_id]).get(item_id)
            if detail is not None:
                ref = registry.register(detail)
                # The registry alias alone proves the model may cite the item, not that it has
                # read it: the tools' own block goes into the prompt, so an answer about S1 rests
                # on stored text that is clipped and attributed like any retrieved source.
                prompt = (
                    f"The reader is asking about {ref.alias}. "
                    "Its stored record is in the block below.\n"
                    f"{item_block(ref, detail)}\n{question}"
                )
        tools = make_tools(
            session,
            deps.embedder,
            registry,
            base_filters=filters,
            mode=lambda: deps.search_mode.current().mode,
            search_log=search_log,
            session_factory=deps.read_session_factory,
        )
        graph = build_agent(deps.models, tools, registry, guard, cancel)
        # The mode is read again per search, so an ingestion starting mid-answer switches the
        # next search; the prewarm and the open-answer window only matter for a hybrid answer.
        hybrid = decision.mode == "hybrid"
        with _answer_window(deps.embedder, hybrid=hybrid):
            stream = stream_answer(
                graph,
                prompt,
                cancel,
                registry,
                history=history,
                skip_sources=restored_sources,
            )
            # Not `yield from`: closing a generator suspended in a yield-from closes the
            # sub-generator before this frame can set the flag, so the flag is set first.
            for event in stream:
                yield event
                if isinstance(event, AnswerEvent):
                    # After the answer, so a failed or aborted answer stores nothing. The write is
                    # one transaction on the writable factory; a failure is logged by type and the
                    # stream ends without a thread event, so the answer still reaches the reader.
                    stored = _save_turn(
                        deps,
                        thread_id,
                        question,
                        event.answer,
                        search_log,
                        registry,
                        expected_turns,
                    )
                    if stored is not None:
                        stored_id, turn = stored
                        yield ThreadEvent(stored_id, turn)
    except AnswerAborted:
        # The consumer left (see `AnswerAborted`): there is nobody to report to. The `finally`
        # below still releases the session, and `ChatCallLog` has already committed any usage row.
        return
    except DailyCapReached as exc:
        # The sheet stays useful: one search still runs, so the reader sees candidate sources
        # beside the message about the reset time (section 15's call).
        embedder = deps.embedder if deps.search_mode.current().mode == "hybrid" else None
        yield from _cap_fallback(
            deps, session, embedder, question, filters, registry, restored_sources
        )
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


def _load_thread(deps: AgentDeps, thread_id: UUID) -> ThreadDetailRow | None:
    """The thread an answer continues: its registry and its finished turns."""
    with closing((deps.read_session_factory or deps.session_factory)()) as reader:
        return Repository(reader).thread(thread_id)


def _save_turn(
    deps: AgentDeps,
    thread_id: UUID | None,
    question: str,
    answer: CheckedAnswer,
    search_log: SearchLog,
    registry: SourceRegistry,
    expected_turns: int,
) -> tuple[UUID, int] | None:
    """Store one finished turn and the thread's registry; None when it could not be stored.

    The answer event has already been yielded, so every failure here is logged by its type only
    and swallowed: the reader keeps the checked answer either way. Nothing is logged at info level
    with the question in it, which is why only the error type is recorded.
    """
    writer = deps.write_session_factory
    if writer is None:
        return None
    # The same payload the stream sent, built without the log line: the answer is not unsupported
    # anew because a thread stored it.
    payload = _answer_fields(answer, search_log, _visible_flags(answer))
    try:
        with writer() as session:
            stored_id, turn = Repository(session).save_thread_turn(
                thread_id=thread_id,
                title=question[:THREAD_TITLE_MAX_CHARS],
                registry=[ref_to_json(ref) for ref in registry.refs()],
                question=question,
                answer=payload.model_dump(mode="json"),
                search_mode=payload.search_mode,
                at=deps.clock(),
                expected_turns=expected_turns,
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 - the answer is already the reader's
        log.warning("thread_save_failed", error_type=type(exc).__name__)
        return None
    return stored_id, turn


@contextmanager
def _answer_window(embedder: Embedder, *, hybrid: bool) -> Iterator[None]:
    """Prewarm the model while the chat call thinks, and keep it alive until the answer ends."""
    if not hybrid:
        yield
        return
    prewarm = getattr(embedder, "prewarm", None)
    if callable(prewarm):
        prewarm()
    opener = getattr(embedder, "answer_open", None)
    if callable(opener):
        with opener():
            yield
    else:
        yield


def _next_midnight(now: datetime) -> datetime:
    """The UTC midnight when the free tier's daily call count starts again."""
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=1)


@contextmanager
def _read_scope(deps: AgentDeps, session: Session | None) -> Iterator[Session | None]:
    if deps.read_session_factory is None:
        yield session
        return
    short = deps.read_session_factory()
    try:
        short.execute(text("SET TRANSACTION READ ONLY"))
        yield short
    finally:
        short.rollback()
        short.close()


def _cap_fallback(
    deps: AgentDeps,
    session: Session | None,
    embedder: Embedder | None,
    question: str,
    filters: Filters,
    registry: SourceRegistry,
    restored_sources: int,
) -> Generator[SourceEvent, None, None]:
    """One plain search when the daily cap refuses the answer before any source was found.

    "Already found" means this answer added sources, not that the registry is non-empty: a
    follow-up's registry starts with the thread's aliases, and the design keeps the search results
    visible on the daily-cap error.

    A failure here is logged by type and skipped: the cap error is what the client needs, and a
    search that cannot run must not replace it with a traceback.
    """
    if len(registry.refs()) > restored_sources:
        return
    try:
        with _read_scope(deps, session) as active:
            if active is None:
                return
            try:
                hits = hybrid_search(active, embedder, question, k=CAP_FALLBACK_K, filters=filters)
            except EmbedderUnavailable:
                hits = hybrid_search(active, None, question, k=CAP_FALLBACK_K, filters=filters)
            details = Repository(active).admitted_details([hit.item_id for hit in hits])
            refs = [
                registry.register(details[hit.item_id]) for hit in hits if hit.item_id in details
            ]
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
    # "Ask about it" from a suggestion card: the item is registered as S1 before the graph starts,
    # so the answer can talk about "this paper" and still cite a source the check can verify.
    item_id: UUID | None = None
    # A follow-up: the thread this answer continues. The route refuses an unknown id with a 404
    # before the stream starts; `item_id` only pre-registers a source on a new thread.
    thread_id: UUID | None = None

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
            "payload: `tool`, `step`, `source`, `token`, `notice`, `answer`, `thread` or "
            "`error`. A `: ping` comment line arrives after 15 s of silence; it has no `data:`. "
            "`answer` is the last event unless a `thread` or an `error` follows; a client replaces "
            "its streamed text with the `answer` event's `text`, never appends it, and takes every "
            "link from its `citations[].url`."
        ),
    },
    404: {"model": Refusal, "description": "The `thread_id` names no thread."},
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


def _visible_flags(answer: CheckedAnswer) -> tuple[Flag, ...]:
    # The fixed message replaces the model's text, so its flags describe words the reader never
    # saw: they go to the log for the operator and never into the event or a stored turn.
    return () if answer.text == NO_SUPPORT else answer.flags


def _answer_fields(
    answer: CheckedAnswer, search_log: SearchLog, flags: Sequence[Flag]
) -> AnswerData:
    return AnswerData(
        text=answer.text,
        search_mode="text_only" if search_log.text_only else "hybrid",
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


def _answer_payload(answer: CheckedAnswer, search_log: SearchLog) -> AnswerData:
    flags = _visible_flags(answer)
    if answer.text == NO_SUPPORT:
        # The one log line for an unsupported answer; the stored turn reuses `_answer_fields`, so
        # re-fetching a turn never logs it a second time.
        log.info(
            "answer_unsupported",
            flags=[
                {"kind": f.kind, "sentence": f.sentence, "detail": f.detail} for f in answer.flags
            ],
        )
    return _answer_fields(answer, search_log, flags)


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
    # One log per answer: the tools mark it when a search runs by keyword only, and the answer
    # event reads it so `search_mode` describes what actually produced the answer.
    search_log = SearchLog()
    # closing() releases the database session as soon as this generator is closed (a client
    # disconnect) instead of waiting for garbage collection. The cancel event is the response's,
    # so the flag is set the moment the client leaves, not when this generator is closed.
    events = run_question(
        deps,
        body.question,
        _filters(body),
        cancel,
        search_log,
        item_id=body.item_id,
        thread_id=body.thread_id,
    )
    with closing(events):
        for event in events:
            if isinstance(event, NoticeEvent):
                yield _sse(
                    "notice", NoticeData(code=event.code, message=event.message).model_dump()
                )
            elif isinstance(event, ToolEvent):
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
                yield _sse("answer", _answer_payload(event.answer, search_log).model_dump())
            elif isinstance(event, ThreadEvent):
                # Last event of a saved turn: the client follows up with this id, and re-fetches
                # it after a dropped stream.
                yield _sse(
                    "thread",
                    ThreadData(id=event.thread_id, turn=event.turn).model_dump(mode="json"),
                )
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

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A hang-up can cancel the response before `stream_response` starts, so the limiter's
            # slot is freed here: the outermost path, which every ending passes through. The
            # release is idempotent per grant, so a double call cannot free a slot twice.
            if self._release is not None:
                self._release()

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
        # The captain's requirement: docs and the schema are served only to an authenticated
        # request, not removed. The gate is in front of every route, so these three inherit it;
        # the committed docs/api/openapi.json remains the client's reference.
    )
    # Every request, including the routes FastAPI adds itself, meets the gate first. Adding it here
    # rather than in `serve` means a test, a script or a future caller cannot build an open app.
    app.add_middleware(ApiGate, verifier=deps.verifier, allowed_hosts=deps.allowed_hosts)
    # The readiness gate lives on the app so a test can replace it, and so two apps never share
    # one limiter (a limiter belongs to one event loop's concurrency budget).
    app.state.ready_limiter = anyio.CapacityLimiter(READY_CONCURRENCY)
    # The page's read routes take their own short session per request, so a page load never holds
    # a connection the answer pool needs. Tests and the CLI have no separate read factory, and
    # their single session is the read one.
    app.include_router(
        make_runs_router(deps.read_session_factory or deps.session_factory, deps.search_mode)
    )
    # The page's list of picks. Its route module owns the ranking cache, so one process serves one
    # list and a dismissal can drop it.
    suggestions = deps.suggestions or _default_suggestions(deps)
    app.include_router(make_suggestions_router(suggestions))
    # The same suggestions service, so a saved draft takes its item out of the cached list.
    app.include_router(make_drafts_router(deps.drafts or _default_drafts(deps, suggestions)))
    # The Recent menu and the re-fetch after a dropped stream: reads only, so no limiter.
    app.include_router(make_threads_router(deps.read_session_factory or deps.session_factory))
    # The corpus table and the item page: read-only, one short session per request, and never a
    # model, so they stay cheap while an answer holds the model's memory. They read the ranking
    # rules only for the item page's `early` flag, which must agree with a draft's own notice.
    app.include_router(
        make_items_router(
            deps.read_session_factory or deps.session_factory,
            deps.profile_names,
            load_suggestions(SUGGESTIONS_PATH),
        )
    )

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

    @app.get("/readyz", dependencies=[Security(_BEARER)], responses=_GATE_RESPONSES)
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

    @app.get(
        "/status",
        response_model=StatusData,
        dependencies=[Security(_BEARER)],
        responses=_GATE_RESPONSES,
    )
    async def status() -> StatusData:
        """The search mode the page reads on load: hybrid, or keyword-only and why."""
        # In a worker thread: the gate reads the run table, and a paused database must not block
        # the event loop that is serving every other request.
        decision = await anyio.to_thread.run_sync(deps.search_mode.current)
        return StatusData(mode=decision.mode, reason=decision.reason, since=decision.since)

    @app.post(
        "/ask",
        response_class=_EventStream,
        response_model=StreamItem,
        dependencies=[Security(_BEARER)],
        responses={**_ASK_RESPONSES, **_ASK_GATE_RESPONSES},
    )
    async def ask(body: AskRequest, request: Request) -> Response:
        """Answer one question as server-sent events, one `itemSchema` event per frame."""
        # Refused here rather than in `AskRequest`: both would only buy an empty search after a
        # model call was paid for, and the profile names are the process's, not the model's.
        unknown = sorted(set(body.profiles or ()) - deps.profile_names)
        if unknown:
            raise _refused(f"unknown profiles: {', '.join(unknown)}", ["profiles"])
        if body.since is not None and body.since > deps.clock().astimezone(UTC).date():
            raise _refused("since must not be in the future", ["since"])
        if body.item_id is not None:
            # Refused before the stream starts: an id that is not in the corpus would only spend a
            # model call on a question about nothing.
            with closing((deps.read_session_factory or deps.session_factory)()) as reader:
                if not Repository(reader).admitted_item(body.item_id):
                    raise _refused("item_id is not an admitted item", ["item_id"])
        if body.thread_id is not None:
            # The thread's turns are small, so the route reads it for the 404 the plan wants before
            # a stream starts; `run_question` then loads it again for the answer it needs.
            with closing((deps.read_session_factory or deps.session_factory)()) as reader:
                if Repository(reader).thread(body.thread_id) is None:
                    raise HTTPException(status_code=404, detail="thread not found")
        principal: Principal = request.state.principal
        admitted = deps.limiter.admit(principal.subject)
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
