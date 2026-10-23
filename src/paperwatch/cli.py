import json
import os
import re
import signal
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import typer
import uvicorn
from sqlalchemy import Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from paperwatch.admission import load_admission
from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer
from paperwatch.agent.graph import AnswerEvent, TokenEvent, ToolEvent
from paperwatch.api.app import (
    AgentDeps,
    ErrorEvent,
    create_app,
    live_deps,
    run_question,
)
from paperwatch.config import CHAT_MODEL, PROFILES_PATH, REPO_ROOT, Settings, get_settings
from paperwatch.domain import Tier, load_profiles
from paperwatch.ingest.pipeline import Followup, Sources, run_ingestion
from paperwatch.llm.chat import (
    ChatModel,
    ChatNotConfigured,
    DailyCapReached,
    make_chat_models,
)
from paperwatch.llm.classifier import Classifier, make_classifier
from paperwatch.llm.embedder import Embedder, FakeEmbedder, make_embedder
from paperwatch.llm.ledger import BudgetExceeded, Ledger
from paperwatch.logging import configure_logging
from paperwatch.retrieval.hybrid import Filters, hybrid_search
from paperwatch.sources.arxiv import ArxivOaiSource
from paperwatch.sources.base import make_http_client
from paperwatch.sources.feeds import FeedSource, load_feeds
from paperwatch.sources.hf_daily import HfDailySource
from paperwatch.sources.semantic_scholar import SemanticScholarSource
from paperwatch.store.db import make_engine, make_session_factory

log = structlog.get_logger()
app = typer.Typer(no_args_is_help=True)
LIVE = typer.Option(False, "--live", help="Use real sources and the configured models")
SINCE = typer.Option(None, "--since", help="ISO datetime; overrides every source cursor")
# Only the scheduled unit passes this: a manual run must never sit in a wait of hours.
FOLLOWUP = typer.Option(
    False,
    "--followup",
    help="Retry the papers Semantic Scholar refused until 22:00 Europe/Rome",
)
TIERS = typer.Option(None, "--tier")
PROFILES_OPT = typer.Option(None, "--profile")
# `search` runs no sources, so its `--live` help names only the model and the database.
SEARCH_LIVE = typer.Option(
    False, "--live", help="Use the configured embedding model and the database"
)
K = typer.Option(10, min=1)
# The API has no sign-in until M3's Entra ID, so it listens on the loopback interface by default:
# a host option is a deliberate choice, never the default.
HOST = typer.Option("127.0.0.1", help="Bind address; the API has no sign-in yet")
PORT = typer.Option(8000, min=1, max=65535)
_TIER_VALUES = frozenset(t.value for t in Tier)
# Unicode categories a terminal or a renderer acts on rather than shows: Cc (C0/C1 controls and
# DEL), Cf (bidi marks, zero-width and tag characters, the byte-order mark), Cs (lone surrogates,
# which cannot even be encoded), Zl (line separator) and Zp (paragraph separator). An escape
# sequence can clear or recolour a terminal, fake a `Sources:` line, or (OSC 8) turn model-written
# text into a clickable link; a line separator can forge a new line of an answer.
_UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
# ZWNJ and ZWJ are Cf, but Persian and Indic scripts and emoji sequences need them, so removing
# them would corrupt real text. Newline and tab are how the answer is laid out.
_SAFE_FORMAT = frozenset({"\n", "\t", "\u200c", "\u200d"})
# The repository root: this module is src/paperwatch/cli.py, so parents[2] is the root. Data paths
# resolve from here, not the current directory, so `paperwatch eval` works from any directory.
_REPO_ROOT = REPO_ROOT
# The hand-edited profile file, read from the repository for the same reason.
_PROFILES_PATH = PROFILES_PATH
_EVAL_DATA = _REPO_ROOT / "evals" / "data"
_BASELINE_PATH = _REPO_ROOT / "evals" / "baseline.json"
_QA_BASELINE_PATH = _REPO_ROOT / "evals" / "qa_baseline.json"
_QA_RECORDED_DIR = _EVAL_DATA / "qa_recorded"
# A model is reported unavailable when more of the QA set fails on provider errors than this; two
# failures can be an upstream hiccup, twenty mean the model cannot be measured today (Task 18).
QA_MAX_PROVIDER_ERRORS = 2
# The throwaway database's name. The name is built only through `_eval_db_name`, and the sweep's
# SQL pattern and Python parse both use `_EVAL_DB_RE`, so the generator and the matcher cannot
# drift apart. The first group is the creation epoch (10 digits, until the year 2286), which is
# what makes a leftover recognisable as stale.
_EVAL_DB_RE = re.compile(r"^paperwatch_eval_([0-9]{10})_[0-9a-f]{32}$")


def _eval_db_name(now: float) -> str:
    # The one place the name is built, so its output always matches `_EVAL_DB_RE`.
    return f"paperwatch_eval_{int(now)}_{uuid.uuid4().hex}"


class EvalDatabaseError(RuntimeError):
    """The throwaway eval database could not be created, migrated, or reached."""


def _create_database(engine: Engine, name: str) -> None:
    with engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))


def _drop_database(engine: Engine, name: str) -> None:
    # FORCE disconnects any lingering session (Postgres 13+), so the drop cannot hang or fail.
    with engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def _sweep_stale_databases(engine: Engine, now: int) -> None:
    # A SIGTERM or SIGKILL cannot run the finally, so a killed run can leave its database behind.
    # The query matches only this code's exact name (the regex above is the one source of truth),
    # only databases this role owns and never the configured one, so a look-alike or another
    # user's database is never dropped. A leftover older than a day is swept; a recent one is left
    # for a concurrent eval.
    stmt = text(
        "SELECT datname FROM pg_database "
        "WHERE datname ~ :pattern AND NOT datistemplate "
        "AND datname <> current_database() AND pg_get_userbyid(datdba) = current_user"
    ).bindparams(pattern=_EVAL_DB_RE.pattern)
    with engine.connect() as conn:
        names = list(conn.scalars(stmt))
    for name in names:
        match = _EVAL_DB_RE.match(name)
        # The SQL filter and this parse share one pattern, so match can only be None defensively.
        if match is None or now - int(match.group(1)) <= 86400:
            continue
        try:
            _drop_database(engine, name)
        except SQLAlchemyError:
            continue  # not ours to drop; leave it for its owner


def _eval_database_error(exc: SQLAlchemyError) -> EvalDatabaseError:
    # 42501 is insufficient_privilege, the one permission reason a CREATE DATABASE is refused;
    # anything else is a connection, authentication or server failure, not a missing privilege.
    if getattr(getattr(exc, "orig", None), "sqlstate", None) == "42501":
        return EvalDatabaseError(
            f"could not create the throwaway eval database ({exc}); "
            "the database role needs the CREATEDB privilege"
        )
    return EvalDatabaseError(f"could not reach the database to create the eval database ({exc})")


def _upgrade_schema(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    # Alembic's Config is a ConfigParser: a percent-encoded URL (a password with %XX) must be
    # escaped before it is stored, or % starts an interpolation and the migration crashes.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    # Keep this run's output to the JSON and the REGRESSION lines, not Alembic's INFO logging.
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")


def _with_throwaway_database[T](run: Callable[[Session], T]) -> T:
    # A rolled-back seed leaves dead HNSW entries behind, so dozens of runs on one database slowly
    # change the vector branch's answers and can end in a false regression. Each eval run gets its
    # own database, so the index always starts clean and is dropped in the finally.
    url = make_url(get_settings().database_url)
    name = _eval_db_name(time.time())
    admin = make_engine(url.render_as_string(hide_password=False)).execution_options(
        isolation_level="AUTOCOMMIT"
    )
    try:
        _sweep_stale_databases(admin, int(time.time()))
        _create_database(admin, name)
    except SQLAlchemyError as exc:
        admin.dispose()
        raise _eval_database_error(exc) from exc
    try:
        temp_url = url.set(database=name).render_as_string(hide_password=False)
        try:
            _upgrade_schema(temp_url)
        except Exception as exc:
            raise EvalDatabaseError(
                f"could not migrate the throwaway eval database ({exc}); the role must be able "
                "to create the vector extension (superuser, or install it in template1)"
            ) from exc
        engine = make_engine(temp_url)
        try:
            session = make_session_factory(engine)()
            try:
                return run(session)
            finally:
                session.rollback()
                session.close()
        finally:
            engine.dispose()
    finally:
        try:
            _drop_database(admin, name)
        except SQLAlchemyError as exc:
            # A failed cleanup must not replace the run's result, and dispose must still run.
            typer.echo(f"warning: could not drop throwaway database {name}: {exc}", err=True)
        finally:
            admin.dispose()


@app.callback()
def main() -> None:
    """Paper Watch: daily ingestion, hybrid search, and retrieval evals."""
    # A callback keeps `ingest` a subcommand: a Typer app with one command and no callback
    # runs that command as the program itself, and `paperwatch ingest` would be refused.


def _require_live(live: bool) -> None:
    if not live:
        typer.echo("This command touches real services; pass --live. Tests run offline.")
        raise typer.Exit(2)


def _parse_since(value: str | None) -> datetime | None:
    # Click's datetime parser rejects a trailing "Z" and numeric offsets, so parse the text
    # ourselves. A naive value is UTC; an aware one is converted, so a cursor is always UTC.
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"--since must be ISO 8601: {exc}") from exc
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)


def terminal_safe(value: str) -> str:
    """Replace characters a terminal acts on with U+FFFD, leaving readable text alone.

    Everything `ask` prints is either model-authored or stored third-party text. A terminal
    interprets escape sequences and some invisible characters reorder or hide text, so they are
    neutralised rather than shown: only `Citation.url` may produce a clickable link. A lone
    surrogate is also replaced here, because it cannot be encoded and would crash the echo.
    """
    return "".join(
        "\ufffd"
        if unicodedata.category(char) in _UNSAFE_CATEGORIES and char not in _SAFE_FORMAT
        else char
        for char in value
    )


def _exit_on_sigterm(signum: int, frame: object) -> None:
    """Turn `systemctl --user stop` into an orderly end of an ingestion run.

    systemd stops a unit with SIGTERM, which Python's default handler turns into a silent exit: the
    run row would stay `running` (or a follow-up window would look hung) until the next run closed
    it. Raising SystemExit reaches the pipeline's `except BaseException`, which closes the row as
    failed first, so a stopped morning reads as one failed run, not as a hang.
    """
    raise SystemExit(128 + signum)


def _models(ledger: Ledger) -> tuple[Classifier, Embedder]:
    # Providers come from settings (Task 7b): local by default, so a live run needs no API key.
    # The classifier shares the embedder, so the local model is loaded once per run.
    s = get_settings()
    embedder = make_embedder(s, ledger)
    profiles = load_profiles(_PROFILES_PATH)
    return make_classifier(s, ledger, profiles, embedder), embedder


@app.command()
def ingest(live: bool = LIVE, since: str | None = SINCE, followup: bool = FOLLOWUP) -> None:
    _require_live(live)
    # Parse before any model, client or engine is built, so a typo fails fast and cheap.
    since_override = _parse_since(since)
    s = get_settings()
    configure_logging(s.log_level)
    http = make_http_client(s)
    profiles = load_profiles(_PROFILES_PATH)
    feeds = load_feeds(Path("config/feeds.yaml"))
    sources = Sources(
        ArxivOaiSource(http, profiles),
        HfDailySource(http),
        SemanticScholarSource(
            http,
            # The source takes a plain header value; only Settings holds it as a secret.
            s.semantic_scholar_api_key.get_secret_value() if s.semantic_scholar_api_key else None,
        ),
        FeedSource(http, feeds),
        feeds,
    )
    ledger = Ledger(s.max_tokens_per_run)
    session = make_session_factory(make_engine(s.database_url))()
    # A stopped unit must leave a finished row, not a `running` one (see _exit_on_sigterm). Only for
    # as long as the run lasts; a caller that is not the main thread keeps its own handlers.
    try:
        previous_term = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    except ValueError:
        previous_term = None
    try:
        # The pipeline builds the models itself and may release them between follow-up passes, so
        # this frame must never hold them: a local reference here would keep the ~520 MB model
        # alive all day. `_models` is only ever called through the factory below.
        res = run_ingestion(
            session,
            sources,
            lambda: _models(ledger),
            ledger,
            load_admission(Path("config/admission.yaml")),
            datetime.now(UTC),
            since_override=since_override,
            followup=followup,
            bounds=Followup(
                max_passes=s.followup_max_passes,
                interval_s=s.followup_interval_s,
                deadline_hour=s.followup_deadline_hour,
            ),
        )
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
    if res.status == "skipped":
        # Another run holds the lock: this invocation did nothing and exits cleanly.
        typer.echo("skipped: another ingestion run is active")
        raise typer.Exit(0)
    typer.echo(f"{res.status}: {res.counts} tokens={res.tokens}")
    if res.status not in {"succeeded", "partial"}:
        # The run itself did not finish its work (failed, over budget): exit 1.
        raise typer.Exit(1)
    # A warning is a finding worth seeing that never makes the day bad, so it goes to stderr (where
    # the journal already carries the diagnostics) and the exit code stays 0 unless a problem says
    # otherwise.
    warnings = res.health.warnings if res.health is not None else ()
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    if res.health is None or not res.health.healthy:
        # A finished run whose health verdict is bad is not a crash and not a success: 3 tells the
        # morning check and a timer apart from exit 1, and the problems go to stderr where a
        # failure's diagnostics belong. A finished run without a verdict counts as unhealthy, so a
        # lost verdict can never pass for a good morning.
        problems = (
            res.health.problems if res.health is not None else ("the run wrote no health verdict",)
        )
        for problem in problems:
            typer.echo(f"unhealthy: {problem}", err=True)
        raise typer.Exit(3)
    raise typer.Exit(0)


def _profile_names() -> frozenset[str]:
    # Profile names are configuration, so a typo is only knowable from the file.
    return frozenset(p.name for p in load_profiles(_PROFILES_PATH))


def _validate_query(
    noun: str, text: str, tier: list[str] | None, profile: list[str] | None
) -> None:
    """Refuse a bad query as a usage error, before any model, client or engine is built."""
    if not text.strip():
        raise typer.BadParameter(f"{noun} must not be blank")
    unknown = [t for t in tier or [] if t not in _TIER_VALUES]
    if unknown:
        raise typer.BadParameter(
            f"--tier must be one of {', '.join(sorted(_TIER_VALUES))}; got {', '.join(unknown)}"
        )
    known = _profile_names() if profile else frozenset()
    unknown_profiles = [p for p in profile or [] if p not in known]
    if unknown_profiles:
        raise typer.BadParameter(
            f"--profile must be one of {', '.join(sorted(known))}; "
            f"got {', '.join(unknown_profiles)}"
        )


@app.command()
def search(
    query: str,
    k: int = K,
    tier: list[str] | None = TIERS,
    profile: list[str] | None = PROFILES_OPT,
    live: bool = SEARCH_LIVE,
) -> None:
    _require_live(live)
    _validate_query("query", query, tier, profile)
    s = get_settings()
    # Only the embedder: building the classifier would embed every profile description for nothing.
    embedder = make_embedder(s, Ledger(s.max_tokens_per_run))
    session = make_session_factory(make_engine(s.database_url))()
    try:
        f = Filters(frozenset(tier) if tier else None, frozenset(profile) if profile else None)
        for h in hybrid_search(session, embedder, query, k=k, filters=f):
            typer.echo(f"{h.score:.4f}  [{h.tier}]  {h.title}\n        {h.url}")
    finally:
        session.close()


@app.command(name="eval")
def eval_cmd(
    write_baseline: bool = typer.Option(False, "--write-baseline"),
    live: bool = typer.Option(
        False,
        "--live",
        help="Use the configured embedding model; never gates and never writes the baseline",
    ),
) -> None:
    # The gate and the committed baseline belong to the offline FakeEmbedder, so a real-model run
    # is a manual checkpoint: refusing the combination keeps real numbers out of the baseline.
    if live and write_baseline:
        typer.echo("--live cannot be combined with --write-baseline", err=True)
        raise typer.Exit(2)

    # The console script's path contains src, not the repository root, so the top-level evals
    # package is invisible without adding the root (the same base the data paths use).
    sys.path.insert(0, str(_REPO_ROOT))
    from evals.retrieval_eval import GATED, DatabaseNotEmpty, evaluate, gate

    # Real embeddings need the configured model and the run's token budget; the default stays the
    # deterministic fake, so CI is offline and a real run is always an explicit --live.
    if live:
        s = get_settings()
        embedder: Embedder = make_embedder(s, Ledger(s.max_tokens_per_run))
    else:
        embedder = FakeEmbedder()

    def run(session: Session) -> dict[str, float]:
        return evaluate(
            session,
            embedder,
            _EVAL_DATA / "corpus.jsonl",
            _EVAL_DATA / "questions.jsonl",
        )

    try:
        m = _with_throwaway_database(run)
    except (DatabaseNotEmpty, EvalDatabaseError) as exc:
        # A refusal is an error, so it goes to stderr rather than into the metrics stream.
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(m, indent=2))
    if write_baseline:
        # Only the gated metrics are committed: the gate validates exactly these keys.
        baseline = {name: m[name] for name in GATED}
        _BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        return
    if live:
        # A live run only reports; the gated baseline is the fake embedder's, so nothing to check.
        return
    failures = gate(m, json.loads(_BASELINE_PATH.read_text()))
    for fail in failures:
        typer.echo(f"REGRESSION: {fail}")
    raise typer.Exit(1 if failures else 0)


def _qa_record_path(model: str) -> Path:
    # One file per measured model; the slug rule is fixed so a model id maps to one file name.
    return _QA_RECORDED_DIR / f"{model.replace('/', '_').replace(':', '_')}.jsonl"


def _qa_models(settings: Settings, model: str) -> list[ChatModel]:
    # One model and no fallback: a recording is evidence for exactly this id, so a fallback
    # provider quietly answering some questions would put two models in one file.
    settings = settings.model_copy(
        update={"chat_model": model, "groq_api_key": None, "groq_chat_model": None}
    )
    return make_chat_models(settings)


@app.command(name="qa-eval")
def qa_eval(
    live: bool = typer.Option(False, "--live", help="Record live answers from a chat model"),
    model: str | None = typer.Option(None, "--model", help="Model id to record (with --live)"),
    write_baseline: bool = typer.Option(False, "--write-baseline"),
) -> None:
    """Score recorded answers against the committed QA baseline, or record a new model."""
    # The baseline belongs to the committed recording of the default model, so a live recording
    # is a manual measurement and can never rewrite the gate.
    if live and write_baseline:
        typer.echo("--live cannot be combined with --write-baseline", err=True)
        raise typer.Exit(2)
    if model is not None and not live:
        typer.echo("--model is only meaningful with --live", err=True)
        raise typer.Exit(2)

    # The console script's path contains src, not the repository root, so the top-level evals
    # package is invisible without adding the root (the same base the data paths use).
    sys.path.insert(0, str(_REPO_ROOT))
    from evals.qa_eval import (
        GATED,
        check_complete,
        gate,
        load_recorded,
        record,
        score,
        write_recorded,
    )

    questions = _EVAL_DATA / "qa_questions.jsonl"
    corpus = _EVAL_DATA / "corpus.jsonl"
    s = get_settings()
    if live:
        target = model or s.chat_model
        try:
            models = _qa_models(s, target)
        except ChatNotConfigured as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        configure_logging(s.log_level)
        try:
            rows = _with_throwaway_database(
                lambda session: record(session, models, questions, corpus)
            )
        except DailyCapReached as exc:
            # Our own cap, not the provider's: a recording that spends 800 calls in a day is a
            # configuration problem, and a traceback would only hide the sentence that says so.
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        except BudgetExceeded as exc:
            typer.echo(f"the recording exceeded one answer's token budget ({exc})", err=True)
            raise typer.Exit(2) from exc
        errors = sum(row.provider_error for row in rows)
        if errors > QA_MAX_PROVIDER_ERRORS:
            # Not written: a recording with more holes than the rule allows would be evidence for
            # a model we cannot measure today, and retrying to chase a better number is refused.
            typer.echo(
                f"{target} is unavailable: {errors} of {len(rows)} questions failed with "
                "provider errors",
                err=True,
            )
            raise typer.Exit(2)
        path = _qa_record_path(target)
        write_recorded(path, rows)
        typer.echo(json.dumps(score(rows, questions), indent=2))
        typer.echo(f"recorded {len(rows)} answers from {target} to {path}")
        return

    # The gate belongs to the committed default, never to a developer's `PW_CHAT_MODEL`: a
    # replay that followed the environment could gate one model against another's baseline.
    target = CHAT_MODEL
    path = _qa_record_path(target)
    if not path.exists():
        typer.echo(
            f"no recording for {target} at {path}; "
            f"run `paperwatch qa-eval --live --model {target}`",
            err=True,
        )
        raise typer.Exit(2)
    recorded = load_recorded(path)
    try:
        check_complete(recorded, questions, target, path)
    except ValueError as exc:
        # A partial, stale-prompt or wrong-model recording is not a gate result; the fix is to
        # re-record it, not to rescore a file the baseline does not describe.
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    m = score(recorded, questions)
    typer.echo(json.dumps(m, indent=2))
    if write_baseline:
        # Only the gated metrics are committed: the gate validates exactly these keys.
        _QA_BASELINE_PATH.write_text(json.dumps({name: m[name] for name in GATED}, indent=2) + "\n")
        return
    if not _QA_BASELINE_PATH.exists():
        typer.echo(
            f"no QA baseline at {_QA_BASELINE_PATH}; run `paperwatch qa-eval --write-baseline`",
            err=True,
        )
        raise typer.Exit(2)
    failures = gate(m, json.loads(_QA_BASELINE_PATH.read_text()))
    for fail in failures:
        typer.echo(f"REGRESSION: {fail}")
    raise typer.Exit(1 if failures else 0)


def _interrupt_handler(
    cancel: threading.Event, interrupted: threading.Event
) -> Callable[[int, object], None]:
    """Ctrl-C: stop the answer at its next chance, and leave at once on a second press.

    The signal is delivered in the main thread, inside the graph's own frame. Unwinding from
    there joins the model call in flight, so the flag has to be set first; the first press then
    prints one line and lets that phase end (a token arrives, or a tool finishes). A model call
    is a thread that cannot be interrupted, and Python joins it at exit, so a second press is
    the way out of a stalled request: flush what is already written and leave.

    `interrupted` is separate from `cancel` on purpose. `run_question` sets `cancel` in its own
    finally on every ending, so only this handler's event records that the user asked to stop;
    without it a stream that ended without an answer looked like Ctrl-C.
    """
    pressed = False

    def handler(signum: int, frame: object) -> None:
        nonlocal pressed
        cancel.set()
        interrupted.set()
        if pressed:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130)
        pressed = True
        typer.echo("stopping; press Ctrl-C again to quit now", err=True)

    return handler


def _render_answer(answer: CheckedAnswer, streamed: str) -> None:
    """Print the checked answer and its sources after the streamed words, in a fixed shape."""
    if answer.text != streamed:
        # The checked text is authoritative: an unknown alias is masked to [?], and an answer
        # with no supporting source is replaced by the no-support message. Show it under a rule.
        typer.echo("\n---")
        typer.echo(terminal_safe(answer.text))
    else:
        typer.echo()
    if answer.citations:
        typer.echo("Sources:")
        for citation in answer.citations:
            typer.echo(
                f"[{terminal_safe(citation.alias)}] {terminal_safe(citation.title)} "
                f"({terminal_safe(citation.tier)}) {terminal_safe(citation.url)}"
            )
    if answer.text == NO_SUPPORT:
        # The reader never saw the model's words, so its flags are kept for the operator only.
        if answer.flags:
            log.info(
                "answer_unsupported",
                flags=[
                    {"kind": f.kind, "sentence": f.sentence, "detail": f.detail}
                    for f in answer.flags
                ],
            )
        return
    for flag in answer.flags:
        typer.echo(
            f"warning: {flag.kind}: {terminal_safe(flag.sentence)} ({terminal_safe(flag.detail)})"
        )


@app.command()
def ask(
    question: str,
    tier: list[str] | None = TIERS,
    profile: list[str] | None = PROFILES_OPT,
    live: bool = LIVE,
) -> None:
    """Ask one question about the corpus and stream a citation-checked answer."""
    _require_live(live)
    _validate_query("question", question, tier, profile)
    s = get_settings()
    configure_logging(s.log_level)
    # A missing key is a setup error (exit 2) with the variable's name, not a traceback from
    # inside the first request.
    try:
        deps = live_deps(s)
    except ChatNotConfigured as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    filters = Filters(
        tiers=frozenset(tier) if tier else None,
        profiles=frozenset(profile) if profile else None,
    )
    streamed: list[str] = []
    answer: CheckedAnswer | None = None
    cancel = threading.Event()
    # The run also sets `cancel` when it ends normally; only the handler's own event says the
    # user pressed Ctrl-C, which is what decides between exit 130 and a wiring error.
    interrupted = threading.Event()
    # Only the ask command takes over SIGINT, and only for as long as the answer runs: the
    # default handler is restored in the finally below.
    previous = signal.signal(signal.SIGINT, _interrupt_handler(cancel, interrupted))
    try:
        # closing() releases the database session at once, including when an error event ends the
        # loop before the generator's last event was requested; closing also aborts a model call
        # still in flight (the Ctrl-C path sets the flag first, see `_interrupt_handler`).
        with closing(run_question(deps, question, filters, cancel)) as stream:
            for event in stream:
                if isinstance(event, TokenEvent):
                    streamed.append(event.text)
                    # No newline per token: the answer reads as it is written, and a redirect
                    # stays plain text.
                    typer.echo(terminal_safe(event.text), nl=False)
                elif isinstance(event, ToolEvent):
                    # Progress goes to stderr so a piped answer is still just the answer.
                    typer.echo(
                        f"tool: {terminal_safe(event.name)} "
                        f"{terminal_safe(json.dumps(event.args))}",
                        err=True,
                    )
                elif isinstance(event, AnswerEvent):
                    answer = event.answer
                elif isinstance(event, ErrorEvent):
                    if streamed:
                        # Tokens already went to stdout without a newline; end that line first.
                        typer.echo()
                    typer.echo(f"error ({event.code}): {terminal_safe(event.message)}", err=True)
                    raise typer.Exit(1)
    except KeyboardInterrupt as exc:
        # The flag was set by the handler and the stream is closed, so the in-flight model call
        # was aborted; exit like a shell program interrupted by the user, one line, no traceback.
        if streamed:
            typer.echo()
        typer.echo("interrupted: the answer was stopped", err=True)
        raise typer.Exit(130) from exc
    finally:
        signal.signal(signal.SIGINT, previous)
    if answer is None:
        if interrupted.is_set():
            # The user's Ctrl-C stopped the answer before it produced a checked one. `cancel`
            # alone is not enough: the run's own finally sets it on every ending.
            if streamed:
                # Tokens went to stdout without a newline; end that line before the message.
                typer.echo()
            typer.echo("interrupted: the answer was stopped", err=True)
            raise typer.Exit(130)
        # The graph always ends in a checked answer, so a stream without one is a wiring bug.
        typer.echo("error (internal): the answer ended without a checked answer", err=True)
        raise typer.Exit(1)
    _render_answer(answer, "".join(streamed))


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


# The names a browser uses for this machine. A DNS-rebinding page can reach a loopback API with
# a hostile `Host` header, so only these are accepted.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# `serve` cancels an open stream after this long on shutdown. uvicorn otherwise waits for every
# in-flight answer, which is minutes: Ctrl-C on a server would sit there until the model finished.
GRACEFUL_SHUTDOWN_S = 5


def _host_name(header: str) -> str:
    """The host part of a `Host` header, lowercased, or "" when it is not a host at all.

    A browser writes an IPv6 host in brackets (`[::1]:8000`), a port may follow any host, and a
    trailing dot or embedded credentials make a name that is not one of the loopback ones. A
    plain string comparison against `::1` can never match a real header, which is why the
    starlette middleware made `--host ::1` refuse every request.
    """
    if header.startswith("["):
        end = header.find("]")
        if end < 0:
            return ""
        name, rest = header[1:end].lower(), header[end + 1 :]
        # Only `[host]` and `[host]:port` are a host header; `[::1].evil.example` is not.
        if rest and (not rest.startswith(":") or not rest[1:].isdigit()):
            return ""
        return name
    name, sep, port = header.partition(":")
    return name.lower() if not sep or port.isdigit() else ""


class _LoopbackHosts:
    """Refuse a request whose `Host` header is not a name a browser uses for this machine.

    Plain ASGI rather than a `BaseHTTPMiddleware`: `/ask` streams a model answer for seconds, and
    a middleware that buffered or wrapped the response would change the disconnect path
    `_ClosingStream` depends on.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self._app(scope, receive, send)
            return
        header = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
        if _host_name(header) not in _LOOPBACK_HOSTS:
            await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)
            return
        await self._app(scope, receive, send)


def _served_app(deps: AgentDeps, host: str) -> Any:
    """The app `serve` runs, with the host allow-list installed for a loopback bind."""
    app_ = create_app(deps)
    if _is_loopback(host):
        app_.add_middleware(_LoopbackHosts)
    else:
        # A public bind is the operator's choice, but the API has no sign-in until M3.
        typer.echo(f"warning: {host} is not loopback and the API has no sign-in yet", err=True)
    return app_


@app.command()
def serve(host: str = HOST, port: int = PORT, live: bool = LIVE) -> None:
    """Serve the streaming API on the loopback interface (needs a chat API key)."""
    _require_live(live)
    s = get_settings()
    configure_logging(s.log_level)
    try:
        deps = live_deps(s)
    except ChatNotConfigured as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    app_ = _served_app(deps, host)
    # IPv6 literals go in brackets in a URL; `http://::1:8000` is not one.
    url_host = f"[{host}]" if ":" in host else host
    typer.echo(f"serving on http://{url_host}:{port}")
    # log_config=None keeps uvicorn from installing its own plain-text logging, so the app's
    # configured level and format stay in charge.
    uvicorn.run(
        app_,
        host=host,
        port=port,
        log_config=None,
        log_level=s.log_level.lower(),
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
