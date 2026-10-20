import json
import re
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import structlog
import typer
import uvicorn
from sqlalchemy import Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from paperwatch.admission import load_admission
from paperwatch.agent.citations import NO_SUPPORT, CheckedAnswer
from paperwatch.agent.graph import AnswerEvent, TokenEvent, ToolEvent
from paperwatch.api.app import ErrorEvent, create_app, live_deps, run_question
from paperwatch.config import get_settings
from paperwatch.domain import Tier, load_profiles
from paperwatch.ingest.pipeline import Sources, run_ingestion
from paperwatch.llm.chat import ChatNotConfigured
from paperwatch.llm.classifier import Classifier, make_classifier
from paperwatch.llm.embedder import Embedder, FakeEmbedder, make_embedder
from paperwatch.llm.ledger import Ledger
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
# C0 controls except tab and newline, DEL, C1 controls, and the bidi overrides and isolates. An
# ANSI sequence can clear or recolour a terminal, fake a `Sources:` line, or (OSC 8) turn
# model-written text into a clickable link; bidi overrides can reorder a whole line.
_UNSAFE_TERMINAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
# The repository root: this module is src/paperwatch/cli.py, so parents[2] is the root. Data paths
# resolve from here, not the current directory, so `paperwatch eval` works from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DATA = _REPO_ROOT / "evals" / "data"
_BASELINE_PATH = _REPO_ROOT / "evals" / "baseline.json"
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
    """Replace control characters and bidi overrides with U+FFFD, leaving readable text alone.

    Everything `ask` prints is either model-authored or stored third-party text. A terminal
    interprets escape sequences, so they are neutralised rather than shown: only `Citation.url`
    may produce a clickable link.
    """
    return _UNSAFE_TERMINAL.sub("\ufffd", value)


def _models(ledger: Ledger) -> tuple[Classifier, Embedder]:
    # Providers come from settings (Task 7b): local by default, so a live run needs no API key.
    # The classifier shares the embedder, so the local model is loaded once per run.
    s = get_settings()
    embedder = make_embedder(s, ledger)
    profiles = load_profiles(Path("config/profiles.yaml"))
    return make_classifier(s, ledger, profiles, embedder), embedder


@app.command()
def ingest(live: bool = LIVE, since: str | None = SINCE) -> None:
    _require_live(live)
    # Parse before any model, client or engine is built, so a typo fails fast and cheap.
    since_override = _parse_since(since)
    s = get_settings()
    configure_logging(s.log_level)
    http = make_http_client(s)
    profiles = load_profiles(Path("config/profiles.yaml"))
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
    classifier, embedder = _models(ledger)
    session = make_session_factory(make_engine(s.database_url))()
    res = run_ingestion(
        session,
        sources,
        classifier,
        embedder,
        ledger,
        load_admission(Path("config/admission.yaml")),
        datetime.now(UTC),
        since_override=since_override,
    )
    if res.status == "skipped":
        # Another run holds the lock: this invocation did nothing and exits cleanly.
        typer.echo("skipped: another ingestion run is active")
        raise typer.Exit(0)
    typer.echo(f"{res.status}: {res.counts} tokens={res.tokens}")
    raise typer.Exit(0 if res.status in {"succeeded", "partial"} else 1)


@app.command()
def search(
    query: str,
    k: int = K,
    tier: list[str] | None = TIERS,
    profile: list[str] | None = PROFILES_OPT,
    live: bool = SEARCH_LIVE,
) -> None:
    _require_live(live)
    # Validate before building anything, so a bad argument is a usage error (exit 2).
    if not query.strip():
        raise typer.BadParameter("query must not be blank")
    unknown = [t for t in tier or [] if t not in _TIER_VALUES]
    if unknown:
        raise typer.BadParameter(
            f"--tier must be one of {', '.join(sorted(_TIER_VALUES))}; got {', '.join(unknown)}"
        )
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
    import sys

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
    # Validate before building anything, so a bad argument is a usage error (exit 2).
    if not question.strip():
        raise typer.BadParameter("question must not be blank")
    unknown = [t for t in tier or [] if t not in _TIER_VALUES]
    if unknown:
        raise typer.BadParameter(
            f"--tier must be one of {', '.join(sorted(_TIER_VALUES))}; got {', '.join(unknown)}"
        )
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
    # closing() releases the database session at once, including when an error event ends the
    # loop before the generator's last event was requested.
    with closing(run_question(deps, question, filters)) as stream:
        for event in stream:
            if isinstance(event, TokenEvent):
                streamed.append(event.text)
                # No newline per token: the answer reads as it is written, and a redirect stays
                # plain text.
                typer.echo(terminal_safe(event.text), nl=False)
            elif isinstance(event, ToolEvent):
                # Progress goes to stderr so a piped answer is still just the answer.
                typer.echo(
                    f"tool: {terminal_safe(event.name)} {terminal_safe(json.dumps(event.args))}",
                    err=True,
                )
            elif isinstance(event, AnswerEvent):
                answer = event.answer
            elif isinstance(event, ErrorEvent):
                typer.echo(f"error ({event.code}): {terminal_safe(event.message)}", err=True)
                raise typer.Exit(1)
    if answer is None:
        # The graph always ends in a checked answer, so a stream without one is a wiring bug.
        typer.echo("error (internal): the answer ended without a checked answer", err=True)
        raise typer.Exit(1)
    _render_answer(answer, "".join(streamed))


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
    typer.echo(f"serving on http://{host}:{port}")
    uvicorn.run(create_app(deps), host=host, port=port)
