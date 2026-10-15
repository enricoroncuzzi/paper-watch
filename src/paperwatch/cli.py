import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import typer
from sqlalchemy import Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from paperwatch.admission import load_admission
from paperwatch.config import get_settings
from paperwatch.domain import Tier, load_profiles
from paperwatch.ingest.pipeline import Sources, run_ingestion
from paperwatch.llm.classifier import Classifier, make_classifier
from paperwatch.llm.embedder import Embedder, FakeEmbedder, make_embedder
from paperwatch.llm.ledger import Ledger
from paperwatch.logging import configure_logging
from paperwatch.retrieval.hybrid import Filters, hybrid_search
from paperwatch.sources.arxiv import ArxivSource
from paperwatch.sources.base import make_http_client
from paperwatch.sources.feeds import FeedSource, load_feeds
from paperwatch.sources.hf_daily import HfDailySource
from paperwatch.sources.semantic_scholar import SemanticScholarSource
from paperwatch.store.db import make_engine, make_session_factory

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
_TIER_VALUES = frozenset(t.value for t in Tier)
# The repository root: this module is src/paperwatch/cli.py, so parents[2] is the root. Data paths
# resolve from here, not the current directory, so `paperwatch eval` works from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DATA = _REPO_ROOT / "evals" / "data"
_BASELINE_PATH = _REPO_ROOT / "evals" / "baseline.json"
# The throwaway database's name: the generator in `_with_throwaway_database`, the sweep's SQL
# pattern and the sweep's Python parse all describe this one shape, so they stay in step. The
# first group is the creation epoch, which is what makes a leftover recognisable as stale.
_EVAL_DB_RE = re.compile(r"^paperwatch_eval_([0-9]+)_[0-9a-f]{32}$")


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
    name = f"paperwatch_eval_{int(time.time())}_{uuid.uuid4().hex}"
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
        ArxivSource(http, profiles),
        HfDailySource(http),
        SemanticScholarSource(http, s.semantic_scholar_api_key),
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
) -> None:
    import json
    import sys

    # The console script's path contains src, not the repository root, so the top-level evals
    # package is invisible without adding the root (the same base the data paths use).
    sys.path.insert(0, str(_REPO_ROOT))
    from evals.retrieval_eval import GATED, DatabaseNotEmpty, evaluate, gate

    def run(session: Session) -> dict[str, float]:
        return evaluate(
            session,
            FakeEmbedder(),
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
    failures = gate(m, json.loads(_BASELINE_PATH.read_text()))
    for fail in failures:
        typer.echo(f"REGRESSION: {fail}")
    raise typer.Exit(1 if failures else 0)
