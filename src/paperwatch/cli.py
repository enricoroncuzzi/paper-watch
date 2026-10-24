import json
import os
import re
import signal
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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
from paperwatch.api.app import (
    AgentDeps,
    ErrorEvent,
    ThreadEvent,
    create_app,
    live_deps,
    run_question,
)
from paperwatch.api.auth import (
    HMAC_SECRET_MIN_CHARS,
    ApiAuthNotConfigured,
    RefuseEveryToken,
    make_verifier,
    mint_hmac_token,
)
from paperwatch.api.routes.drafts import draft_source
from paperwatch.config import (
    ADMISSION_PATH,
    CHAT_MODEL,
    FEEDS_PATH,
    LOOPBACK_HOSTS,
    PROFILES_PATH,
    REPO_ROOT,
    SUGGESTIONS_PATH,
    USAGE_POOL_SIZE,
    Settings,
    get_settings,
)
from paperwatch.domain import Tier, load_profiles
from paperwatch.drafts.check import check_draft
from paperwatch.drafts.generate import DraftUnusable, draft_text, generate_draft
from paperwatch.drafts.prompts import MAX_POV_CHARS, clean_language
from paperwatch.ingest.pipeline import Followup, Sources, run_ingestion
from paperwatch.llm.chat import (
    CallGuard,
    ChatModel,
    ChatNotConfigured,
    DailyCapReached,
    ModelUnavailable,
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
from paperwatch.store.models import ItemSummary
from paperwatch.store.repository import ChatCallLog, Repository
from paperwatch.suggestions.cards import basis_label, momentum_label
from paperwatch.suggestions.config import MAX_CANDIDATE_DAYS, load_suggestions
from paperwatch.suggestions.prompts import SUMMARY_PROMPT_VERSION
from paperwatch.suggestions.ranking import Scored, rank
from paperwatch.suggestions.summary import SummaryWriter, text_hash

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
# A follow-up continues a thread; without it the answer starts one and the CLI prints its id.
THREAD_OPT = typer.Option(None, "--thread", help="Continue a thread by id")
# `search` runs no sources, so its `--live` help names only the model and the database.
SEARCH_LIVE = typer.Option(
    False, "--live", help="Use the configured embedding model and the database"
)
K = typer.Option(10, min=1)
# The API listens on the loopback interface by default, behind the tunnel: a host option is a
# deliberate choice, never the default.
HOST = typer.Option("127.0.0.1", help="Bind address; keep the API on loopback behind the tunnel")
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
# The feed list and the admission thresholds the ingestion run reads, also from the repository:
# both used to resolve against the working directory, so `paperwatch ingest --live` only worked
# from the checkout.
_FEEDS_PATH = FEEDS_PATH
_ADMISSION_PATH = ADMISSION_PATH
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


@contextmanager
def _throwaway_database() -> Iterator[Session]:
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
                yield session
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


def _with_throwaway_database[T](run: Callable[[Session], T]) -> T:
    with _throwaway_database() as session:
        return run(session)


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
    feeds = load_feeds(_FEEDS_PATH)
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
            load_admission(_ADMISSION_PATH),
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


@app.command()
def suggest(
    live: bool = LIVE,
    summaries: bool = typer.Option(
        False,
        "--summaries",
        help="Write the missing summaries now and report how many were checked",
    ),
) -> None:
    """Print the ranked picks and the Rising shelf, as the page shows them."""
    _require_live(live)
    s = get_settings()
    configure_logging(s.log_level)
    config = load_suggestions(SUGGESTIONS_PATH)
    profiles = {profile.name: profile for profile in load_profiles(PROFILES_PATH)}
    now = datetime.now(UTC)
    engine = make_engine(s.database_url)
    session = make_session_factory(engine)()
    writer: SummaryWriter | None = None
    # Counted as the loop goes, so a cap or an outage mid-run still reports what was written.
    checked = excerpts = 0
    rows: list[Any] = []
    try:
        if summaries:
            usage_engine = make_engine(s.database_url, pool_size=USAGE_POOL_SIZE, max_overflow=0)
            usage = ChatCallLog(make_session_factory(usage_engine))
            writer = SummaryWriter(
                make_chat_models(s),
                lambda: _suggestion_guard(usage, s),
            )
        repo = Repository(session)
        candidates = repo.suggestion_candidates(
            now=now,
            max_age=timedelta(
                days=min(
                    max(config.window_days.paper, config.window_days.lab_post),
                    MAX_CANDIDATE_DAYS,
                )
            ),
            momentum_days=config.momentum_days,
        )
        ranking = rank(
            candidates,
            now=now,
            config=config,
            profiles=profiles,
            excluded=repo.excluded_items(),
        )
        rows = [*ranking.picks, *ranking.rising]
        ids = [entry.candidate.item_id for entry in rows]
        sources = repo.summary_sources(ids)
        stored = repo.summaries(ids)
        for lane, entries in (("pick", ranking.picks), ("rising", ranking.rising)):
            for position, entry in enumerate(entries, start=1):
                item_id = entry.candidate.item_id
                if item_id not in sources:
                    continue
                row = stored.get(item_id)
                title, text = sources[item_id]
                if writer is not None and _needs_summary(row, title, text, model=s.chat_model):
                    summary = writer.write(title=title, text=text)
                    repo.upsert_summary(
                        item_id,
                        summary,
                        text_hash=text_hash(title, text),
                        model=s.chat_model,
                        at=datetime.now(UTC),
                    )
                    session.commit()
                    row = repo.summaries([item_id])[item_id]
                status = row.status if row is not None else "pending"
                checked += status == "checked"
                excerpts += status == "excerpt"
                typer.echo(_suggestion_lines(position, lane, entry, status))
        if writer is not None:
            typer.echo(f"summaries: {checked} checked, {excerpts} excerpt of {len(rows)} cards")
    except ChatNotConfigured as exc:
        # A missing key is a setup error with the variable's name, not a traceback from the
        # provider chain, exactly like `ask`.
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except (DailyCapReached, BudgetExceeded, ModelUnavailable) as exc:
        # The checked share is the reason this command exists: report the partial count before
        # the limit that stopped it, so a capped run still says something about the cards.
        if writer is not None:
            typer.echo(f"summaries: {checked} checked, {excerpts} excerpt of {len(rows)} cards")
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    finally:
        session.close()
        engine.dispose()


def _suggestion_guard(usage: ChatCallLog, s: Settings) -> CallGuard:
    # The same two free-tier limits the API uses, from the same settings: a CLI summary run must
    # not be able to spend past the day's cap.
    return CallGuard(
        usage,
        Ledger(s.chat_max_tokens_per_answer),
        s.chat_max_calls_per_day,
        lambda: datetime.now(UTC),
    )


def _needs_summary(row: ItemSummary | None, title: str, text: str, *, model: str) -> bool:
    if row is None:
        return True
    return (
        row.text_hash != text_hash(title, text)
        or row.prompt_version != SUMMARY_PROMPT_VERSION
        or row.model != model
    )


def _suggestion_lines(position: int, lane: str, entry: Scored, status: str) -> str:
    candidate = entry.candidate
    parts = [
        basis_label(
            candidate.tier,
            candidate.basis,
            candidate.venue,
            candidate.hf_upvotes,
            candidate.citation_count,
        )
    ]
    momentum = momentum_label(candidate, datetime.now(UTC))
    if momentum:
        parts.append(momentum)
    parts.append(f"summary: {status}")
    return f"{position}. [{lane}] {candidate.title}\n   {' | '.join(parts)}\n   {candidate.url}"


@app.command()
def draft(
    item_id: str = typer.Argument(..., help="The admitted item's id"),
    pov: str = typer.Option("", "--pov", help="Your own point of view, as text"),
    language: str = typer.Option("en", "--language", help="Language code for the post"),
    live: bool = LIVE,
) -> None:
    """Print one checked LinkedIn draft for an item, with its numbered flags."""
    _require_live(live)
    s = get_settings()
    configure_logging(s.log_level)
    try:
        parsed = uuid.UUID(item_id)
    except ValueError as exc:
        raise typer.BadParameter(f"not a UUID: {item_id}") from exc
    if len(pov) > MAX_POV_CHARS:
        raise typer.BadParameter(f"--pov is longer than {MAX_POV_CHARS} characters")
    try:
        language = clean_language(language)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    config = load_suggestions(SUGGESTIONS_PATH)
    engine = make_engine(s.database_url)
    usage_engine = make_engine(s.database_url, pool_size=USAGE_POOL_SIZE, max_overflow=0)
    session = make_session_factory(engine)()
    try:
        repo = Repository(session)
        item = repo.draft_item(parsed)
        if item is None:
            typer.echo("no admitted item with that id", err=True)
            raise typer.Exit(2)
        summary = repo.summaries([parsed]).get(parsed)
        source = draft_source(item, summary, config)
        # The same guard and chain the API uses: a CLI draft spends the same daily free calls, and
        # its point of view may only go to a provider that refuses training.
        guard = _suggestion_guard(ChatCallLog(make_session_factory(usage_engine)), s)
        text, model = generate_draft(
            make_chat_models(s, purpose="draft"), guard, source, pov=pov, language=language
        )
        check = check_draft(text, source, pov=pov)
        typer.echo(f"draft from {source.title} ({model})")
        typer.echo("")
        typer.echo(draft_text(text))
        typer.echo("")
        if check.flags:
            for number, flag in enumerate(check.flags, start=1):
                typer.echo(f"{number}. [{flag.kind}] {flag.text}: {flag.detail}")
        else:
            typer.echo("No flags: every number, quote and link was found in the stored evidence.")
        counts = ", ".join(
            f"{kind} {checked} checked, {passed} passed"
            for kind, (checked, passed) in check.counts.items()
        )
        typer.echo(f"counts: {counts}")
    except ChatNotConfigured as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except (DailyCapReached, BudgetExceeded, ModelUnavailable, DraftUnusable) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    finally:
        session.close()
        engine.dispose()
        usage_engine.dispose()


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
    thread: uuid.UUID | None = THREAD_OPT,
    live: bool = LIVE,
) -> None:
    """Ask one question about the corpus and stream a citation-checked answer."""
    _require_live(live)
    _validate_query("question", question, tier, profile)
    s = get_settings()
    configure_logging(s.log_level)
    # A missing key is a setup error (exit 2) with the variable's name, not a traceback from
    # inside the first request. `ask` serves no route, so its deps get a verifier that accepts no
    # token rather than failing on API credential settings this command never reads.
    try:
        deps = live_deps(s, RefuseEveryToken())
    except ChatNotConfigured as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    filters = Filters(
        tiers=frozenset(tier) if tier else None,
        profiles=frozenset(profile) if profile else None,
    )
    streamed: list[str] = []
    answer: CheckedAnswer | None = None
    saved_thread: tuple[uuid.UUID, int] | None = None
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
        with closing(run_question(deps, question, filters, cancel, thread_id=thread)) as stream:
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
                elif isinstance(event, ThreadEvent):
                    saved_thread = (event.thread_id, event.turn)
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
    if saved_thread is not None:
        # stderr, like the tool progress: a piped answer is still just the answer. The id is what
        # the next `ask --thread` continues.
        typer.echo(f"thread {saved_thread[0]} turn {saved_thread[1]}", err=True)


def _is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


# `serve` cancels an open stream after this long on shutdown. uvicorn otherwise waits for every
# in-flight answer, which is minutes: Ctrl-C on a server would sit there until the model finished.
GRACEFUL_SHUTDOWN_S = 5


def _served_app(deps: AgentDeps, host: str) -> Any:
    """The app `serve` runs. The credential and host gate is part of `create_app` itself."""
    # Warm the OIDC key set before the first request. This lives in the serve path, not in
    # `live_deps`: `paperwatch ask` builds the same deps and never uses the credential.
    prefetch = getattr(getattr(deps, "verifier", None), "prefetch", None)
    if callable(prefetch):
        prefetch()
    app_ = create_app(deps)
    if not _is_loopback(host):
        # A public bind is the operator's choice, but the gate's host allow-list expects the tunnel
        # name, and binding publicly would put the API on the network without it.
        typer.echo(f"warning: {host} is not loopback; keep the API behind the tunnel", err=True)
    return app_


@app.command()
def serve(host: str = HOST, port: int = PORT, live: bool = LIVE) -> None:
    """Serve the streaming API behind a credential gate (needs a chat API key)."""
    _require_live(live)
    s = get_settings()
    configure_logging(s.log_level)
    try:
        deps = live_deps(s, make_verifier(s))
    except (ChatNotConfigured, ApiAuthNotConfigured) as exc:
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


@app.command()
def api_token(live: bool = LIVE) -> None:
    """Print one short-lived HMAC token for the API (only when PW_API_AUTH=hmac)."""
    _require_live(live)
    s = get_settings()
    if s.api_auth != "hmac":
        # A token minted under the OIDC setting would never be accepted, so this refuses rather
        # than printing something useless.
        typer.echo("PW_API_AUTH is not hmac, so an API token would not be accepted", err=True)
        raise typer.Exit(2)
    secret = s.api_hmac_secret.get_secret_value().strip() if s.api_hmac_secret else ""
    if len(secret) < HMAC_SECRET_MIN_CHARS:
        typer.echo(
            f"PW_API_HMAC_SECRET must be at least {HMAC_SECRET_MIN_CHARS} characters", err=True
        )
        raise typer.Exit(2)
    typer.echo(mint_hmac_token(secret, "operator", time.time()))
