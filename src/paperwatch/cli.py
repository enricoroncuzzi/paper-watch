from datetime import UTC, datetime
from pathlib import Path

import typer

from paperwatch.admission import load_admission
from paperwatch.config import get_settings
from paperwatch.domain import load_profiles
from paperwatch.ingest.pipeline import Sources, run_ingestion
from paperwatch.llm.classifier import Classifier, make_classifier
from paperwatch.llm.embedder import Embedder, make_embedder
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
    k: int = 10,
    tier: list[str] | None = TIERS,
    profile: list[str] | None = PROFILES_OPT,
    live: bool = LIVE,
) -> None:
    _require_live(live)
    s = get_settings()
    # Only the embedder: building the classifier would embed every profile description for nothing.
    embedder = make_embedder(s, Ledger(s.max_tokens_per_run))
    session = make_session_factory(make_engine(s.database_url))()
    f = Filters(frozenset(tier) if tier else None, frozenset(profile) if profile else None)
    for h in hybrid_search(session, embedder, query, k=k, filters=f):
        typer.echo(f"{h.score:.4f}  [{h.tier}]  {h.title}\n        {h.url}")
