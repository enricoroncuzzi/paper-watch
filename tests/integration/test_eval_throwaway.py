"""Integration tests for the eval's throwaway database: it is fresh, migrated, used, and dropped."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from evals.retrieval_eval import evaluate
from paperwatch import cli
from paperwatch.config import Settings
from paperwatch.llm.embedder import FakeEmbedder

pytestmark = pytest.mark.integration

CORPUS = Path("evals/data/corpus.jsonl")
QUESTIONS = Path("evals/data/questions.jsonl")


def _admin(db_url: str) -> Engine:
    # AUTOCOMMIT: CREATE/DROP DATABASE cannot run inside a transaction.
    return create_engine(db_url, isolation_level="AUTOCOMMIT")


def _database_names(db_url: str) -> set[str]:
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            return set(conn.scalars(text("SELECT datname FROM pg_database")))
    finally:
        engine.dispose()


def _settings_for(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    # Point the wrapper at the container server the fixtures already own, not the developer's.
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(database_url=url))


def test_throwaway_database_is_fresh_and_dropped(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_for(monkeypatch, db_url)
    seen: list[str] = []

    def body(session: object) -> None:
        seen.append(session.scalar(text("SELECT current_database()")))  # type: ignore[attr-defined]

    cli._with_throwaway_database(body)
    cli._with_throwaway_database(body)
    configured = make_url(db_url).database
    assert seen[0].startswith("paperwatch_eval_")
    assert seen[0] != seen[1]
    assert seen[1] != configured
    remaining = _database_names(db_url)
    assert seen[0] not in remaining
    assert seen[1] not in remaining


def test_throwaway_database_is_dropped_when_the_body_raises(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_for(monkeypatch, db_url)
    seen: list[str] = []

    def body(session: object) -> None:
        seen.append(session.scalar(text("SELECT current_database()")))  # type: ignore[attr-defined]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cli._with_throwaway_database(body)
    assert seen[0] not in _database_names(db_url)


def test_the_configured_database_is_never_migrated(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A plain database with no schema: the wrapper must migrate only its throwaway copy of it.
    admin = _admin(db_url)
    plain = f"plain_{uuid4().hex[:12]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{plain}"'))
    plain_url = make_url(db_url).set(database=plain).render_as_string(hide_password=False)
    try:
        _settings_for(monkeypatch, plain_url)
        cli._with_throwaway_database(lambda session: session.scalar(text("SELECT 1")))
        engine = create_engine(plain_url)
        try:
            with engine.connect() as conn:
                assert conn.scalar(text("SELECT to_regclass('alembic_version')")) is None
        finally:
            engine.dispose()
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{plain}" WITH (FORCE)'))
        admin.dispose()


def test_a_role_without_createdb_is_refused(db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _admin(db_url)
    role = f"nocreatedb_{uuid4().hex[:8]}"
    role_pass = "x"  # noqa: S105 - a throwaway role in the test container, not a real secret
    with admin.connect() as conn:
        conn.execute(text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{role_pass}' NOCREATEDB"))
    role_url = (
        make_url(db_url)
        .set(username=role, password=role_pass)
        .render_as_string(hide_password=False)
    )
    try:
        _settings_for(monkeypatch, role_url)
        with pytest.raises(cli.EvalDatabaseError, match="CREATEDB"):
            cli._with_throwaway_database(lambda session: None)
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP ROLE "{role}"'))
        admin.dispose()


def test_a_percent_encoded_password_still_migrates(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A password with URL-encoded characters must survive Alembic's ConfigParser interpolation.
    admin = _admin(db_url)
    role = f"specpw_{uuid4().hex[:8]}"
    role_pass = "p@ss:/w#?%x"  # noqa: S105 - the special characters are the point of the test
    with admin.connect() as conn:
        conn.execute(text(f"CREATE ROLE \"{role}\" SUPERUSER LOGIN PASSWORD '{role_pass}'"))
    role_url = (
        make_url(db_url)
        .set(username=role, password=role_pass)
        .render_as_string(hide_password=False)
    )
    try:
        _settings_for(monkeypatch, role_url)
        assert cli._with_throwaway_database(lambda session: session.scalar(text("SELECT 1"))) == 1
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP ROLE "{role}"'))
        admin.dispose()


def test_two_throwaway_runs_give_identical_metrics(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_for(monkeypatch, db_url)

    def body(session: object) -> dict[str, float]:
        return evaluate(session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)  # type: ignore[arg-type]

    assert cli._with_throwaway_database(body) == cli._with_throwaway_database(body)
