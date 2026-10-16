"""Integration tests for the eval's throwaway database: it is fresh, migrated, used, and dropped."""

import logging
import socket
import time
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError

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
    # The sweep matches `_EVAL_DB_RE`, so the generated names must always fullmatch it.
    assert cli._EVAL_DB_RE.fullmatch(seen[0])
    assert cli._EVAL_DB_RE.fullmatch(seen[1])
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


def test_eval_is_identical_across_fresh_databases(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The evaluator's reproducibility guarantee: two fresh databases get different random uuids,
    # yet every metric must match, because tied branch scores share a rank and equal fused scores
    # are ordered by the item's stable key. This replaced the old `_check_no_ties` tripwire, which
    # could only catch a nondeterminism after the fact and only for the questions the sampled ids
    # happened to expose.
    _settings_for(monkeypatch, db_url)

    def body(session: object) -> dict[str, float]:
        return evaluate(session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)  # type: ignore[arg-type]

    assert cli._with_throwaway_database(body) == cli._with_throwaway_database(body)


def test_sweep_drops_only_its_own_stale_databases(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    admin = _admin(db_url)
    now = int(time.time())
    stale = f"paperwatch_eval_{now - 100000}_{uuid4().hex}"
    recent = f"paperwatch_eval_{now - 60}_{uuid4().hex}"
    owner_role = f"evalowner_{uuid4().hex[:8]}"
    other_owner = f"paperwatch_eval_{now - 100000}_{uuid4().hex}"
    lookalikes = [
        "paperwatch_eval-5_1",
        "paperwatch_eval_20200101",
        "paperwatch_eval_2020_backup",
        f"paperwatch_eval_{uuid4().hex}",  # the old hex-only name, with no epoch
        f"plain_{uuid4().hex[:10]}",
    ]
    created = [stale, recent, other_owner, *lookalikes]
    with admin.connect() as conn:
        conn.execute(text(f'CREATE ROLE "{owner_role}" LOGIN'))
        for name in (stale, recent, *lookalikes):
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        conn.execute(text(f'CREATE DATABASE "{other_owner}" OWNER "{owner_role}"'))
    try:
        _settings_for(monkeypatch, db_url)
        cli._with_throwaway_database(lambda session: None)
        remaining = _database_names(db_url)
        assert stale not in remaining
        assert recent in remaining
        for name in (*lookalikes, other_owner):
            assert name in remaining, name
    finally:
        with admin.connect() as conn:
            for name in created:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(text(f'DROP ROLE IF EXISTS "{owner_role}"'))
        admin.dispose()


def test_sweep_tolerates_a_failing_drop(db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _admin(db_url)
    now = int(time.time())
    stale = f"paperwatch_eval_{now - 100000}_{uuid4().hex}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{stale}"'))
    attempted: list[str] = []

    def failing_drop(engine: Engine, name: str) -> None:
        attempted.append(name)
        raise OperationalError("DROP DATABASE", {}, Exception("drop boom"))

    try:
        monkeypatch.setattr(cli, "_drop_database", failing_drop)
        cli._sweep_stale_databases(admin, now)  # must swallow the failure, not raise
        assert stale in attempted
        assert stale in _database_names(db_url)  # the failed drop left it behind
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{stale}" WITH (FORCE)'))
        admin.dispose()


def test_a_closed_port_says_could_not_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A connection failure is not a missing privilege, so the message must not blame CREATEDB.
    # Bind a socket to get a free port, then close it: nothing listens there, so the connect is
    # refused, and no hard-coded port can ever be in use by accident.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _settings_for(
        monkeypatch, f"postgresql+psycopg://paperwatch:paperwatch@localhost:{port}/paperwatch"
    )
    with pytest.raises(cli.EvalDatabaseError) as info:
        cli._with_throwaway_database(lambda session: None)
    assert "could not reach" in str(info.value)
    assert "CREATEDB" not in str(info.value)


def test_a_createdb_role_without_superuser_blames_the_vector_extension(
    db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    admin = _admin(db_url)
    role = f"createdb_{uuid4().hex[:8]}"
    role_pass = "x"  # noqa: S105 - a throwaway role in the test container, not a real secret
    with admin.connect() as conn:
        conn.execute(
            text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{role_pass}' CREATEDB NOSUPERUSER")
        )
    role_url = (
        make_url(db_url)
        .set(username=role, password=role_pass)
        .render_as_string(hide_password=False)
    )
    try:
        _settings_for(monkeypatch, role_url)
        with pytest.raises(cli.EvalDatabaseError, match="vector extension"):
            cli._with_throwaway_database(lambda session: None)
        # The throwaway database it created is dropped even though the migration failed.
        engine = create_engine(db_url)
        try:
            with engine.connect() as conn:
                left = list(
                    conn.scalars(
                        text(
                            "SELECT datname FROM pg_database WHERE datname ~ :p "
                            "AND pg_get_userbyid(datdba) = :role"
                        ).bindparams(p=cli._EVAL_DB_RE.pattern, role=role)
                    )
                )
        finally:
            engine.dispose()
        assert left == []
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP ROLE IF EXISTS "{role}"'))
        admin.dispose()


def test_a_failing_drop_does_not_replace_the_result(
    db_url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _settings_for(monkeypatch, db_url)
    original_drop = cli._drop_database
    created: list[str] = []

    def failing_drop(engine: Engine, name: str) -> None:
        created.append(name)
        raise OperationalError("DROP DATABASE", {}, Exception("drop boom"))

    monkeypatch.setattr(cli, "_drop_database", failing_drop)
    disposed: list[bool] = []
    original_dispose = Engine.dispose

    def spy_dispose(self: Engine) -> None:
        disposed.append(True)
        original_dispose(self)

    monkeypatch.setattr(Engine, "dispose", spy_dispose)

    def raising_body(session: object) -> None:
        raise RuntimeError("body boom")

    try:
        assert cli._with_throwaway_database(lambda session: 42) == 42
        assert "could not drop throwaway database" in capsys.readouterr().err
        assert len(disposed) >= 2  # the throwaway-session engine and the admin engine
        with pytest.raises(RuntimeError, match="body boom"):
            cli._with_throwaway_database(raising_body)
    finally:
        admin = _admin(db_url)
        for name in created:
            original_drop(admin, name)
        admin.dispose()


def test_upgrade_schema_leaves_logging_alone(db_url: str) -> None:
    # Tests that configure_logger=False is honoured: fileConfig must not replace the root handlers.
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    cli._upgrade_schema(db_url)  # the fixture database is already at head, so no migration runs
    assert list(root.handlers) == before_handlers
    assert root.level == before_level
