"""Integration fixtures: a migrated pgvector Postgres container, one session per test."""

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

from paperwatch.store.db import make_engine, make_session_factory


@pytest.fixture(scope="session")
def db_url() -> Iterator[str]:
    # One container for the whole test session: starting Postgres and migrating is the slow part.
    with PostgresContainer("pgvector/pgvector:pg16", driver="psycopg") as pg:
        url = pg.get_connection_url()
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        yield url


@pytest.fixture(scope="session")
def engine(db_url: str) -> Iterator[Engine]:
    # One engine for the whole session: a fresh engine per test would leave each pool
    # undisposed. Disposing after the last test closes every connection.
    eng = make_engine(db_url)
    yield eng
    eng.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    # Each test runs inside one outer transaction that is rolled back, so tests never see each
    # other's writes and the schema migration is paid for only once.
    conn = engine.connect()
    trans = conn.begin()
    session = make_session_factory(engine)(bind=conn, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    trans.rollback()
    conn.close()
