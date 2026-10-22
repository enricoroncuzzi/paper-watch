"""Shared fixtures: hermetic settings for every test, plus a Postgres container for integration."""

import os
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from testcontainers.community.postgres import PostgresContainer

from paperwatch.config import Settings
from paperwatch.domain import Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.store.db import make_engine, make_session_factory
from paperwatch.store.repository import Repository


@pytest.fixture(autouse=True)
def _hermetic_settings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a test function's own Settings away from the developer's environment and real .env.

    The captain keeps real keys in .env (Global Constraints), so a test that built Settings from
    it would pass here and fail on their machine, and vice versa. A test that wants an env var
    still sets it with monkeypatch, which runs after this fixture.

    Two exceptions, both deliberate. Real-model tests (`-m model`) read the real environment on
    purpose: `PW_LOCAL_EMBEDDING_CACHE_DIR` is how `model-evals.yml` points them at the cache it
    primed, and clearing it would make them download the model again. And this fixture is
    function-scoped, so module- and session-scoped fixtures run before it and are not covered.
    """
    if request.node.get_closest_marker("model"):
        return  # real-model tests read PW_LOCAL_EMBEDDING_CACHE_DIR on purpose
    for name in [n for n in os.environ if n.startswith("PW_")]:
        monkeypatch.delenv(name, raising=False)
    # Unset the file rather than passing `_env_file=None` at every call site: the class default
    # then holds for every Settings the code under test builds on its own.
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def build_settings() -> Callable[..., Settings]:
    """Build Settings from the test's own environment only (see `_hermetic_settings`)."""

    def build(**values: object) -> Settings:
        return Settings(**values)  # type: ignore[arg-type]

    return build


@pytest.fixture(scope="session")
def pg_container() -> Iterator[PostgresContainer]:
    # One container for the whole test session: starting Postgres and migrating is the slow part.
    # Pin the pgvector release the search settings need: hnsw.iterative_scan needs 0.8 or later,
    # so an older image would make every search raise, not silently degrade.
    # It is a fixture of its own so a test that runs a real shell script can reach the container
    # itself: `deploy/vps/check.sh` needs `psql`, and the server's client lives in there.
    with PostgresContainer("pgvector/pgvector:0.8.7-pg16", driver="psycopg") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(pg_container: PostgresContainer) -> str:
    url = pg_container.get_connection_url()
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    return url


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


# (external id, title, body, tier): one admitted row each, embedded with FakeEmbedder.
SeedRow = tuple[str, str, str, Tier]

AGENT_ROWS: list[SeedRow] = [
    (
        "swe",
        "SWE-bench results for coding agents",
        "Coding agent harness evaluated on SWE-bench Verified with real repository bugs.",
        Tier.PEER_REVIEWED,
    ),
    (
        "mem",
        "Memory for long-horizon agents",
        "Agents that remember across sessions plan better over long tasks.",
        Tier.VETTED_PREPRINT,
    ),
    (
        "blog",
        "How we build agent harnesses",
        "Our team ships an agent harness with sandboxed tools and evals.",
        Tier.LAB_BLOG,
    ),
]


@pytest.fixture
def seed_items(db_session: Session) -> Callable[[list[SeedRow]], dict[str, uuid.UUID]]:
    def seed(rows: list[SeedRow]) -> dict[str, uuid.UUID]:
        repo, embedder, ids = Repository(db_session), FakeEmbedder(), {}
        for eid, title, body, tier in rows:
            item = SourceItem(
                "arxiv",
                eid,
                title,
                ("Ada Lovelace",),
                body,
                f"https://example.org/{eid}",
                datetime(2026, 10, 1, tzinfo=UTC),
            )
            ids[eid] = repo.upsert_item(
                item,
                tier,
                "agentic-engineering",
                0.9,
                Signals(),
                embedder.embed([f"{title}\n{body}"])[0],
            )
        db_session.flush()
        return ids

    return seed
