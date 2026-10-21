"""Engine and session factory: the only place a database connection is created."""

from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from paperwatch.config import CONNECT_TIMEOUT_S


def make_engine(
    url: str,
    *,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    pool_timeout: float | None = None,
) -> Engine:
    # pool_pre_ping drops a stale pooled connection (for example after Postgres restarts) instead
    # of failing the first query of a run. hide_parameters keeps bound values (a user's question,
    # part of an embedding) out of every error message and its log line. The connect timeout
    # (libpq's) turns a paused or unreachable server into a quick failure, not a hung request.
    options: dict[str, Any] = {}
    # None means SQLAlchemy's default, which is right for the short-lived connections of the CLI.
    if pool_size is not None:
        options["pool_size"] = pool_size
    if max_overflow is not None:
        options["max_overflow"] = max_overflow
    if pool_timeout is not None:
        options["pool_timeout"] = pool_timeout
    return create_engine(
        url,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_S},
        **options,
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False keeps loaded rows usable after a commit, so a caller can read an
    # object it just wrote without another query.
    return sessionmaker(engine, expire_on_commit=False)
