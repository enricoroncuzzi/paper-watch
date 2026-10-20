"""Engine and session factory: the only place a database connection is created."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    # pool_pre_ping drops a stale pooled connection (for example after Postgres restarts) instead
    # of failing the first query of a run. hide_parameters keeps bound values (a user's question,
    # part of an embedding) out of every error message and its log line.
    return create_engine(url, pool_pre_ping=True, hide_parameters=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False keeps loaded rows usable after a commit, so a caller can read an
    # object it just wrote without another query.
    return sessionmaker(engine, expire_on_commit=False)
