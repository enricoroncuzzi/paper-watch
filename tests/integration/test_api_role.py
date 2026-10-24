"""The API's Postgres role writes only what the guide grants it.

The grant block is read from `docs/deploy/vps.md`, so the document and the test cannot drift: a
grant removed from the guide fails here, and a grant that never worked would fail on the server.
The test container has no `paperwatch_api`, so this test creates it (and the two other roles the
block names) itself, as the container's superuser.
"""

import re
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from paperwatch.config import REPO_ROOT
from paperwatch.store.grants import API_ROLE, grant_api_writes

pytestmark = pytest.mark.integration
GUIDE = REPO_ROOT / "docs" / "deploy" / "vps.md"


def _grant_block() -> list[str]:
    document = GUIDE.read_text()
    blocks = re.findall(r"```sql\n(.*?)```", document, re.DOTALL)
    [block] = [block for block in blocks if "GRANT USAGE ON SCHEMA public" in block]
    # Strip comments, then split on semicolons: each statement must run on its own.
    without_comments = re.sub(r"--[^\n]*", "", block)
    return [statement.strip() for statement in without_comments.split(";") if statement.strip()]


def _roles(engine: Engine) -> None:
    with engine.begin() as conn:
        for role in ("paperwatch", "paperwatch_ro", API_ROLE):
            exists = conn.scalar(
                text("SELECT 1 FROM pg_roles WHERE rolname = :name"), {"name": role}
            )
            if exists is None:
                conn.exec_driver_sql(f"CREATE ROLE {role} LOGIN")


def test_the_api_role_writes_only_what_it_is_granted(engine: Engine) -> None:
    _roles(engine)
    with engine.begin() as conn:
        for statement in _grant_block():
            conn.exec_driver_sql(statement)
        # Later slices call this from their migrations; here it proves the helper's statement is
        # valid and reaches the table and sequence the API needs today.
        conn.exec_driver_sql(grant_api_writes(["chat_calls"], sequences=["chat_calls_id_seq"]))

    # The cap's own row is the API's one write today, and its id comes from the sequence.
    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        conn.execute(
            text(
                "INSERT INTO chat_calls (at, provider, model, input_tokens, output_tokens) "
                "VALUES (now(), 'check', 'check', 0, 0)"
            )
        )
        assert conn.scalar(text("SELECT count(*) FROM chat_calls")) >= 1
        conn.rollback()

    # The corpus and the run history stay read-only for the API: a bug or a compromised token
    # cannot rewrite what the ingestion wrote.
    for statement in (
        "UPDATE items SET title = 'rewritten'",
        "DELETE FROM runs",
    ):
        with engine.connect() as conn:
            conn.execute(text("SET ROLE paperwatch_api"))
            with pytest.raises(SQLAlchemyError):
                conn.execute(text(statement))
            conn.rollback()


def test_the_grant_helper_skips_a_missing_role() -> None:
    # CI has no paperwatch_api, and the server has it; the same migration must run in both.
    statement = grant_api_writes(
        ["items_summary"], update=["items_summary"], sequences=["items_summary_id_seq"]
    )
    assert "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'paperwatch_api')" in statement
    assert "GRANT INSERT ON items_summary TO paperwatch_api;" in statement
    assert "GRANT UPDATE ON items_summary TO paperwatch_api;" in statement
    assert "GRANT USAGE ON SEQUENCE items_summary_id_seq TO paperwatch_api;" in statement
    assert Path(GUIDE).exists()  # the document is the source this module reads
