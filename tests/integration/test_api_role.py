"""The API's Postgres role writes only what the guide grants it.

The grant block is read from `docs/deploy/vps.md`, so the document and the test cannot drift: a
grant removed from the guide fails here, and a grant that never worked would fail on the server.
The test container has no `paperwatch_api`, so this test creates it (and the two other roles the
block names) itself, as the container's superuser.
"""

import importlib.util
import re
from pathlib import Path
from typing import Any

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

    # The cap's own row is the API's one write today, and its id comes from the sequence. The
    # reservation is inserted first and completed by a column-level UPDATE, as the guard does.
    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        call_id = conn.scalar(
            text(
                "INSERT INTO chat_calls (at, provider, model, input_tokens, output_tokens) "
                "VALUES (now(), 'check', 'reserved', 0, 0) RETURNING id"
            )
        )
        conn.execute(
            text(
                "UPDATE chat_calls SET model = 'check', input_tokens = 1, output_tokens = 2 "
                "WHERE id = :id"
            ),
            {"id": call_id},
        )
        assert conn.scalar(text("SELECT count(*) FROM chat_calls")) >= 1
        conn.rollback()

    # Column level: the API can complete a reservation but cannot move it to another day.
    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        call_id = conn.scalar(
            text(
                "INSERT INTO chat_calls (at, provider, model, input_tokens, output_tokens) "
                "VALUES (now(), 'check', 'reserved', 0, 0) RETURNING id"
            )
        )
        with pytest.raises(SQLAlchemyError):
            conn.execute(text("UPDATE chat_calls SET at = now() WHERE id = :id"), {"id": call_id})
        conn.rollback()

    # The corpus and the run history stay read-only for the API: a bug or a compromised token
    # cannot rewrite what the ingestion wrote.
    for statement in (
        "UPDATE items SET title = 'rewritten'",
        "DELETE FROM runs",
        "DELETE FROM chat_calls",
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


def test_the_grant_helper_never_emits_an_empty_grant() -> None:
    # A migration that only needs an UPDATE grant must not produce `GRANT INSERT ON  TO ...`.
    statement = grant_api_writes(update=["chat_calls"])
    assert "GRANT INSERT" not in statement
    assert "GRANT UPDATE ON chat_calls TO paperwatch_api;" in statement


def test_the_suggestion_grants_cover_what_the_api_writes(engine: Engine) -> None:
    # The API records interactions and stores summaries; those are its only writes in slice 4, and
    # they come from migration 0006 rather than the admin's block, so the captain has no new step.
    _roles(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            grant_api_writes(["suggestion_events"], sequences=["suggestion_events_id_seq"])
        )
        conn.exec_driver_sql(grant_api_writes(["item_summaries"], update=["item_summaries"]))
        item_id = conn.scalar(
            text(
                "INSERT INTO items "
                "(id, source, external_id, status, profile, relevance, title, authors, text, url, "
                "published_at, version, licence, venue_aliases, citation_count, admit_attempts) "
                "VALUES (gen_random_uuid(), 'arxiv', 'grant-check', 'candidate', NULL, 0.0, 't', "
                "'[]', 'body', 'https://example.org/x', now(), 1, 'unknown', '[]', 0, 0) "
                "RETURNING id"
            )
        )

    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        # An id from the table's own sequence, then the completion the queue does.
        event_id = conn.scalar(
            text(
                "INSERT INTO suggestion_events (item_id, event, at) "
                "VALUES (:item_id, 'shown', now()) RETURNING id"
            ),
            {"item_id": item_id},
        )
        assert event_id is not None
        conn.rollback()

    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        conn.execute(
            text(
                "INSERT INTO item_summaries "
                "(item_id, text_hash, model, prompt_version, status, fields, failed_checks, "
                "created_at) VALUES (:item_id, 'hash', 'model', 's1', 'excerpt', '{}', '[]', now())"
            ),
            {"item_id": item_id},
        )
        conn.execute(
            text("UPDATE item_summaries SET status = 'checked' WHERE item_id = :item_id"),
            {"item_id": item_id},
        )
        conn.rollback()

    # Still no write grant on the corpus itself: the API records interactions, it does not curate.
    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        with pytest.raises(SQLAlchemyError):
            conn.execute(text("UPDATE items SET title = 'rewritten'"))
        conn.rollback()

    # This test commits its own row (the grants are connection-level), so it removes it again: a
    # leftover candidate item would make every later test in the session see one more due paper.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM items WHERE external_id = 'grant-check'"))


def test_the_draft_grant_lets_the_api_save_a_draft(engine: Engine) -> None:
    # Generating is saving, so the API inserts a draft and nothing else: no UPDATE (a saved draft
    # is what the check saw) and no DELETE, and the grant comes from migration 0007.
    _roles(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(grant_api_writes(["drafts"]))
        item_id = conn.scalar(
            text(
                "INSERT INTO items "
                "(id, source, external_id, status, profile, relevance, title, authors, text, url, "
                "published_at, version, licence, venue_aliases, citation_count, admit_attempts) "
                "VALUES (gen_random_uuid(), 'arxiv', 'draft-grant-check', 'candidate', NULL, 0.0, "
                "'t', '[]', 'body', 'https://example.org/x', now(), 1, 'unknown', '[]', 0, 0) "
                "RETURNING id"
            )
        )

    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        draft_id = conn.scalar(
            text(
                "INSERT INTO drafts "
                "(id, item_id, pov, language, model, prompt_version, inputs, hook, body, hashtags, "
                'source_link, "check", created_at) VALUES (gen_random_uuid(), :item_id, '
                "'', 'en', 'model', 'd1', '{}', 'h', 'b', '[]', "
                "'https://example.org/x', '{}', now()) "
                "RETURNING id"
            ),
            {"item_id": item_id},
        )
        assert draft_id is not None
        conn.rollback()

    for statement in ("UPDATE drafts SET body = 'rewritten'", "DELETE FROM drafts"):
        with engine.connect() as conn:
            conn.execute(text("SET ROLE paperwatch_api"))
            with pytest.raises(SQLAlchemyError):
                conn.execute(text(statement))
            conn.rollback()

    # This test commits its own item (the grants are connection-level), so remove it again.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM items WHERE external_id = 'draft-grant-check'"))


def test_migration_0007_grants_the_api_insert_on_drafts() -> None:
    # The grant is in the migration, not the admin's block, so the revision itself must carry it:
    # the CI container has no paperwatch_api role to make a runtime check meaningful.
    text = (REPO_ROOT / "migrations" / "versions" / "0007_drafts.py").read_text()
    assert 'grant_api_writes(["drafts"])' in text


def _migration(number: str) -> Any:
    """The real migration module: the test checks what ships, not a copy of its arguments."""
    path = REPO_ROOT / "migrations" / "versions" / f"{number}.py"
    spec = importlib.util.spec_from_file_location(f"migration_{number}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_thread_migration_runs_the_grant_it_defines() -> None:
    # The test above runs `grant_statement()` by hand, so a migration whose `upgrade()` stopped
    # executing it would pass there and deny the API on the server. The DDL is stubbed: what is
    # under test is that the grant is one of the statements the upgrade executes.
    migration = _migration("0008_threads")
    executed: list[str] = []

    class Recorder:
        def create_table(self, *args: object, **kwargs: object) -> None:
            return None

        def create_index(self, *args: object, **kwargs: object) -> None:
            return None

        def execute(self, statement: object) -> None:
            executed.append(str(statement))

    migration.op = Recorder()
    migration.upgrade()
    assert executed == [migration.grant_statement()]


def test_the_thread_grants_cover_what_the_api_writes(engine: Engine) -> None:
    # Migration 0008: the API creates a thread and updates it as turns arrive, inserts each
    # finished turn from its own sequence, and can do nothing else with either table. The
    # statements come from the migration itself, so a grant dropped from it fails here.
    migration = _migration("0008_threads")
    statement = migration.grant_statement()
    assert "GRANT INSERT ON threads, thread_turns TO paperwatch_api;" in statement
    assert "GRANT UPDATE ON threads TO paperwatch_api;" in statement
    assert "GRANT USAGE ON SEQUENCE thread_turns_id_seq TO paperwatch_api;" in statement

    _roles(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(statement)

    with engine.connect() as conn:
        conn.execute(text("SET ROLE paperwatch_api"))
        thread_id = conn.scalar(
            text(
                "INSERT INTO threads (id, title, source_registry, created_at, updated_at) "
                "VALUES (gen_random_uuid(), 'first question', '[]', now(), now()) RETURNING id"
            )
        )
        assert thread_id is not None
        turn_id = conn.scalar(
            text(
                "INSERT INTO thread_turns (thread_id, n, question, answer, search_mode, "
                "created_at) VALUES (:thread_id, 1, 'q', '{}', 'hybrid', now()) RETURNING id"
            ),
            {"thread_id": thread_id},
        )
        assert turn_id is not None
        conn.execute(
            text("UPDATE threads SET updated_at = now() WHERE id = :id"), {"id": thread_id}
        )
        conn.rollback()

    for statement in ("DELETE FROM thread_turns", "DELETE FROM threads"):
        with engine.connect() as conn:
            conn.execute(text("SET ROLE paperwatch_api"))
            with pytest.raises(SQLAlchemyError):
                conn.execute(text(statement))
            conn.rollback()
