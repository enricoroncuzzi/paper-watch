"""The grant a migration gives the API role on the tables it adds.

The tables are owned by `paperwatch`, which runs the migrations, so it may grant on its own
tables: no admin step is needed per table. The role exists only on the server, not in CI, so the
whole block is skipped where it is absent and a migration runs unchanged in both places.
"""

from collections.abc import Sequence

API_ROLE = "paperwatch_api"


def grant_api_writes(
    tables: Sequence[str] = (),
    *,
    update: Sequence[str] = (),
    sequences: Sequence[str] = (),
) -> str:
    """One `DO` block: INSERT on `tables`, UPDATE on `update`, USAGE on `sequences`.

    Every list is optional, so a migration that only adds an UPDATE or a sequence grant can use
    the same helper; an empty list is no GRANT at all, never `GRANT INSERT ON  TO ...`.
    """
    statements: list[str] = []
    if tables:
        statements.append(f"GRANT INSERT ON {', '.join(tables)} TO {API_ROLE};")
    if update:
        statements.append(f"GRANT UPDATE ON {', '.join(update)} TO {API_ROLE};")
    if sequences:
        statements.append(f"GRANT USAGE ON SEQUENCE {', '.join(sequences)} TO {API_ROLE};")
    body = " ".join(statements)
    # S608: every table and sequence name is a code constant from a migration, never user input.
    return (
        "DO $$ BEGIN "  # noqa: S608
        f"IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{API_ROLE}') THEN "
        f"{body} END IF; END $$;"
    )
