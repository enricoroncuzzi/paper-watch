"""The database URL the VPS guide tells the captain to write must be the socket form.

A `localhost:5432` URL would silently try TCP, where the server's Postgres has no password
authentication configured and no listening port for the `paperwatch` role: the run would fail on
the server but work on a laptop, which is the worst kind of documentation bug. These tests read
the guide itself, so a drift in the doc fails the suite.
"""

import os
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.engine import make_url
from typer.testing import CliRunner

from paperwatch import cli
from paperwatch.config import REPO_ROOT, Settings
from paperwatch.ingest.health import Health

GUIDE = REPO_ROOT / "docs" / "deploy" / "vps.md"


def _documented_urls() -> list[str]:
    # Every place the guide shows the variable, so prose and code block cannot disagree.
    return re.findall(r"^PW_DATABASE_URL=(\S+)$", GUIDE.read_text(), re.MULTILINE)


def test_the_guide_shows_one_database_url() -> None:
    urls = _documented_urls()
    assert urls, "docs/deploy/vps.md must show the PW_DATABASE_URL line of the env file"
    assert len(set(urls)) == 1, f"the guide shows conflicting URLs: {sorted(set(urls))}"


def test_the_documented_database_url_is_a_passwordless_socket_url() -> None:
    [url] = set(_documented_urls())
    parsed = make_url(url)
    assert parsed.drivername == "postgresql+psycopg"
    assert parsed.username == "paperwatch" and parsed.password is None
    assert parsed.database == "paperwatch"
    # No TCP host and a host in the query string: psycopg reads an absolute path as the directory
    # holding the server's Unix socket, exactly as peer authentication expects.
    assert parsed.host is None
    assert parsed.query["host"] == "/var/run/postgresql"
    args, options = PGDialect_psycopg().create_connect_args(parsed)
    assert args == []  # positional args would mean a DSN string, not the keyword form
    assert options["host"] == "/var/run/postgresql"
    assert options["dbname"] == "paperwatch" and options["user"] == "paperwatch"
    assert "password" not in options


# The unit is the deployment contract for the follow-up window: it must enable the loop, and its
# timeout must stay above the loop's own window cap, or systemd would kill a healthy wait.
UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-ingest.service"


def test_the_ingest_unit_enables_follow_ups(monkeypatch: pytest.MonkeyPatch) -> None:
    # Run the unit's own arguments through the real command: the flags must parse and the loop
    # must reach the pipeline switched on.
    [line] = [line for line in UNIT.read_text().splitlines() if line.startswith("ExecStart=")]
    words = shlex.split(line.removeprefix("ExecStart="))
    args = words[words.index("paperwatch") + 1 :]
    assert args[0] == "ingest"

    seen: dict[str, object] = {}

    def stub(*_args: object, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ()))

    for name in [n for n in os.environ if n.startswith("PW_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(cli, "load_profiles", lambda path: [])
    monkeypatch.setattr(cli, "load_feeds", lambda path: [])
    monkeypatch.setattr(cli, "load_admission", lambda path: object())
    monkeypatch.setattr(cli, "make_http_client", lambda settings: object())
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: object())
    monkeypatch.setattr(cli, "run_ingestion", stub)
    result = CliRunner().invoke(cli.app, args, catch_exceptions=False)
    assert result.exit_code == 0
    assert seen["followup"] is True


def test_the_unit_timeout_stays_above_the_follow_up_window() -> None:
    # FOLLOWUP_MAX_WINDOW is the second bound on the loop (16 h from the start); the unit's
    # TimeoutStartSec is the last resort, so it must be strictly larger.
    import re

    from paperwatch.config import FOLLOWUP_MAX_WINDOW

    [line] = [line for line in UNIT.read_text().splitlines() if line.startswith("TimeoutStartSec=")]
    value = line.split("=", 1)[1].strip()
    match = re.fullmatch(r"(\d+)h", value)
    assert match, f"unexpected TimeoutStartSec format: {value}"
    assert int(match.group(1)) * 3600 > FOLLOWUP_MAX_WINDOW.total_seconds()


BACKUP_UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-backup.service"


def _service_section(unit: Path) -> list[str]:
    # The [Service] lines only: systemd ignores a setting in the wrong section with a warning, so a
    # NoNewPrivileges line under [Unit] would silently vanish.
    lines = unit.read_text().splitlines()
    start = lines.index("[Service]") + 1
    section: list[str] = []
    for line in lines[start:]:
        if line.startswith("[") and line.endswith("]"):
            break
        section.append(line)
    return section


@pytest.mark.parametrize("unit", [UNIT, BACKUP_UNIT], ids=["ingest", "backup"])
def test_each_unit_refuses_to_gain_privileges(unit: Path) -> None:
    # Both units run code from the account's home. Neither needs a setuid or setgid binary, so the
    # setting that stops one from raising its own privileges costs nothing to keep.
    assert "NoNewPrivileges=yes" in _service_section(unit)


def test_the_guide_shows_how_to_read_the_unit_security_score() -> None:
    # The number belongs to the deployed unit on the server, so the guide names the command and how
    # to read it rather than quoting a score that would go stale.
    guide = GUIDE.read_text()
    assert "systemd-analyze --user security paperwatch-ingest" in guide
    assert "Overall exposure level" in guide


def test_the_guide_binds_its_tunnel_to_loopback() -> None:
    # `-L 5433:...` listens on every interface, so on a shared wifi the database would be reachable
    # through the Mac; the guide's forward must name 127.0.0.1 and say when to close it.
    forwards = re.findall(r"^ssh -N -L (\S+) shared-vps$", GUIDE.read_text(), re.MULTILINE)
    assert forwards == ["127.0.0.1:5433:/var/run/postgresql/.s.PGSQL.5432"]
    assert "Close it when you are done" in GUIDE.read_text()


def test_the_guide_gives_each_of_the_two_roles_its_grants() -> None:
    # Beekeeper only reads, but the API and agent are not read-only: the daily chat cap records one
    # row per call in `chat_calls`, so a SELECT-only role fails the first answer. The owning role
    # would let a stray query write the corpus. Both roles and every grant they need are named, so
    # an admin can paste them.
    guide = GUIDE.read_text()
    # The sentence itself, so an unrelated mention of a read-only mode cannot satisfy the test.
    assert "the read-only role `paperwatch_ro` (SELECT only)" in guide
    assert "recommended connection for Beekeeper" in guide
    assert "The M3 API and agent are not read-only" in guide
    # The write the API needs is the cap's own row, and `id` comes from the sequence the migration
    # that creates `chat_calls` adds, so USAGE on that exact name is part of the grant.
    assert "GRANT INSERT ON chat_calls TO paperwatch_api;" in guide
    assert "GRANT USAGE ON SEQUENCE chat_calls_id_seq TO paperwatch_api;" in guide
    # Peer authentication sees the ssh account, so each role needs an ident map (or a password),
    # and default privileges keep the roles reading tables that later migrations create.
    assert "`pg_ident.conf`" in guide
    assert "ALTER DEFAULT PRIVILEGES FOR ROLE paperwatch" in guide

    # Letting those roles in takes a dedicated pg_hba.conf line, and the role field must list all
    # three: the second field is the database and the third the requested role, so a line naming
    # only `paperwatch` would send the other two to the generic local line without a map.
    assert [
        line.split() for line in guide.splitlines() if line.split()[:2] == ["local", "paperwatch"]
    ] == [
        [
            "local",
            "paperwatch",
            "paperwatch,paperwatch_ro,paperwatch_api",
            "peer",
            "map=paperwatch",
        ]
    ]
    # That line must come first, and never carry the map itself: `local all all peer` also covers
    # the other project's database, and a map there would gate that project's connections too.
    assert "placed above the generic" in guide
    assert "Never put `map=` on `local all all peer`" in guide

    # The map's first line keeps the owner, hence the ingestion, in; all three lines are the only
    # system-user/role pairs peer accepts, which is what still restricts access to Unix paperwatch.
    assert [
        line.split()
        for line in guide.splitlines()
        if line.split()[:2] == ["paperwatch", "paperwatch"]
    ] == [
        ["paperwatch", "paperwatch", "paperwatch"],
        ["paperwatch", "paperwatch", "paperwatch_ro"],
        ["paperwatch", "paperwatch", "paperwatch_api"],
    ]

    # The admin reloads the config and proves the owner still connects before trying the two
    # roles, because a bad line in either file would lock out the ingestion.
    assert "pg_reload_conf()" in guide and "pg_ctl reload" in guide
    assert "psql -d paperwatch -c 'select 1'" in guide
