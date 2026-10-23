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


def test_the_guide_binds_its_tunnel_to_loopback() -> None:
    # `-L 5433:...` listens on every interface, so on a shared wifi the database would be reachable
    # through the Mac; the guide's forward must name 127.0.0.1 and say when to close it.
    forwards = re.findall(r"^ssh -N -L (\S+) shared-vps$", GUIDE.read_text(), re.MULTILINE)
    assert forwards == ["127.0.0.1:5433:/var/run/postgresql/.s.PGSQL.5432"]
    assert "Close it when you are done" in GUIDE.read_text()


def test_the_guide_recommends_the_read_only_role() -> None:
    # Beekeeper and the M3 API and agent only read: the owning role would let a stray query write.
    guide = GUIDE.read_text()
    assert "paperwatch_ro" in guide
    assert "read-only" in guide and "SELECT only" in guide
