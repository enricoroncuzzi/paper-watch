"""The database URL the VPS guide tells the captain to write must be the socket form.

A `localhost:5432` URL would silently try TCP, where the server's Postgres has no password
authentication configured and no listening port for the `paperwatch` role: the run would fail on
the server but work on a laptop, which is the worst kind of documentation bug. These tests read
the guide itself, so a drift in the doc fails the suite.
"""

import re

from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.engine import make_url

from paperwatch.config import REPO_ROOT

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


def test_the_ingest_unit_enables_follow_ups() -> None:
    exec_start = [line for line in UNIT.read_text().splitlines() if line.startswith("ExecStart=")]
    assert len(exec_start) == 1
    assert exec_start[0].endswith("paperwatch ingest --live --followup")


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
