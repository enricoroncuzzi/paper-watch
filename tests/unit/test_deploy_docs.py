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
