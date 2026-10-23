"""The compose file's Postgres must be reachable only from the machine that runs it.

A bare `5432:5432` publishes on every interface, so the dev database (password `paperwatch`)
answers on whatever network the laptop is on: a cafe, a co-working space, a train. The server
itself has no listening TCP port and the guide's tunnel binds loopback, so the laptop should not
be looser than the thing it mirrors. This reads the compose file, so a drift fails the suite.
"""

import os

import pytest
import yaml
from sqlalchemy.engine import make_url

from paperwatch.config import REPO_ROOT, Settings

COMPOSE = REPO_ROOT / "docker-compose.yml"


def _published_ports() -> list[str]:
    compose = yaml.safe_load(COMPOSE.read_text())
    ports = compose["services"]["db"]["ports"]
    assert isinstance(ports, list)
    return ports


def test_compose_publishes_postgres_on_loopback_only() -> None:
    assert _published_ports() == ["127.0.0.1:5432:5432"]


def test_compose_publishes_the_port_the_local_default_url_points_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default database_url is how `paperwatch ingest` on a laptop finds the container; binding
    # a different host port would break the quickstart while still looking correct.
    for name in [n for n in os.environ if n.startswith("PW_")]:
        monkeypatch.delenv(name)
    [mapping] = _published_ports()
    host, host_port, container_port = mapping.rsplit(":", 2)
    assert host == "127.0.0.1" and container_port == "5432"
    assert make_url(Settings(_env_file=None).database_url).port == int(host_port)
