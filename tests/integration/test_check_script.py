"""`deploy/vps/check.sh` run against the test database, the way the morning check runs on the VPS.

The script speaks `psql`, and the only psql here lives inside the test container, so the tests
put a shim named `psql` first on PATH that forwards to the container's client. That keeps the
script itself unchanged and still checks the real SQL against the real schema the code writes.
"""

import os
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, make_url
from sqlalchemy.engine import Engine
from testcontainers.community.postgres import PostgresContainer

from paperwatch.domain import Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.store.db import make_session_factory
from paperwatch.store.models import Item, Run
from paperwatch.store.repository import Repository

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("docker") is None, reason="the psql shim needs docker exec"),
]

# The working tree, not the installed package: deploy/ ships with the repo (docs/deploy/vps.md).
SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "vps" / "check.sh"
# Absolute, so the test runs the same shell the VPS's unit does (/bin/sh is dash on Debian).
SHELL = "/bin/sh"
# A fixed instant, so a run's ages are arithmetic and not a race with the clock.
NOW = datetime(2026, 10, 22, 4, 30, tzinfo=UTC)
EXIT_BAD = 1


@pytest.fixture
def write_rows(engine: Engine) -> Iterator[Callable[..., None]]:
    """Write run and item rows another process must see, then delete exactly those rows.

    The `db_session` fixture cannot be used: it rolls back, and check.sh reads through its own
    psql session. Every row is written through the repository, so the test also proves the
    script reads the columns the code writes.
    """
    factory = make_session_factory(engine)
    runs: list[uuid.UUID] = []
    items: list[uuid.UUID] = []

    def write(
        status: str,
        health: dict[str, object] | None = None,
        *,
        finished_ago: timedelta = timedelta(hours=2),
        admitted: int = 2,
    ) -> None:
        finished = datetime.now(UTC) - finished_ago
        started = finished - timedelta(minutes=21)
        with factory() as session:
            repo = Repository(session)
            run_id = repo.start_run(started, finished, at=started)
            counts = {"fetched": 12, "stored": 3, "failed": 0}
            repo.finish_run(run_id, status, counts, 0, health, at=finished)
            runs.append(run_id)
            for n in range(admitted):
                item = SourceItem(
                    "arxiv",
                    f"2510.{n}",
                    f"Agent harness {n}",
                    ("A",),
                    "We study tool use in LLM agent harnesses.",
                    f"https://arxiv.org/abs/2510.{n}",
                    NOW,
                )
                items.append(
                    repo.upsert_item(
                        item,
                        Tier.VETTED_PREPRINT,
                        "agentic-engineering",
                        1.0,
                        Signals(),
                        FakeEmbedder().embed([f"item {n}"])[0],
                    )
                )
            session.commit()

    yield write

    with factory() as cleanup:
        # Items first: a run has no foreign key to them, but the order keeps the intent clear.
        cleanup.execute(delete(Item).where(Item.id.in_(items)))
        cleanup.execute(delete(Run).where(Run.id.in_(runs)))
        cleanup.commit()


@pytest.fixture
def run_check(
    tmp_path: Path, db_url: str, pg_container: PostgresContainer
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run check.sh with an isolated HOME, a backup directory, and the container's psql."""

    def run(
        *,
        backup_age: timedelta | None = timedelta(hours=3),
        min_free_gb: str = "0",
    ) -> subprocess.CompletedProcess[str]:
        shim_dir, home = tmp_path / "bin", tmp_path / "home"
        shim_dir.mkdir(exist_ok=True)
        (home / "backups").mkdir(parents=True, exist_ok=True)
        shim = shim_dir / "psql"
        user = make_url(db_url).username
        shim.write_text(
            "#!/bin/sh\n"
            f'exec docker exec -i {pg_container.get_container_id()} psql -U {user} "$@"\n'
        )
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        if backup_age is not None:
            dump = home / "backups" / "paperwatch-20261022T031500Z.dump"
            dump.write_bytes(b"PGDMP" + b"\0" * 4096)
            stamp = (datetime.now(UTC) - backup_age).timestamp()
            os.utime(dump, (stamp, stamp))
        env = {
            **os.environ,
            "HOME": str(home),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "PW_CHECK_DB": make_url(db_url).database or "",
            "PW_CHECK_MIN_FREE_GB": min_free_gb,
        }
        return subprocess.run(  # noqa: S603 (a fixed script path, no shell)
            [SHELL, str(SCRIPT)], capture_output=True, text=True, env=env, timeout=120, check=False
        )

    return run


def test_check_passes_on_a_healthy_run(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check()
    assert result.returncode == 0, result.stdout + result.stderr
    # The four facts the captain reads every morning, from the columns the code writes.
    assert "succeeded" in result.stdout
    assert '"stored": 3' in result.stdout  # the run's counts, read out of the JSON column
    assert "2 admitted" in result.stdout
    assert "backup" in result.stdout and "check passed" in result.stdout


def test_check_names_the_health_problems(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    health = {"healthy": False, "problems": ["arxiv fetched 0 items", "1 items failed"]}
    write_rows("succeeded", health)
    result = run_check()
    assert result.returncode == EXIT_BAD
    assert "arxiv fetched 0 items" in result.stdout
    assert "1 items failed" in result.stdout
    assert "check failed" in result.stdout


def test_check_fails_when_the_run_itself_failed(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("failed", {"healthy": False, "problems": ["status=failed"]})
    result = run_check()
    assert result.returncode == EXIT_BAD and "failed" in result.stdout


def test_check_fails_without_a_finished_run(
    run_check: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    # A brand-new deployment whose first run never finished must not read as a good morning.
    result = run_check()
    assert result.returncode == EXIT_BAD and "no finished ingestion run" in result.stdout


def test_check_fails_when_the_last_run_is_stale(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # 30 h is past the 26 h freshness limit, so a missed morning is caught even though the run
    # it reports on was healthy.
    write_rows("succeeded", {"healthy": True, "problems": []}, finished_ago=timedelta(hours=30))
    result = run_check()
    assert result.returncode == EXIT_BAD and "stale" in result.stdout


def test_check_fails_without_a_recent_backup(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(backup_age=timedelta(hours=40))
    assert result.returncode == EXIT_BAD and "backup" in result.stdout


def test_check_fails_when_disk_is_short(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    # No disk has 10^9 GB free, so this fails on any machine the suite runs on.
    result = run_check(min_free_gb="1000000000")
    assert result.returncode == EXIT_BAD and "disk" in result.stdout
