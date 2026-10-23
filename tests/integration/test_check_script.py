"""`deploy/vps/check.sh` run against the test database, the way the morning check runs on the VPS.

The script speaks `psql`, and the only psql here lives inside the test container, so the tests
put a shim named `psql` first on PATH that forwards to the container's client. That keeps the
script itself unchanged and still checks the real SQL against the real schema the code writes.
A `systemctl` stub does the same for the unit and timer state, so a test never depends on whether
the machine running the suite has a systemd user manager (macOS has none, CI may have one).
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
# The systemctl stub's answers to `--user show-environment`, `is-failed` and `is-active`.
SYSTEMCTL_STUB = """#!/bin/sh
case "$*" in
    "--user show-environment") exit "${STUB_SHOW_ENVIRONMENT:-0}" ;;
    "--user is-failed --quiet paperwatch-ingest.service") exit "${STUB_IS_FAILED:-1}" ;;
    "--user is-active --quiet paperwatch-ingest.timer") exit "${STUB_IS_ACTIVE:-0}" ;;
esac
echo "unexpected systemctl call: $*" >&2
exit 99
"""


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
        status: str | None,
        health: dict[str, object] | None = None,
        *,
        started_ago: timedelta = timedelta(hours=2, minutes=21),
        admitted: int = 2,
        running_for: timedelta | None = None,
    ) -> None:
        with factory() as session:
            repo = Repository(session)
            if status is not None:
                # The run took 21 minutes, so a freshness limit on its start and one on its end
                # differ by exactly that much: a boundary test can tell the two apart.
                started = datetime.now(UTC) - started_ago
                finished = started + timedelta(minutes=21)
                run_id = repo.start_run(started, finished, at=started)
                counts = {"fetched": 12, "stored": 3, "failed": 0}
                repo.finish_run(run_id, status, counts, 0, health, at=finished)
                runs.append(run_id)
            if running_for is not None:
                # A run that never finished: killed by the memory cap, hung, or still working.
                started = datetime.now(UTC) - running_for
                runs.append(repo.start_run(started, started, at=started))
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
        backup: str = "fresh",
        min_free_gb: str = "0",
        extra_env: dict[str, str] | None = None,
        unit_failed: bool = False,
        timer_active: bool = True,
        user_manager: bool = True,
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
        systemctl = shim_dir / "systemctl"
        systemctl.write_text(SYSTEMCTL_STUB)
        systemctl.chmod(systemctl.stat().st_mode | stat.S_IXUSR)

        backups = home / "backups"
        if backup == "no-dir":
            backups.rmdir()
        elif backup != "missing":
            dump = backups / "paperwatch-20261022T031501Z.dump"
            # A stub dump: only its existence, size and mtime matter to the check.
            dump.write_bytes(b"" if backup == "empty" else b"PGDMP" + b"\0" * (4096 - 5))
            age = timedelta(hours=40) if backup == "old" else timedelta(hours=3)
            stamp = (datetime.now(UTC) - age).timestamp()
            os.utime(dump, (stamp, stamp))
        env = {
            **os.environ,
            "HOME": str(home),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "PW_CHECK_DB": make_url(db_url).database or "",
            "PW_CHECK_MIN_FREE_GB": min_free_gb,
            "STUB_SHOW_ENVIRONMENT": "0" if user_manager else "1",
            "STUB_IS_FAILED": "0" if unit_failed else "1",
            "STUB_IS_ACTIVE": "0" if timer_active else "3",
            **(extra_env or {}),
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
    # The facts the captain reads every morning, from the columns the code writes.
    assert "succeeded" in result.stdout
    assert '"stored": 3' in result.stdout  # the run's counts, read out of the JSON column
    assert "2 admitted" in result.stdout
    # The duration the run really took, from finished_at - started_at: it used to print 0h00m.
    assert "took 0h21m" in result.stdout
    assert "backup" in result.stdout and "check passed" in result.stdout
    assert "not failed" in result.stdout and "timer" in result.stdout


def test_check_skips_the_unit_checks_without_a_user_manager(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A bare cron job, or a laptop: no user manager answers, so unit state must not be guessed.
    # It is a warning, not a pass, so a cron that lost its environment cannot hide.
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(user_manager=False, unit_failed=True, timer_active=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN" in result.stdout and "not checked" in result.stdout


def test_check_fails_when_the_unit_is_failed(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A startup crash, an OOM kill, a timeout or exit 1/3 all leave the unit failed.
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(unit_failed=True)
    assert result.returncode == EXIT_BAD and "failed state" in result.stdout


def test_check_fails_when_the_timer_is_not_active(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A disabled or stopped timer means no morning run at all, and no row would ever say so.
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(timer_active=False)
    assert result.returncode == EXIT_BAD and "not active" in result.stdout


def test_check_names_the_health_problems(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    health = {"healthy": False, "problems": ["arxiv fetched 0 items", "1 items failed"]}
    write_rows("succeeded", health)
    result = run_check()
    assert result.returncode == EXIT_BAD
    assert "arxiv fetched 0 items" in result.stdout
    assert "1 items failed" in result.stdout
    assert "check failed (2 problem(s))" in result.stdout


def test_check_prints_warnings_without_failing(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # One missed lab-feed day is a warning, not a bad morning: the check says so and still passes.
    warning = "feed missed once, retried next run: https://lab.example/feed.xml"
    write_rows("succeeded", {"healthy": True, "problems": [], "warnings": [warning]})
    result = run_check()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN" in result.stdout and warning in result.stdout
    assert "FAIL" not in result.stdout
    assert "check passed" in result.stdout


def test_check_reads_a_row_written_before_warnings_existed(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A health verdict from before the warnings key existed must read as no warnings, not crash
    # the query or print an empty warning.
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN" not in result.stdout


def test_check_fails_on_the_verdict_alone(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # The verdict is its own check: no problems listed, and still unhealthy, must not pass.
    write_rows("succeeded", {"healthy": False, "problems": []})
    result = run_check()
    assert result.returncode == EXIT_BAD and "health: false" in result.stdout


@pytest.mark.parametrize("status", ["failed", "budget_exhausted"])
def test_check_fails_when_the_run_did_not_finish(
    write_rows: Callable[..., None],
    run_check: Callable[..., subprocess.CompletedProcess[str]],
    status: str,
) -> None:
    # A healthy-looking verdict on a run that never finished its work must not pass either.
    write_rows(status, {"healthy": True, "problems": []})
    result = run_check()
    assert result.returncode == EXIT_BAD and f"last run {status}" in result.stdout


def test_check_accepts_a_finished_partial_run_with_a_healthy_verdict(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # `partial` is a finished run: the status check accepts it and the health verdict decides.
    # The code always marks a partial run unhealthy, so this pins that the two are separate.
    write_rows("partial", {"healthy": True, "problems": []})
    result = run_check()
    assert result.returncode == 0, result.stdout + result.stderr


def test_check_fails_without_a_finished_run(
    run_check: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    # A brand-new deployment whose first run never finished must not read as a good morning.
    result = run_check()
    assert result.returncode == EXIT_BAD and "no finished ingestion run" in result.stdout


def test_check_fails_when_the_last_run_is_stale(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # 30 h from the start is past the 25 h freshness limit, so a missed morning is caught even
    # though the run it reports on was healthy.
    write_rows("succeeded", {"healthy": True, "problems": []}, started_ago=timedelta(hours=30))
    result = run_check()
    assert result.returncode == EXIT_BAD and "stale" in result.stdout


@pytest.mark.parametrize(
    ("started_ago", "expected"),
    [
        (timedelta(hours=25, minutes=30), EXIT_BAD),
        (timedelta(hours=24, minutes=30), 0),
        # Started 25 h 11 m ago, finished 24 h 50 m ago: only a limit measured from the start sees
        # this as stale, which is what makes a missed 06:00 run show by 07:00 rather than 07:21.
        (timedelta(hours=25, minutes=11), EXIT_BAD),
    ],
)
def test_check_measures_freshness_from_the_run_start(
    write_rows: Callable[..., None],
    run_check: Callable[..., subprocess.CompletedProcess[str]],
    started_ago: timedelta,
    expected: int,
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []}, started_ago=started_ago)
    result = run_check()
    assert result.returncode == expected, result.stdout + result.stderr


def test_check_fails_when_a_run_stayed_running(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A run killed by the memory cap or hung leaves its row 'running' and no finished row, so
    # without this the check reads yesterday's green run as today's.
    write_rows("succeeded", {"healthy": True, "problems": []})
    write_rows("succeeded", {"healthy": True, "problems": []}, running_for=timedelta(hours=4))
    result = run_check()
    assert result.returncode == EXIT_BAD and "running" in result.stdout


@pytest.mark.parametrize(
    ("running_for", "expected"),
    [
        # The 3 h default, pinned from both sides so no other value can pass this pair. It matches
        # the unit's TimeoutStartSec, so a run that is genuinely still working is not called dead.
        (timedelta(hours=3, minutes=30), EXIT_BAD),
        (timedelta(hours=2, minutes=30), 0),
    ],
)
def test_check_pins_the_three_hour_stuck_run_limit(
    write_rows: Callable[..., None],
    run_check: Callable[..., subprocess.CompletedProcess[str]],
    running_for: timedelta,
    expected: int,
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    write_rows(None, running_for=running_for)
    result = run_check()
    assert result.returncode == expected, result.stdout + result.stderr


def test_check_reports_a_closed_killed_run_once(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A run killed by the memory cap (or SIGKILL) is closed by the next run as `failed` with no
    # health verdict, and the pipeline's own crash path writes the same shape. One event, one FAIL:
    # the status, health and freshness lines would each report it again, and the last two are noise
    # because a killed run has nothing to say about them.
    write_rows("failed", None, started_ago=timedelta(hours=30))
    result = run_check()
    assert result.returncode == EXIT_BAD
    assert result.stdout.count("FAIL") == 1, result.stdout
    assert "killed or crashed" in result.stdout
    assert "health:" not in result.stdout and "stale" not in result.stdout


def test_check_accepts_a_run_that_is_still_working(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    write_rows(None, running_for=timedelta(minutes=10))
    result = run_check()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "in progress" in result.stdout


def test_check_fails_without_a_recent_backup(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(backup="old")
    assert result.returncode == EXIT_BAD and "backup:" in result.stdout


def test_check_fails_on_a_zero_byte_dump(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(backup="empty")
    assert result.returncode == EXIT_BAD and "backup:" in result.stdout


def test_check_prints_a_small_dump_in_kilobytes(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A 4 KiB stub dump is not a plausible backup but is the smallest size the check accepts, and
    # "0 MB" would read like an empty file.
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "backup: 4 KB" in result.stdout


def test_check_fails_without_a_dump(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(backup="missing")
    assert result.returncode == EXIT_BAD and "no backup in" in result.stdout


def test_check_fails_without_a_backup_directory(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(backup="no-dir")
    assert result.returncode == EXIT_BAD and "no backup directory" in result.stdout


def test_check_fails_when_disk_is_short(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    write_rows("succeeded", {"healthy": True, "problems": []})
    # No disk has 10^9 GB free, so this fails on any machine the suite runs on.
    result = run_check(min_free_gb="1000000000")
    assert result.returncode == EXIT_BAD and "disk" in result.stdout


def test_check_rejects_a_non_numeric_threshold(
    write_rows: Callable[..., None], run_check: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    # A typo must be a clean exit 1 with a message, not a shell arithmetic error with exit 2.
    write_rows("succeeded", {"healthy": True, "problems": []})
    result = run_check(extra_env={"PW_CHECK_MIN_FREE_GB": "2 GB"})
    assert result.returncode == EXIT_BAD
    assert "PW_CHECK_MIN_FREE_GB" in result.stdout
