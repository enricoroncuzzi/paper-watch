"""`deploy/vps/backup.sh`: the logic around pg_dump that decides what a restore can use.

The dump itself is Postgres' job, so `pg_dump` here is a stub that writes the file it was asked
for. What is worth testing is the wrapper: the timestamped name, the rename that makes a short
dump impossible, the seven-night retention, and the lock that keeps two backups from sharing the
one fixed temporary file, because a quiet bug in any of those is only found on the morning a
restore is needed.
"""

import fcntl
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "vps" / "backup.sh"
# Absolute, so the test runs the same shell the VPS's unit does (/bin/sh is dash on Debian).
SHELL = "/bin/sh"
# dash runs the script on the VPS, and it is the shell whose signal behaviour differs from
# bash: without a TERM trap bash still runs the EXIT trap, dash does not. The signal test names
# it explicitly and skips where it is not installed.
DASH = shutil.which("dash")
# `date -u +%Y%m%dT%H%M%SZ`: sortable and unambiguous, and it names the morning it can restore.
NAME = re.compile(r"paperwatch-\d{8}T\d{6}Z\.dump")


def _stub_pg_dump(
    directory: Path, *, writes: bool = True, fails: bool = False, signals: bool = False
) -> None:
    """A `pg_dump` first on PATH: it honours `--file=`, and can fail or signal its parent."""
    lines = [
        "#!/bin/sh",
        'for arg in "$@"; do case $arg in --file=*) out=${arg#--file=} ;; esac; done',
    ]
    if writes or fails or signals:
        lines.append("printf 'PGDMP{}' > \"$out\"")
    if fails:
        # A dump that died halfway: a short file is on disk when the process exits non-zero.
        lines.append("exit 1")
    if signals:
        # What `systemctl --user stop` and the 1 h timeout do: SIGTERM, then SIGKILL if ignored.
        # The stub keeps running afterwards, so the script is still waiting for it when the signal
        # arrives, which is the state a real pg_dump is in.
        lines.append('kill -TERM "$PPID"')
        lines.append("sleep 1")
        lines.append("exit 0")
    stub = directory / "pg_dump"
    stub.write_text("\n".join(lines) + "\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)


def _stub_flock(directory: Path, exit_code: int) -> Path:
    """A `flock` first on PATH: it records its arguments and reports the lock held or free.

    A kernel lock cannot be faked, so the real-flock test below needs the real binary. This stub
    is what pins the script's flock branch on a machine that has none (macOS), and it is the only
    way to make `flock -n` refuse without racing a real second process.
    """
    calls = directory / "flock.calls"
    stub = directory / "flock"
    stub.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\nexit {exit_code}\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return calls


# The externals backup.sh calls, so a PATH without flock can still run it: that is the mkdir
# fallback, which is the path a Mac (no flock) and a minimal install take.
LOCK_FALLBACK_TOOLS = ("cat", "date", "find", "mkdir", "mv", "rm", "rmdir", "wc")


def _minimal_bin(tmp_path: Path) -> Path:
    bindir = tmp_path / "minbin"
    bindir.mkdir(exist_ok=True)
    for tool in LOCK_FALLBACK_TOOLS:
        target = shutil.which(tool)
        assert target is not None, tool
        link = bindir / tool
        if not link.exists():
            link.symlink_to(target)
    return bindir


def _run(
    tmp_path: Path,
    backups: Path,
    *,
    writes: bool = True,
    fails: bool = False,
    signals: bool = False,
    shell: str = SHELL,
    keep_days: str | None = None,
    flock_exit: int | None = None,
    no_flock: bool = False,
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    _stub_pg_dump(stub_dir, writes=writes, fails=fails, signals=signals)
    if flock_exit is not None:
        _stub_flock(stub_dir, flock_exit)
    # no_flock keeps neither the stub nor the system's flock on PATH, so the script must fall back.
    tail = str(_minimal_bin(tmp_path)) if no_flock else os.environ["PATH"]
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{tail}",
        "HOME": str(tmp_path),
        "PW_BACKUP_DIR": str(backups),
    }
    if keep_days is not None:
        env["PW_BACKUP_KEEP_DAYS"] = keep_days
    return subprocess.run(  # noqa: S603 (a fixed script path, no shell)
        [shell, str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60, check=False
    )


def _dumps(backups: Path) -> list[str]:
    return sorted(p.name for p in backups.iterdir() if p.name.startswith("paperwatch-"))


def _parts(backups: Path) -> list[str]:
    # iterdir, not glob("*.part"): globbing skips the leading dot of the hidden temporary name.
    return sorted(p.name for p in backups.iterdir() if p.name.endswith(".part"))


def test_backup_writes_one_new_dump_and_no_temporary_file(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    result = _run(tmp_path, backups)
    assert result.returncode == 0, result.stdout + result.stderr
    [dump] = _dumps(backups)
    assert NAME.fullmatch(dump), dump
    assert (backups / dump).read_text() == "PGDMP{}"
    assert _parts(backups) == [], "a renamed dump must leave no partial file behind"
    assert dump in result.stdout


def test_backup_keeps_the_dump_and_its_directory_private(tmp_path: Path) -> None:
    # The dump is the whole corpus and every stored row: owner only, directory included, because
    # the script may be the thing that creates it.
    backups = tmp_path / "backups"
    assert _run(tmp_path, backups).returncode == 0
    [dump] = _dumps(backups)
    assert stat.S_IMODE(backups.stat().st_mode) == 0o700
    assert stat.S_IMODE((backups / dump).stat().st_mode) == 0o600


def test_backup_fails_and_leaves_nothing_when_pg_dump_dies_halfway(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    # The dangerous case: a short file on disk and a non-zero exit. Nothing a restore could pick
    # up may survive, and no partial file may sit there looking like a dump either.
    result = _run(tmp_path, backups, fails=True)
    assert result.returncode != 0
    assert _dumps(backups) == []
    assert _parts(backups) == []


@pytest.mark.skipif(DASH is None, reason="dash is the shell this pins, and it is not installed")
def test_backup_cleans_up_when_the_unit_signals_it(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    # A stop or the 1 h timeout sends SIGTERM while pg_dump is working. Run it under dash, the
    # VPS's /bin/sh, which ends without running the EXIT trap unless TERM is trapped: the script
    # traps TERM, INT and HUP into an exit so the partial file is still removed.
    result = _run(tmp_path, backups, signals=True, shell=str(DASH))
    assert result.returncode != 0
    assert _dumps(backups) == []
    assert _parts(backups) == []


def test_backup_keeps_seven_nights(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    backups.mkdir()
    # Ages in whole days: one to six nights ago are kept, seven and older are past the window.
    for age in (1, 2, 3, 4, 5, 6, 7, 20):
        old = backups / f"paperwatch-20260101T0500{age:02d}Z.dump"
        old.write_bytes(b"PGDMPold")
        stamp = time.time() - age * 86400
        os.utime(old, (stamp, stamp))
    result = _run(tmp_path, backups)
    assert result.returncode == 0, result.stdout + result.stderr
    names = _dumps(backups)
    assert len(names) == 7, names  # six nights inside the window plus tonight's
    assert "paperwatch-20260101T050001Z.dump" in names
    assert "paperwatch-20260101T050007Z.dump" not in names
    assert "paperwatch-20260101T050020Z.dump" not in names


def test_backup_fails_loudly_when_pg_dump_writes_nothing(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    # A dump that produced nothing must not leave a file that looks restorable.
    result = _run(tmp_path, backups, writes=False)
    assert result.returncode != 0
    assert "empty" in result.stderr
    assert _dumps(backups) == []


@pytest.mark.parametrize(
    "keep_days",
    [
        "",  # an explicitly empty value is a mistake, not the default
        "seven",
        "1.5",
        " 7",
        "-1",
        "0",
        "00",
        # dash reads a leading zero as octal: 010 would keep eight nights and 08 would die with
        # "Illegal number" after the dump was written, before retention ran.
        "08",
        "010",
        "10000",
        "99999999999999999999",  # longer than any shell integer: -lt would itself fail
    ],
)
def test_backup_rejects_a_keep_days_that_is_not_a_positive_whole_number(
    tmp_path: Path, keep_days: str
) -> None:
    # A typo must be a clean exit 1 with a message, not an arithmetic error that ends with exit 2
    # and no reason, and not `find -mtime +-1`, which would delete today's dump.
    backups = tmp_path / "backups"
    result = _run(tmp_path, backups, keep_days=keep_days)
    assert result.returncode == 1, (keep_days, result.returncode, result.stderr)
    assert "PW_BACKUP_KEEP_DAYS must be" in result.stderr
    assert list(tmp_path.rglob("*.dump")) == []
    assert list(tmp_path.rglob("*.part")) == []


def test_backup_uses_a_directory_lock_when_flock_is_missing(tmp_path: Path) -> None:
    # macOS and a minimal install have no flock: the fallback lock must still run the backup, and
    # must remove its own lock directory afterwards or every later night would refuse.
    backups = tmp_path / "backups"
    first = _run(tmp_path, backups, no_flock=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert not (backups / ".lock").exists(), "the fallback lock must be released on exit"
    # A second night must not be blocked by the first one's lock.
    second = _run(tmp_path, backups, no_flock=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert _dumps(backups) != [] and _parts(backups) == []


def test_backup_releases_the_directory_lock_when_it_fails(tmp_path: Path) -> None:
    # The EXIT trap has to remove the lock on the error path too, otherwise one failed dump (an
    # outage, a bad argument the script cannot see) would block every night after it.
    backups = tmp_path / "backups"
    result = _run(tmp_path, backups, fails=True, no_flock=True)
    assert result.returncode != 0
    assert not (backups / ".lock").exists()
    assert _dumps(backups) == [] and _parts(backups) == []


def test_backup_refuses_while_the_directory_lock_exists(tmp_path: Path) -> None:
    # The lock is a directory here, so a concurrent run, or a killed one that could not clean up,
    # leaves exactly this: the script must refuse and say how to clear it rather than share the
    # fixed temporary file.
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / ".lock").mkdir()
    result = _run(tmp_path, backups, no_flock=True)
    assert result.returncode == 1
    assert "another backup is running" in result.stderr
    assert "remove it" in result.stderr, "a stale lock needs an action, not just a refusal"
    assert (backups / ".lock").is_dir(), "a held lock is never stolen"
    assert _dumps(backups) == [] and _parts(backups) == []


# Above every kernel's pid_max, so `kill -0` always fails for it: it cannot be a live backup.
DEAD_PID = "2147483647"


def test_backup_clears_a_fallback_lock_whose_process_is_gone(tmp_path: Path) -> None:
    # The fallback lock records the pid: a SIGKILL or a reboot leaves the directory behind, and
    # refusing for ever would mean no backup ever again. The pid is what tells a stale lock from a
    # live one.
    backups = tmp_path / "backups"
    lock = backups / ".lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{DEAD_PID}\n")
    result = _run(tmp_path, backups, no_flock=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clearing the lock a killed run left" in result.stderr
    assert not lock.exists(), "the stale lock and its pid file are gone after the run"
    assert _dumps(backups) != [] and _parts(backups) == []


def test_backup_refuses_a_fallback_lock_whose_process_is_alive(tmp_path: Path) -> None:
    # The other half: a lock naming a live process is a running backup, and the pid alone must
    # never be enough to break into it.
    backups = tmp_path / "backups"
    lock = backups / ".lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{os.getpid()}\n")
    result = _run(tmp_path, backups, no_flock=True)
    assert result.returncode == 1
    assert "another backup is running" in result.stderr
    assert "clearing the lock" not in result.stderr
    assert (lock / "pid").is_file(), "a live lock is never stolen"
    assert _dumps(backups) == [] and _parts(backups) == []


@pytest.mark.parametrize("no_flock", [True, False], ids=["mkdir-fallback", "flock"])
def test_a_losing_backup_never_deletes_the_winners_partial_file(
    tmp_path: Path, no_flock: bool
) -> None:
    # The EXIT trap is armed only after the lock is won, so this is the mutant worth pinning: if it
    # moved above the lock block, a run that loses the race would unlink the winner's half-written
    # `.paperwatch.dump.part` and the winner would rename a truncated dump.
    backups = tmp_path / "backups"
    backups.mkdir()
    part = backups / ".paperwatch.dump.part"
    part.write_text("winner partial")
    if no_flock:
        (backups / ".lock").mkdir()
        result = _run(tmp_path, backups, no_flock=True)
    else:
        result = _run(tmp_path, backups, flock_exit=1)  # flock reports the lock held
    assert result.returncode == 1
    assert part.read_text() == "winner partial", "the loser deleted the winner's partial dump"


def test_backup_takes_the_flock_lock_and_keeps_its_file(tmp_path: Path) -> None:
    # The flock branch: `-n` on the script's own descriptor, and the lock file stays behind (it is
    # a lock, not a dump), which only holds the kernel lock, not the path.
    backups = tmp_path / "backups"
    calls = tmp_path / "bin" / "flock.calls"
    result = _run(tmp_path, backups, flock_exit=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.exists() and calls.read_text().strip() == "-n 9", "a non-blocking flock on fd 9"
    assert (backups / ".lock").is_file()
    assert _dumps(backups) != [] and _parts(backups) == []


def test_backup_refuses_when_flock_reports_the_lock_held(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    result = _run(tmp_path, backups, flock_exit=1)
    assert result.returncode == 1
    assert "another backup is running" in result.stderr
    assert _dumps(backups) == [] and _parts(backups) == []


@pytest.mark.skipif(
    shutil.which("flock") is None, reason="the real flock (util-linux) is not installed here"
)
def test_backup_refuses_a_second_run_holding_the_real_flock(tmp_path: Path) -> None:
    # The production path: the timer starts a run, then a manual run starts. The kernel lock the
    # first one holds is what refuses the second, and releasing it lets the next night through.
    backups = tmp_path / "backups"
    backups.mkdir()
    lock = backups / ".lock"
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(tmp_path, backups)
        assert result.returncode == 1
        assert "another backup is running" in result.stderr
        assert _dumps(backups) == [] and _parts(backups) == []
    result = _run(tmp_path, backups)
    assert result.returncode == 0, result.stdout + result.stderr
