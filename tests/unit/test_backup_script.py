"""`deploy/vps/backup.sh`: the logic around pg_dump that decides what a restore can use.

The dump itself is Postgres' job, so `pg_dump` here is a stub that writes the file it was asked
for. What is worth testing is the wrapper: the timestamped name, the rename that makes a short
dump impossible, and the seven-night retention, because a quiet bug in any of those is only found
on the morning a restore is needed.
"""

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


def _run(
    tmp_path: Path,
    backups: Path,
    *,
    writes: bool = True,
    fails: bool = False,
    signals: bool = False,
    shell: str = SHELL,
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    _stub_pg_dump(stub_dir, writes=writes, fails=fails, signals=signals)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PW_BACKUP_DIR": str(backups),
    }
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
