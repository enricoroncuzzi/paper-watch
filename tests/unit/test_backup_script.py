"""`deploy/vps/backup.sh`: the logic around pg_dump that decides what a restore can use.

The dump itself is Postgres' job, so `pg_dump` here is a stub that writes the file it was asked
for. What is worth testing is the wrapper: the timestamped name, the rename that makes a short
dump impossible, and the seven-night retention, because a quiet bug in any of those is only found
on the morning a restore is needed.
"""

import os
import re
import stat
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "vps" / "backup.sh"
# Absolute, so the test runs the same shell the VPS's unit does (/bin/sh is dash on Debian).
SHELL = "/bin/sh"
# `date -u +%Y%m%dT%H%M%SZ`: sortable and unambiguous, and it names the morning it can restore.
NAME = re.compile(r"paperwatch-\d{8}T\d{6}Z\.dump")


def _stub_pg_dump(directory: Path, *, writes: bool = True) -> None:
    """A `pg_dump` first on PATH that honours `--file=` and can succeed without writing."""
    lines = [
        "#!/bin/sh",
        'for arg in "$@"; do case $arg in --file=*) out=${arg#--file=} ;; esac; done',
    ]
    if writes:
        lines.append("printf 'PGDMP{}' > \"$out\"")
    stub = directory / "pg_dump"
    stub.write_text("\n".join(lines) + "\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, backups: Path, *, writes: bool = True) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    _stub_pg_dump(stub_dir, writes=writes)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PW_BACKUP_DIR": str(backups),
    }
    return subprocess.run(  # noqa: S603 (a fixed script path, no shell)
        [SHELL, str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60, check=False
    )


def _dumps(backups: Path) -> list[str]:
    return sorted(p.name for p in backups.glob("paperwatch-*.dump"))


def test_backup_writes_one_new_dump_and_no_temporary_file(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    result = _run(tmp_path, backups)
    assert result.returncode == 0, result.stdout + result.stderr
    [dump] = _dumps(backups)
    assert NAME.fullmatch(dump), dump
    assert (backups / dump).read_text() == "PGDMP{}"
    assert not list(backups.glob("*.part")), "a renamed dump must leave no partial file behind"
    assert dump in result.stdout


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
