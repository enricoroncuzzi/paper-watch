"""`deploy/vps/move-api-account.sh`: the root-run migration in `docs/deploy/vps.md` section 18.

The script only ever runs as root on the live server, so what a test can own is the logic it
decides: which commands, in which order, against which file, and every refusal. Each test runs the
real script against a temporary tree (`PW_MOVE_ROOT`) with the privileged commands stubbed on
`PATH` and their arguments recorded, so the assertions are about what the script asked the system
to do and about the files it wrote, never about the script's own text.
"""

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from paperwatch.config import REPO_ROOT

SCRIPT = REPO_ROOT / "deploy" / "vps" / "move-api-account.sh"
GUIDE = REPO_ROOT / "docs" / "deploy" / "vps.md"
BASH = shutil.which("bash") or "/bin/bash"
REAL_CURL = shutil.which("curl") or "/usr/bin/curl"
REAL_INSTALL = shutil.which("install") or "/usr/bin/install"
REAL_GIT = shutil.which("git") or "/usr/bin/git"
REAL_PATH = os.environ["PATH"]

OWNER = "paperwatch"
API = "paperwatch-api"
TAG = "paperwatch-v9.9.9"

# The owner's env file, as the server has it: the API's least-privilege role, the chat key, and the
# ingestion's model cache. The two secret values are fake and are only here so a test can prove the
# script never prints them.
OWNER_ENV = """\
PW_DATABASE_URL=postgresql+psycopg://paperwatch_api@/paperwatch?host=/var/run/postgresql
PW_OPENROUTER_API_KEY=fake-openrouter-key-for-the-test
PW_LOCAL_EMBEDDING_CACHE_DIR=/home/paperwatch/.cache/paperwatch-models
PW_API_AUTH=hmac
PW_API_HMAC_SECRET=fake-hmac-secret-for-the-test
PW_API_PUBLIC_HOST=paperwatch.example.ts.net
"""
RETIRED_ENV = "PW_API_HMAC_SECRET=fake-retired-secret-for-the-test\n"
SECRETS = ("fake-openrouter-key-for-the-test", "fake-hmac-secret-for-the-test")
API_ROLE = "paperwatch_api"

PG_IDENT = """\
# MAPNAME     SYSTEM-USERNAME  PG-USERNAME
paperwatch    paperwatch       paperwatch
paperwatch    paperwatch       paperwatch_ro
paperwatch    paperwatch       paperwatch_api
"""

UNIT_FROM_CHECKOUT = "# the unit this release installs\n"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 (a fixed binary and argv, no shell)
        [REAL_GIT, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
    )
    return done.stdout


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body).lstrip())
    path.chmod(0o755)


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under the tree with its content and mode, so a no-op can be proven.

    `.git` is left out: a second `git fetch` legitimately rewrites its own internals, and what
    matters here is the checkout, the env files and pg_ident.conf.
    """
    seen: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seen[str(path.relative_to(root))] = f"{oct(path.stat().st_mode & 0o777)} {digest}"
    return seen


class _Health(BaseHTTPRequestHandler):
    """The one route readiness polls, answering as the real API does."""

    def do_GET(self) -> None:  # http.server's spelling, not ours
        body, status = (b'{"status":"ok"}', 200) if self.path == "/healthz" else (b"{}", 404)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@dataclass
class Sandbox:
    """A temporary tree standing in for the server, plus the stubs of everything root needs."""

    root: Path
    bin: Path
    rec: Path
    repo: Path
    api_url: str

    # -- assertions ---------------------------------------------------------------
    @property
    def api_home(self) -> Path:
        return self.path(f"home/{API}")

    @property
    def api_checkout(self) -> Path:
        return self.api_home / "paper-watch"

    @property
    def api_env(self) -> Path:
        return self.api_home / ".config" / "paperwatch" / "api.env"

    @property
    def owner_env(self) -> Path:
        return self.path(f"home/{OWNER}/.config/paperwatch/api.env")

    @property
    def retired_env(self) -> Path:
        return self.path(f"home/{OWNER}/.config/paperwatch/api.env.bak-hmac")

    @property
    def owner_unit(self) -> Path:
        return self.path(f"home/{OWNER}/.config/systemd/user/paperwatch-api.service")

    @property
    def system_unit(self) -> Path:
        return self.path("etc/systemd/system/paperwatch-api.service")

    @property
    def pg_ident(self) -> Path:
        return self.path("etc/postgresql/17/main/pg_ident.conf")

    @property
    def owner_cache(self) -> str:
        return str(self.path(f"home/{OWNER}/.cache/paperwatch-models"))

    @property
    def api_cache(self) -> str:
        return str(self.api_home / ".cache" / "paperwatch-models")

    def path(self, relative: str) -> Path:
        return self.root / relative

    def records(self, name: str) -> list[str]:
        log = self.rec / name
        return log.read_text().splitlines() if log.exists() else []

    def map_lines(self) -> list[list[str]]:
        """The map's own lines, the comment header aside."""
        lines = [line for line in self.pg_ident.read_text().splitlines() if line.strip()]
        return [line.split() for line in lines if not line.lstrip().startswith("#")]

    def owner_map_lines(self) -> list[list[str]]:
        """The owner's own path to the API's role, which tidy removes last."""
        return [
            line for line in self.map_lines() if line[1:2] == [OWNER] and line[2:] == [API_ROLE]
        ]

    # -- running ------------------------------------------------------------------
    def env(self, **extra: str) -> dict[str, str]:
        return {
            "PATH": f"{self.bin}:{REAL_PATH}",
            "HOME": str(self.root),
            "REC": str(self.rec),
            "PW_MOVE_ROOT": str(self.root),
            # The tree has no such Unix accounts and no root, so the script runs its real logic
            # with every privileged command stubbed and its root check skipped.
            "PW_MOVE_UNPRIVILEGED": "1",
            "PW_MOVE_REPO_URL": str(self.repo),
            "PW_MOVE_API_URL": self.api_url,
            "PW_MOVE_API_CHECK": str(self.rec / "api-check.sh"),
            "PW_MOVE_READY_TIMEOUT": "5",
            "PW_MOVE_POLL": "1",
            **extra,
        }

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 (the repository's own script, no shell)
            [BASH, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=self.env(**extra),
        )

    def prepare(self) -> None:
        result = self.run("prepare")
        assert result.returncode == 0, result.stdout + result.stderr

    def accept(self, *, passes: bool = True) -> None:
        """Point the script at an api-check that passes or fails, as section 15.4 would."""
        body = "#!/bin/sh\nprintf 'all checks passed\\n'\nexit 0\n"
        if not passes:
            body = "#!/bin/sh\nprintf 'FAIL: healthz answers\\n' >&2\nexit 1\n"
        check = self.rec / "api-check.sh"
        check.write_text(body)
        check.chmod(0o755)


def _build_stubs(sandbox: Sandbox) -> None:
    rec = sandbox.rec
    home = sandbox.path(f"home/{OWNER}")
    (rec / "passwd").write_text(f"{OWNER}:x:1001:1001::{home}:/bin/bash\n")
    # The installer the release's uv install would run: enough to put a working `uv` in the
    # account's home, and no network.
    (rec / "installer.sh").write_text(
        "#!/bin/sh\n"
        'mkdir -p "$HOME/.local/bin"\n'
        "cat > \"$HOME/.local/bin/uv\" <<'UV'\n"
        "#!/bin/sh\n"
        'printf \'%s\\n\' "uv $*" >> "$REC/uv"\n'
        "mkdir -p .venv\n"
        "exit 0\n"
        "UV\n"
        'chmod 755 "$HOME/.local/bin/uv"\n'
    )
    _stub(
        sandbox.bin,
        "getent",
        f"""
        [ "$1" = passwd ] || exit 2
        grep "^$2:" {rec}/passwd
        """,
    )
    _stub(
        sandbox.bin,
        "id",
        """
        case $* in
            "-u") printf '501\\n' ;;
            "-u "*) getent passwd "$2" >/dev/null || exit 1; getent passwd "$2" | cut -d: -f3 ;;
            *) exit 1 ;;
        esac
        """,
    )
    _stub(
        sandbox.bin,
        "useradd",
        f"""
        printf '%s\\n' "useradd $*" >> {rec}/useradd
        home=
        name=
        prev=
        for arg do
            if [ "$prev" = "--home-dir" ]; then home=$arg; fi
            name=$arg
            prev=$arg
        done
        mkdir -p "$home"
        printf '%s:x:1002:1002::%s:/usr/sbin/nologin\\n' "$name" "$home" >> {rec}/passwd
        """,
    )
    # runuser is the one stub that must really run what it is given: the proofs, the checkout and
    # the acceptance check are the script's own commands, and they are stubs of their own.
    _stub(
        sandbox.bin,
        "runuser",
        """
        [ "$1" = "-u" ] || exit 2
        MOVE_AS=$2
        export MOVE_AS
        shift 2
        [ "${1:-}" = "--" ] && shift
        exec "$@"
        """,
    )
    _stub(
        sandbox.bin,
        "psql",
        f"""
        printf '%s\\n' "$MOVE_AS psql $*" >> {rec}/psql
        case $* in
            *postgresql://*)
                if [ -f {rec}/psql.role-reachable ]; then printf 'paperwatch_api\\n'; exit 0; fi
                printf 'psql: error: connection failed\\n' >&2
                exit 1
                ;;
            *current_user*)
                if [ -f "{rec}/current_user.$MOVE_AS" ]; then
                    cat "{rec}/current_user.$MOVE_AS"
                elif [ "$MOVE_AS" = {API} ]; then
                    printf 'paperwatch_api\\n'
                else
                    printf 'paperwatch\\n'
                fi
                ;;
            *pg_reload_conf*) printf 't\\n' ;;
            *'select 1'*) printf '1\\n' ;;
        esac
        exit 0
        """,
    )
    _stub(
        sandbox.bin,
        "systemctl",
        f"""
        printf '%s\\n' "$*" >> {rec}/systemctl
        printf '%s|%s|%s\\n' "$MOVE_AS" "${{XDG_RUNTIME_DIR:-}}" "$*" >> {rec}/systemctl.who
        case $* in
            *--user*) state={rec}/user.active ;;
            *) state={rec}/system.active ;;
        esac
        case $* in
            *is-active*) [ -f "$state" ] && exit 0; exit 3 ;;
            *"enable --now paperwatch-api"*) : > "$state" ;;
            *"disable --now paperwatch-api"*) rm -f "$state" ;;
            *--property=User*) printf 'paperwatch-api\\n' ;;
        esac
        exit 0
        """,
    )
    _stub(
        sandbox.bin,
        "journalctl",
        """
        printf 'paperwatch-api.service: Failed at step NAMESPACE (fake journal line)\\n'
        exit 0
        """,
    )
    _stub(sandbox.bin, "sleep", "exit 0\n")
    # `install -o/-g` names the server's accounts, which this tree does not have: the copy and the
    # mode stay real, and the ownership is recorded instead.
    (sandbox.bin / "install").write_text(
        f"""#!{sys.executable}
import os
import subprocess
import sys

args = sys.argv[1:]
kept: list[str] = []
owners: list[str] = []
index = 0
while index < len(args):
    if args[index] in ("-o", "-g"):
        owners.append(args[index] + " " + args[index + 1])
        index += 2
        continue
    kept.append(args[index])
    index += 1
with open(os.path.join(os.environ["REC"], "install"), "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
sys.exit(subprocess.run(["{REAL_INSTALL}", *kept], check=False).returncode)
"""
    )
    _stub(
        sandbox.bin,
        "curl",
        f"""
        case $* in
            *astral.sh*) cat {rec}/installer.sh ;;
            *) exec {REAL_CURL} "$@" ;;
        esac
        """,
    )
    sandbox.bin.joinpath("install").chmod(0o755)


def _sandbox(tmp_path: Path, api_url: str) -> Sandbox:
    root = tmp_path / "server"
    bin_dir = tmp_path / "bin"
    rec = tmp_path / "records"
    repo = tmp_path / "release"
    for directory in (bin_dir, rec, root / "etc/postgresql/17/main", root / "etc/systemd/system"):
        directory.mkdir(parents=True)
    (root / "home" / OWNER / ".config" / "paperwatch").mkdir(parents=True)
    (root / "home" / OWNER / ".config" / "systemd" / "user").mkdir(parents=True)

    repo.mkdir()
    (repo / "deploy" / "vps").mkdir(parents=True)
    (repo / "deploy" / "vps" / "paperwatch-api.service").write_text(UNIT_FROM_CHECKOUT)
    (repo / "pyproject.toml").write_text("[project]\nname = 'paperwatch'\n")
    _git(repo, "init", "-q", "--initial-branch=main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "the release")
    _git(repo, "tag", TAG)

    ingestion = root / "home" / OWNER / "paper-watch"
    shutil.copytree(repo, ingestion, ignore=shutil.ignore_patterns(".git"))
    _git(ingestion, "init", "-q", "--initial-branch=main")
    _git(ingestion, "add", "-A")
    _git(ingestion, "commit", "-q", "-m", "the release")
    _git(ingestion, "tag", TAG)

    (root / "home" / OWNER / ".config" / "paperwatch" / "api.env").write_text(OWNER_ENV)
    retired = root / "home" / OWNER / ".config" / "paperwatch" / "api.env.bak-hmac"
    retired.write_text(RETIRED_ENV)
    retired.chmod(0o600)
    (root / "home" / OWNER / ".config" / "systemd" / "user" / "paperwatch-api.service").write_text(
        "# the old user unit\n"
    )
    (root / "etc" / "postgresql" / "17" / "main" / "pg_ident.conf").write_text(PG_IDENT)
    for directory in (
        root / "home" / OWNER / ".config" / "paperwatch",
        root / "home" / OWNER / ".config" / "systemd" / "user",
        root / "etc" / "postgresql" / "17" / "main",
    ):
        for child in directory.iterdir():
            child.chmod(0o600)

    sandbox = Sandbox(root=root, bin=bin_dir, rec=rec, repo=repo, api_url=api_url)
    _build_stubs(sandbox)
    sandbox.accept()
    return sandbox


@pytest.fixture
def api() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def nowhere() -> str:
    """An address nothing answers on, as a unit that failed to start leaves the port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def sandbox(tmp_path: Path, api: str) -> Sandbox:
    return _sandbox(tmp_path, api)


@pytest.fixture
def dead_api(tmp_path: Path, nowhere: str) -> Sandbox:
    return _sandbox(tmp_path, nowhere)


# --- the command line ----------------------------------------------------------------------


def test_running_without_a_subcommand_prints_the_usage(sandbox: Sandbox) -> None:
    result = sandbox.run()
    assert result.returncode == 2
    assert "usage:" in result.stderr
    for name in ("prepare", "switch", "rollback", "tidy", "status"):
        assert name in result.stderr


def test_an_unknown_subcommand_exits_2_with_its_name(sandbox: Sandbox) -> None:
    result = sandbox.run("preprare")
    assert result.returncode == 2
    assert "preprare" in result.stderr
    assert "usage:" in result.stderr


def test_help_is_not_an_error(sandbox: Sandbox) -> None:
    result = sandbox.run("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout


# --- status --------------------------------------------------------------------------------


def test_status_before_prepare_reports_every_step_missing(sandbox: Sandbox) -> None:
    result = sandbox.run("status")
    assert result.returncode == 0, result.stderr
    assert f"step 1 account {API}: missing" in result.stdout
    assert "step 2 map line: missing" in result.stdout
    assert "step 3 checkout: missing" in result.stdout
    assert "step 4 env file: missing" in result.stdout
    assert "step 5 system unit: missing" in result.stdout
    assert "verdict: nothing done" in result.stdout


# --- prepare (steps 1 to 4) ----------------------------------------------------------------


def test_prepare_creates_the_account_and_maps_it(sandbox: Sandbox) -> None:
    result = sandbox.run("prepare")
    assert result.returncode == 0, result.stdout + result.stderr
    [created] = sandbox.records("useradd")
    assert created.startswith(f"useradd --system --create-home --home-dir {sandbox.api_home} ")
    assert f"{API}:" in (sandbox.rec / "passwd").read_text()
    # One line, in the columns of the block section 12 shows, and the owner's own path to the
    # API's role is still there for the rollback.
    assert sandbox.pg_ident.read_text().splitlines()[-1] == (f"{OWNER}    {API}   {API_ROLE}")
    assert sandbox.owner_map_lines() == [[OWNER, OWNER, API_ROLE]]


def test_prepare_is_idempotent(sandbox: Sandbox) -> None:
    sandbox.prepare()
    before = _snapshot(sandbox.root)
    result = sandbox.run("prepare")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _snapshot(sandbox.root) == before
    assert len(sandbox.owner_map_lines()) == 1
    assert len(sandbox.records("useradd")) == 1


def test_prepare_moves_the_env_file_and_rewrites_only_the_cache_dir(sandbox: Sandbox) -> None:
    sandbox.prepare()
    assert oct(sandbox.api_env.stat().st_mode & 0o777) == "0o600"
    moved = sandbox.api_env.read_text().splitlines()
    original = OWNER_ENV.splitlines()
    assert len(moved) == len(original)
    for before, after in zip(original, moved, strict=True):
        if before.startswith("PW_LOCAL_EMBEDDING_CACHE_DIR="):
            assert after == f"PW_LOCAL_EMBEDDING_CACHE_DIR={sandbox.api_cache}"
        else:
            assert after == before
    # The owner's copy and its retired backup wait for tidy, so a rollback has nothing to copy.
    assert sandbox.owner_env.read_text() == OWNER_ENV
    assert sandbox.retired_env.exists()


def test_prepare_never_prints_a_secret(sandbox: Sandbox) -> None:
    result = sandbox.run("prepare")
    assert result.returncode == 0, result.stderr
    for secret in SECRETS:
        assert secret not in result.stdout + result.stderr


def test_prepare_stops_when_the_new_account_maps_to_the_owner_role(sandbox: Sandbox) -> None:
    # The proofs of section 18's step 2 exist for this: peer authentication fell through to the
    # generic rule, and the account must not be trusted with a checkout before it is fixed.
    (sandbox.rec / f"current_user.{API}").write_text("paperwatch\n")
    result = sandbox.run("prepare")
    assert result.returncode == 1
    assert "paperwatch_api" in result.stderr and "map" in result.stderr
    assert not sandbox.api_checkout.exists()


def test_prepare_uses_the_release_tag_the_ingestion_serves(sandbox: Sandbox) -> None:
    sandbox.prepare()
    assert _git(sandbox.api_checkout, "describe", "--tags", "--exact-match").strip() == TAG
    unit = sandbox.api_checkout / "deploy" / "vps" / "paperwatch-api.service"
    assert unit.read_text() == UNIT_FROM_CHECKOUT
    # The environment is the committed lock and nothing else, as section 3 installs it.
    assert "uv sync --frozen --no-dev" in sandbox.records("uv")


def test_prepare_refuses_when_the_ingestion_checkout_is_not_on_a_release_tag(
    tmp_path: Path, api: str
) -> None:
    sandbox = _sandbox(tmp_path, api)
    _git(sandbox.path(f"home/{OWNER}/paper-watch"), "tag", "-d", TAG)
    result = sandbox.run("prepare")
    assert result.returncode == 1
    assert "release tag" in result.stderr
    assert not sandbox.api_checkout.exists()


def test_prepare_leaves_the_running_api_alone(sandbox: Sandbox) -> None:
    sandbox.prepare()
    assert sandbox.records("systemctl") == []
    assert not (sandbox.rec / "system.active").exists()


def test_status_reads_the_tree_and_changes_nothing(sandbox: Sandbox) -> None:
    sandbox.prepare()
    before = _snapshot(sandbox.root)
    result = sandbox.run("status")
    assert result.returncode == 0, result.stderr
    assert f"step 1 account {API}: done" in result.stdout
    assert f"step 3 checkout: done {TAG}" in result.stdout
    assert "verdict: prepared, ready to switch" in result.stdout
    assert _snapshot(sandbox.root) == before


def test_status_never_prints_a_secret(sandbox: Sandbox) -> None:
    sandbox.prepare()
    result = sandbox.run("status")
    for secret in SECRETS:
        assert secret not in result.stdout + result.stderr


# --- switch (steps 5 and 6) ----------------------------------------------------------------


def test_switch_refuses_before_prepare(sandbox: Sandbox) -> None:
    result = sandbox.run("switch", "--yes")
    assert result.returncode == 1
    assert "prepare" in result.stderr
    assert sandbox.records("systemctl") == []
    assert not sandbox.system_unit.exists()


def test_switch_installs_the_unit_and_never_holds_the_port_twice(sandbox: Sandbox) -> None:
    sandbox.prepare()
    result = sandbox.run("switch", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sandbox.system_unit.read_text() == UNIT_FROM_CHECKOUT
    calls = sandbox.records("systemctl")
    stop = calls.index("--user disable --now paperwatch-api")
    start = calls.index("enable --now paperwatch-api")
    assert calls.index("daemon-reload") < stop < start
    # Everything a rollback needs is still there: the way back is not given up until tidy.
    assert sandbox.owner_unit.exists()
    assert sandbox.owner_env.exists()
    # The old unit is the owner's user manager's, reached as that account: as root it has no
    # runtime directory of its own to ask.
    assert f"{OWNER}|/run/user/1001|--user disable --now paperwatch-api" in sandbox.records(
        "systemctl.who"
    )


def test_switch_runs_the_acceptance_check_before_it_reports_success(sandbox: Sandbox) -> None:
    sandbox.prepare()
    result = sandbox.run("switch", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks passed" in result.stdout
    assert "tidy" in result.stdout


def test_switch_rolls_back_when_the_new_unit_never_becomes_healthy(dead_api: Sandbox) -> None:
    dead_api.prepare()
    result = dead_api.run("switch", "--yes")
    assert result.returncode == 1
    calls = dead_api.records("systemctl")
    assert "disable --now paperwatch-api" in calls
    assert calls.index("disable --now paperwatch-api") < calls.index(
        "--user enable --now paperwatch-api"
    )
    # The evidence is in the failure itself: the operator is looking at one terminal.
    assert "fake journal line" in result.stderr
    assert not (dead_api.rec / "system.active").exists()
    assert (dead_api.rec / "user.active").exists()


def test_switch_rolls_back_when_the_acceptance_check_fails(sandbox: Sandbox) -> None:
    sandbox.accept(passes=False)
    sandbox.prepare()
    result = sandbox.run("switch", "--yes")
    assert result.returncode == 1
    assert "FAIL: healthz answers" in result.stdout + result.stderr
    assert not (sandbox.rec / "system.active").exists()
    assert (sandbox.rec / "user.active").exists()


def test_switch_asks_before_it_swaps_the_units(sandbox: Sandbox) -> None:
    sandbox.prepare()
    result = subprocess.run(  # noqa: S603 (the repository's own script, no shell)
        [BASH, str(SCRIPT), "switch"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
        env=sandbox.env(),
    )
    assert result.returncode == 1
    assert "--yes" in result.stderr
    assert sandbox.records("systemctl") == []


# --- rollback ------------------------------------------------------------------------------


def test_rollback_stops_the_system_unit_before_starting_the_old_one(sandbox: Sandbox) -> None:
    sandbox.prepare()
    assert sandbox.run("switch", "--yes").returncode == 0
    (sandbox.rec / "systemctl").unlink()
    result = sandbox.run("rollback", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = sandbox.records("systemctl")
    assert calls.index("disable --now paperwatch-api") < calls.index(
        "--user enable --now paperwatch-api"
    )
    assert (sandbox.rec / "user.active").exists()
    assert not (sandbox.rec / "system.active").exists()
    # The unit the rollback starts is the one the migration kept, and it is still there.
    assert sandbox.owner_unit.exists()


def test_rollback_leaves_the_owners_env_file_exactly_as_prepare_left_it(sandbox: Sandbox) -> None:
    sandbox.prepare()
    assert sandbox.run("switch", "--yes").returncode == 0
    result = sandbox.run("rollback", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sandbox.owner_env.read_text() == OWNER_ENV


def test_rollback_restores_the_owners_env_file_when_it_is_gone(sandbox: Sandbox) -> None:
    # The guide's step 4 removes the owner's copy by hand. When it is gone, the rollback puts it
    # back 0600 with the ingestion's own model cache, not the API's.
    sandbox.prepare()
    assert sandbox.run("switch", "--yes").returncode == 0
    sandbox.owner_env.unlink()
    result = sandbox.run("rollback", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert oct(sandbox.owner_env.stat().st_mode & 0o777) == "0o600"
    restored = sandbox.owner_env.read_text().splitlines()
    assert f"PW_LOCAL_EMBEDDING_CACHE_DIR={sandbox.owner_cache}" in restored
    assert len(restored) == len(OWNER_ENV.splitlines())


def test_rollback_refuses_once_tidy_has_removed_the_old_unit(sandbox: Sandbox) -> None:
    # After step 7 the way back needs the owner's ident line and a unit file from a release before
    # ADR 0017, which is not something this script can put back on its own.
    sandbox.prepare()
    assert sandbox.run("switch", "--yes").returncode == 0
    sandbox.owner_unit.unlink()
    result = sandbox.run("rollback", "--yes")
    assert result.returncode == 1
    assert "tidy already ran" in result.stderr


# --- tidy (step 7) -------------------------------------------------------------------------


def _switched(sandbox: Sandbox) -> None:
    sandbox.prepare()
    result = sandbox.run("switch", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr


def test_tidy_refuses_unless_the_system_unit_is_active(sandbox: Sandbox) -> None:
    # Prepared but not switched: retiring the owner's way in would leave the API on an account
    # that may not be serving at all.
    sandbox.prepare()
    result = sandbox.run("tidy", "--yes")
    assert result.returncode == 1
    assert "not active" in result.stderr
    assert sandbox.owner_env.exists()
    assert sandbox.owner_map_lines() == [[OWNER, OWNER, API_ROLE]]


def test_tidy_refuses_when_the_acceptance_check_fails(sandbox: Sandbox) -> None:
    _switched(sandbox)
    sandbox.accept(passes=False)
    result = sandbox.run("tidy", "--yes")
    assert result.returncode == 1
    assert "acceptance check" in result.stderr
    assert sandbox.owner_env.exists()
    assert sandbox.owner_map_lines() == [[OWNER, OWNER, API_ROLE]]


def test_tidy_removes_the_owners_way_in_and_the_old_files(sandbox: Sandbox) -> None:
    _switched(sandbox)
    result = sandbox.run("tidy", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    # The map is now the block section 12 ends with: the owner's two lines and the API's one.
    assert sandbox.map_lines() == [
        [OWNER, OWNER, OWNER],
        [OWNER, OWNER, "paperwatch_ro"],
        [OWNER, API, API_ROLE],
    ]
    # The refusal is proven before the files go: the owner asking for the API's role is refused.
    assert f"{OWNER} psql postgresql://{API_ROLE}@" in (sandbox.rec / "psql").read_text()
    assert not sandbox.owner_env.exists()
    assert not sandbox.retired_env.exists()
    assert not sandbox.owner_unit.exists()
    assert "--user daemon-reload" in sandbox.records("systemctl")


def test_tidy_leaves_every_file_when_the_owner_can_still_take_the_role(sandbox: Sandbox) -> None:
    # If the owner still reaches paperwatch_api with its line gone, something else lets it in, and
    # deleting the rest would hide that instead of closing it.
    _switched(sandbox)
    (sandbox.rec / "psql.role-reachable").write_text("")
    result = sandbox.run("tidy", "--yes")
    assert result.returncode == 1
    assert API_ROLE in result.stderr
    assert sandbox.owner_env.exists()
    assert sandbox.retired_env.exists()
    assert sandbox.owner_unit.exists()
    # The line it removed is put back, so the next run starts from the same state.
    assert sandbox.owner_map_lines() == [[OWNER, OWNER, API_ROLE]]


# --- the guide -----------------------------------------------------------------------------


def test_the_guide_points_at_the_script_for_the_migration() -> None:
    # Section 18 is what the owner reads on the server, so it names the script and each subcommand
    # the migration runs; the numbered steps stay as the reference for what it does and why.
    guide = GUIDE.read_text()
    section = guide[guide.index("## 18. Moving an existing server onto the API account") :]
    assert "deploy/vps/move-api-account.sh" in section
    for subcommand in ("status", "prepare", "switch", "tidy"):
        assert f"move-api-account.sh {subcommand}" in section
    assert SCRIPT.is_file()
    # The guide runs it by absolute path, from a checkout that git keeps executable.
    assert SCRIPT.stat().st_mode & 0o111


@pytest.mark.skipif(os.geteuid() == 0, reason="the check only fires for a non-root caller")
def test_a_subcommand_refuses_to_run_without_root(sandbox: Sandbox) -> None:
    # The account, the system unit and pg_ident.conf are root's, so the refusal has to come before
    # any of them is touched. The tests skip this check the rest of the time, so it is tested here.
    result = subprocess.run(  # noqa: S603 (the repository's own script, no shell)
        [BASH, str(SCRIPT), "prepare"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=sandbox.env(PW_MOVE_UNPRIVILEGED="0"),
    )
    assert result.returncode == 1
    assert "run as root" in result.stderr
    assert sandbox.records("useradd") == []
