"""`deploy/vps/update-api.sh`: the one root command the routine API update runs.

The script only ever runs as root on the live server, so what a test owns is the logic it decides:
which argument it accepts, which commands it runs for which account and in which order, what it
refuses, and what it puts back when a release fails. Each test runs the real script against a
temporary tree (`PW_UPDATE_ROOT`) with the privileged commands stubbed on `PATH` and their arguments
recorded.

The stubs write to their record files by absolute path rather than through an environment variable:
the script clears the environment of every command it runs as the API's account, on purpose, so a
record that depended on `REC` would prove the opposite of what it claims.
"""

import hashlib
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from paperwatch.config import REPO_ROOT

SCRIPT = REPO_ROOT / "deploy" / "vps" / "update-api.sh"
SUDOERS = REPO_ROOT / "deploy" / "vps" / "sudoers-paperwatch-update-api"
GUIDE = REPO_ROOT / "docs" / "deploy" / "vps.md"
ADR = REPO_ROOT / "docs" / "adr" / "0018-the-api-update-through-one-root-owned-script.md"
BASH = shutil.which("bash") or "/bin/bash"
REAL_GIT = shutil.which("git") or "/usr/bin/git"
REAL_PATH = os.environ["PATH"]

API = "paperwatch-api"
OLD_TAG = "paperwatch-v1.0.0"
NEW_TAG = "paperwatch-v1.1.0"
INSTALLED_PATH = "/usr/local/sbin/paperwatch-update-api"

UNIT = "# the API unit this release installs\n"
CHANGED_UNIT = "# the API unit this release changes, which only the owner may install\n"

# The API account's env file, as section 15.2 has it. The two secrets are fake and are here so a
# test can prove the script never prints them.
API_ENV = """\
PW_DATABASE_URL=postgresql+psycopg://paperwatch_api@/paperwatch?host=/var/run/postgresql
PW_OPENROUTER_API_KEY=fake-openrouter-key-for-the-test
PW_LOCAL_EMBEDDING_CACHE_DIR={cache}
PW_API_AUTH=hmac
PW_API_HMAC_SECRET=fake-hmac-secret-for-the-test
PW_API_PUBLIC_HOST=paperwatch.example.ts.net
"""
SECRETS = ("fake-openrouter-key-for-the-test", "fake-hmac-secret-for-the-test")

# The release's acceptance check, published in the release so the script runs the checkout's own
# copy. It answers as section 15.4 does, and fails only when a test asks it to through `$HOME`.
CHECK = """\
#!/bin/sh
if [ -f "$HOME/.check-fails" ]; then
    printf 'FAIL: healthz answers\\n' >&2
    exit 1
fi
if [ -n "${PW_CHECK_TOKEN_CMD:-}" ]; then
    token=$(sh -c "$PW_CHECK_TOKEN_CMD") || token=
    [ -n "$token" ] || { printf 'FAIL: no token to send\\n' >&2; exit 1; }
fi
printf 'all checks passed\\n'
"""


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

    `.git` is left out: a fetch legitimately rewrites its own internals, and what matters here is
    the checkout, the env file and the installed unit.
    """
    seen: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seen[str(path.relative_to(root))] = f"{oct(path.stat().st_mode & 0o777)} {digest}"
    return seen


@dataclass
class Sandbox:
    """A temporary tree standing in for the server, plus the stubs of everything root needs."""

    root: Path
    bin: Path
    rec: Path
    origin: Path

    # -- paths ----------------------------------------------------------------------
    def path(self, relative: str) -> Path:
        return self.root / relative

    @property
    def api_home(self) -> Path:
        return self.path(f"home/{API}")

    @property
    def checkout(self) -> Path:
        return self.api_home / "paper-watch"

    @property
    def api_env(self) -> Path:
        return self.api_home / ".config" / "paperwatch" / "api.env"

    @property
    def system_unit(self) -> Path:
        return self.path("etc/systemd/system/paperwatch-api.service")

    # -- assertions -----------------------------------------------------------------
    def records(self, name: str) -> list[str]:
        log = self.rec / name
        return log.read_text().splitlines() if log.exists() else []

    def checkout_tag(self) -> str:
        return _git(self.checkout, "describe", "--tags", "--exact-match").strip()

    def accept(self, *, passes: bool = True) -> None:
        marker = self.api_home / ".check-fails"
        if passes:
            marker.unlink(missing_ok=True)
        else:
            marker.write_text("the new release fails its acceptance check\n")

    def healthy(self, *, answers: bool = True) -> None:
        marker = self.rec / "healthy"
        if answers:
            marker.write_text("")
        else:
            marker.unlink(missing_ok=True)

    # -- running --------------------------------------------------------------------
    def env(self, **extra: str) -> dict[str, str]:
        return {
            "PATH": f"{self.bin}:{REAL_PATH}",
            "HOME": str(self.root),
            # The tree has no root and no such Unix accounts, so the script runs its real logic
            # with every privileged command stubbed and its root check skipped.
            "PW_UPDATE_UNPRIVILEGED": "1",
            "PW_UPDATE_ROOT": str(self.root),
            "PW_UPDATE_API_URL": "http://127.0.0.1:8000",
            "PW_UPDATE_READY_TIMEOUT": "4",
            "PW_UPDATE_POLL": "1",
            **extra,
        }

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 (the repository's own script, no shell)
            [BASH, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            # Root's own working directory, which is no project of the API's.
            cwd=self.root,
            env=self.env(**extra),
        )


def _build_release(release: Path, new_unit: str) -> None:
    """The origin: two tagged releases, `paperwatch-v1.0.0` and `paperwatch-v1.1.0`."""
    (release / "deploy" / "vps").mkdir(parents=True)
    (release / "pyproject.toml").write_text("[project]\nname = 'paperwatch'\nversion = '1.0.0'\n")
    (release / "deploy" / "vps" / "paperwatch-api.service").write_text(UNIT)
    check = release / "deploy" / "vps" / "api-check.sh"
    check.write_text(CHECK)
    check.chmod(0o755)
    _git(release, "init", "-q", "--initial-branch=main")
    _git(release, "add", "-A")
    _git(release, "commit", "-q", "-m", "release 1.0.0")
    _git(release, "tag", OLD_TAG)

    (release / "pyproject.toml").write_text("[project]\nname = 'paperwatch'\nversion = '1.1.0'\n")
    (release / "deploy" / "vps" / "paperwatch-api.service").write_text(new_unit)
    _git(release, "add", "-A")
    _git(release, "commit", "-q", "-m", "release 1.1.0")
    _git(release, "tag", NEW_TAG)


def _build_stubs(sandbox: Sandbox) -> None:
    rec = sandbox.rec
    bin_dir = sandbox.bin
    _stub(
        bin_dir,
        "id",
        """
        [ "$1" = "-u" ] || exit 2
        printf '501\\n'
        """,
    )
    # runuser is the one stub that must really run what it is given: the checkout, the venv and the
    # acceptance check are the script's own commands, and they are stubs of their own.
    _stub(
        bin_dir,
        "runuser",
        f"""
        [ "$1" = "-u" ] || exit 2
        printf '%s|%s\\n' "$2" "$*" >> {rec}/runuser
        shift 2
        [ "${{1:-}}" = "--" ] && shift
        exec "$@"
        """,
    )
    # git runs for real, as the API account, against the clone's own origin.
    _stub(
        bin_dir,
        "git",
        f"""
        printf '%s\\n' "$*" >> {rec}/git
        exec {REAL_GIT} "$@"
        """,
    )
    _stub(
        bin_dir,
        "systemctl",
        f"""
        printf '%s\\n' "$*" >> {rec}/systemctl
        case $* in
            *is-active*) [ -f {rec}/active ] && exit 0; exit 3 ;;
            *status*) printf 'paperwatch-api.service: fake unit state\\n' ;;
            *restart*)
                if [ -f {rec}/restart.fails ]; then
                    rm -f {rec}/restart.fails
                    exit 1
                fi
                : > {rec}/active
                ;;
        esac
        exit 0
        """,
    )
    _stub(
        bin_dir,
        "journalctl",
        """
        printf 'paperwatch-api.service: Failed at step NAMESPACE (fake journal line)\\n'
        exit 0
        """,
    )
    _stub(
        bin_dir,
        "curl",
        f"""
        [ -f {rec}/healthy ] && exit 0
        # A release that does not come up leaves the port silent until the rollback puts the
        # previous one back, which is what `previous.answers` models.
        if [ -f {rec}/previous.answers ]; then
            tag=$(git -C {sandbox.checkout} describe --tags --exact-match 2>/dev/null)
            [ "$tag" = {OLD_TAG} ] && exit 0
        fi
        exit 22
        """,
    )
    _stub(bin_dir, "sleep", "exit 0\n")
    # uv is the API account's own binary, so the fixture writes it into that account's home.
    uv = sandbox.api_home / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {rec}/uv\n"
        "if [ ! -f pyproject.toml ]; then\n"
        "    printf 'error: no project found in %s\\n' \"$PWD\" >&2\n"
        "    exit 2\n"
        "fi\n"
        "case $* in\n"
        "    *api-token*) printf 'fake-minted-token\\n' ;;\n"
        "    *) mkdir -p .venv ;;\n"
        "esac\n"
        "exit 0\n"
    )
    uv.chmod(0o755)


def _sandbox(tmp_path: Path, *, new_unit: str = UNIT) -> Sandbox:
    root = tmp_path / "server"
    bin_dir = tmp_path / "bin"
    rec = tmp_path / "records"
    origin = tmp_path / "release"
    for directory in (bin_dir, rec, root / "etc/systemd/system"):
        directory.mkdir(parents=True)
    _build_release(origin, new_unit)

    sandbox = Sandbox(root=root, bin=bin_dir, rec=rec, origin=origin)
    # The API serves its own clone, at the release before the one the tests update to.
    sandbox.api_home.mkdir(parents=True)
    subprocess.run(  # noqa: S603 (a fixed binary and argv, no shell)
        [REAL_GIT, "clone", "--quiet", str(origin), str(sandbox.checkout)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid"},
    )
    _git(sandbox.checkout, "checkout", "--quiet", OLD_TAG)
    (sandbox.api_home / ".config" / "paperwatch").mkdir(parents=True)
    sandbox.api_env.write_text(
        API_ENV.format(cache=sandbox.api_home / ".cache" / "paperwatch-models")
    )
    sandbox.api_env.chmod(0o600)
    sandbox.system_unit.write_text(UNIT)
    sandbox.healthy()
    _build_stubs(sandbox)
    return sandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return _sandbox(tmp_path)


# --- the command line ----------------------------------------------------------------------


def test_running_without_a_release_tag_prints_the_usage(sandbox: Sandbox) -> None:
    result = sandbox.run()
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "paperwatch-vX.Y.Z" in result.stderr
    assert sandbox.records("runuser") == []


def test_two_arguments_are_refused(sandbox: Sandbox) -> None:
    result = sandbox.run(OLD_TAG, NEW_TAG)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert sandbox.records("runuser") == []


@pytest.mark.parametrize(
    "tag",
    [
        "1.0.0",
        "v1.0.0",
        "paperwatch-1.0.0",
        "paperwatch-v1.0",
        "paperwatch-v1.0.0.1",
        "paperwatch-v1.0.0-rc1",
        "paperwatch-v1.0.0 ",
        "paperwatch-V1.0.0",
        "../paperwatch-v1.0.0",
        "paperwatch-v1.0.0/..",
        "paperwatch-v1.0.0; true",
        "paperwatch-v1.0.0$(true)",
        "",
    ],
)
def test_a_tag_that_is_not_a_release_tag_is_refused(sandbox: Sandbox, tag: str) -> None:
    # The tag is the only thing a sudo caller can pass, so this refusal is the whole argument
    # boundary: nothing runs, and nothing in the tree moves.
    before = _snapshot(sandbox.root)
    result = sandbox.run(tag)
    assert result.returncode == 2
    assert "not a release tag" in result.stderr
    assert sandbox.records("runuser") == []
    assert sandbox.records("systemctl") == []
    assert _snapshot(sandbox.root) == before


def test_help_is_not_an_error(sandbox: Sandbox) -> None:
    result = sandbox.run("--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert sandbox.records("runuser") == []


@pytest.mark.skipif(os.geteuid() == 0, reason="the check only fires for a non-root caller")
def test_it_refuses_to_run_without_root(sandbox: Sandbox) -> None:
    # The checkout and the unit are another account's and the system manager's, so the refusal has
    # to come before either is touched. The tests skip this check the rest of the time.
    result = sandbox.run(OLD_TAG, PW_UPDATE_UNPRIVILEGED="0")
    assert result.returncode == 1
    assert "run as root" in result.stderr
    assert sandbox.records("runuser") == []


# --- the preconditions ---------------------------------------------------------------------


def test_a_checkout_that_is_not_a_git_clone_is_refused(sandbox: Sandbox) -> None:
    shutil.rmtree(sandbox.checkout / ".git")
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert str(sandbox.checkout) in result.stderr
    assert sandbox.records("systemctl") == []


def test_a_missing_installed_unit_is_refused(sandbox: Sandbox) -> None:
    # Without the installed unit there is nothing to compare a release against, and restarting a
    # unit that is not installed would report a success that never served.
    sandbox.system_unit.unlink()
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert str(sandbox.system_unit) in result.stderr
    assert sandbox.records("systemctl") == []


def test_a_checkout_with_no_release_behind_it_is_refused(sandbox: Sandbox) -> None:
    # A failed update goes back to the previous release, so a checkout that cannot name one must
    # stop before it moves: otherwise a failed update has no way back.
    _git(sandbox.checkout, "checkout", "--quiet", "--orphan", "nowhere")
    _git(sandbox.checkout, "commit", "-q", "--allow-empty", "-m", "not a release")
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert "previous release" in result.stderr
    assert sandbox.records("systemctl") == []


# --- the release, its unit and the checkout ------------------------------------------------


def test_a_tag_the_origin_does_not_have_is_refused(sandbox: Sandbox) -> None:
    result = sandbox.run("paperwatch-v9.9.9")
    assert result.returncode == 1
    assert "paperwatch-v9.9.9" in result.stderr and "no release tag" in result.stderr
    assert sandbox.checkout_tag() == OLD_TAG
    assert sandbox.records("systemctl") == []


def test_a_release_whose_unit_changed_stops_before_the_checkout_moves(tmp_path: Path) -> None:
    # The comparison is against the tag's own file, not the checkout's copy: a release that changes
    # the unit must not start on the old one, because a unit change is a compatibility change
    # (memory, sandbox paths, the ExecStart line) and installing it is the owner's step.
    sandbox = _sandbox(tmp_path, new_unit=CHANGED_UNIT)
    before = _snapshot(sandbox.checkout)
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert str(sandbox.system_unit) in result.stderr
    assert "differs" in result.stderr and "11.1" in result.stderr
    assert sandbox.checkout_tag() == OLD_TAG
    assert _snapshot(sandbox.checkout) == before
    assert sandbox.records("uv") == []
    assert sandbox.records("systemctl") == []


def test_a_release_with_the_installed_unit_moves_the_checkout_and_syncs_it(
    sandbox: Sandbox,
) -> None:
    sandbox.run(NEW_TAG)
    assert sandbox.checkout_tag() == NEW_TAG
    assert "sync --frozen --no-dev" in sandbox.records("uv")
    # The sync ran in the checkout's own project, which is what the stub's pyproject.toml check
    # proves: a sync from root's directory would have failed instead.
    assert (sandbox.checkout / ".venv").is_dir()


def test_the_fetch_uses_the_clones_own_origin_and_no_url(sandbox: Sandbox) -> None:
    # The caller names a tag and nothing else: a URL, a branch or a repository from the arguments
    # or the environment would make this a way to run code from anywhere as the API's account.
    sandbox.run(NEW_TAG, PW_UPDATE_REPOSITORY="https://example.invalid/other.git")
    git_calls = sandbox.records("git")
    assert any(call.endswith("fetch --tags --quiet origin") for call in git_calls)
    assert not [call for call in git_calls if "://" in call]


def test_every_repository_step_runs_as_the_api_account(sandbox: Sandbox) -> None:
    sandbox.run(NEW_TAG)
    accounts = [line.split("|", 1)[0] for line in sandbox.records("runuser")]
    assert accounts, "the script ran no command as an account, so it did not use runuser"
    assert set(accounts) == {API}
    ran = "\n".join(sandbox.records("runuser"))
    assert "git -C" in ran and "fetch --tags --quiet origin" in ran
    assert "checkout --quiet paperwatch-v1.1.0" in ran
    assert "uv sync --frozen --no-dev" in ran


# --- the restart, the health wait, the acceptance check and the rollback --------------------


def test_a_good_release_is_restarted_and_accepted(sandbox: Sandbox) -> None:
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sandbox.checkout_tag() == NEW_TAG
    assert sandbox.records("systemctl").count("restart paperwatch-api") == 1
    assert "all checks passed" in result.stdout
    assert NEW_TAG in result.stdout


def test_the_acceptance_check_runs_from_the_checkout_as_the_api_account(sandbox: Sandbox) -> None:
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 0, result.stdout + result.stderr
    ran = "\n".join(sandbox.records("runuser"))
    assert "api-check.sh http://127.0.0.1:8000" in ran
    # The token is minted by the release's own project, as section 15.4 does, and the token itself
    # never reaches an argv or the output.
    minted = [call for call in sandbox.records("uv") if "api-token" in call]
    assert minted == [
        f"run --frozen --no-dev --env-file {sandbox.api_env} paperwatch api-token --live"
    ]
    assert "fake-minted-token" not in result.stdout + result.stderr


def test_a_release_that_never_answers_healthz_is_put_back(sandbox: Sandbox) -> None:
    # The new release does not come up; its predecessor does. The rollback is the whole point of
    # the script: the API answers again, and the command still fails so the caller knows.
    sandbox.healthy(answers=False)
    (sandbox.rec / "previous.answers").write_text("")
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert sandbox.checkout_tag() == OLD_TAG
    assert sandbox.records("systemctl").count("restart paperwatch-api") == 2
    assert [call for call in sandbox.records("uv") if call.startswith("sync")] == [
        "sync --frozen --no-dev",
        "sync --frozen --no-dev",
    ]
    assert "serves paperwatch-v1.0.0 again" in result.stderr


def test_a_failed_acceptance_check_puts_the_previous_release_back(sandbox: Sandbox) -> None:
    sandbox.accept(passes=False)
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert "FAIL: healthz answers" in result.stdout + result.stderr
    assert sandbox.checkout_tag() == OLD_TAG
    assert sandbox.records("systemctl").count("restart paperwatch-api") == 2
    assert "serves paperwatch-v1.0.0 again" in result.stderr


def test_a_failed_restart_puts_the_previous_release_back(sandbox: Sandbox) -> None:
    # One restart fails, the next one (the rollback's) succeeds: a systemd that refuses once must
    # not leave the API on code the caller was told had failed.
    (sandbox.rec / "restart.fails").write_text("")
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert sandbox.checkout_tag() == OLD_TAG
    assert sandbox.records("systemctl").count("restart paperwatch-api") == 2
    assert (sandbox.rec / "active").exists()
    assert "serves paperwatch-v1.0.0 again" in result.stderr


def test_a_rollback_that_does_not_answer_reports_the_api_as_down(sandbox: Sandbox) -> None:
    # Nothing answers, so the caller is told the API is down and gets the journal rather than a
    # cheerful summary: whoever reads that terminal has one place to look.
    sandbox.healthy(answers=False)
    result = sandbox.run(NEW_TAG)
    assert result.returncode == 1
    assert "did not finish" in result.stderr
    assert "fake journal line" in result.stderr
    assert sandbox.checkout_tag() == OLD_TAG


def test_the_release_already_served_is_synced_and_restarted(sandbox: Sandbox) -> None:
    # Rerunning after a fix is a normal update, not an error: the venv is synced and the unit
    # restarted, and the way back is the same tag.
    result = sandbox.run(OLD_TAG)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sandbox.checkout_tag() == OLD_TAG
    assert sandbox.records("systemctl").count("restart paperwatch-api") == 1
    assert "already on" in result.stdout


def test_no_run_prints_a_secret(sandbox: Sandbox) -> None:
    for outcome in (True, False):
        sandbox.accept(passes=outcome)
        result = sandbox.run(NEW_TAG)
        output = result.stdout + result.stderr
        for secret in SECRETS:
            assert secret not in output
