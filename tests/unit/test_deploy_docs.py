"""The VPS guide and the units it installs are the deployment contract, read here as one.

`docs/deploy/vps.md` and `deploy/vps/*.service`/`.slice` are the exact text a human runs and
installs on a server no test can reach, so they are an owned contract: a unit key in the wrong
section is silently ignored by systemd, and a URL the guide tells the captain to write must be the
socket form, because a `localhost:5432` one would silently try TCP, where the server's Postgres has
no password authentication and no listening port for the `paperwatch` role. Every assertion below
is about that contract's meaning - which identity reaches which role, which account a unit runs
as, which steps happen in which order - not about arbitrary text, and a drift fails the suite.

What a unit *does* when it runs is not testable here, and is not claimed: the keys are asserted in
the section systemd reads them from, and `api-check.sh` reads the deployed unit back on the server.
"""

import os
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.engine import make_url
from typer.testing import CliRunner

from paperwatch import cli
from paperwatch.config import REPO_ROOT, Settings
from paperwatch.ingest.health import Health

GUIDE = REPO_ROOT / "docs" / "deploy" / "vps.md"


def _map_block(text: str) -> list[list[str]]:
    """The pg_ident.conf block the guide shows, as the three system-user/role pairs.

    Read from the block itself, so a sentence about the mapping cannot satisfy a test that the
    mapping is right. The block is indented inside a numbered step, so the lines are stripped.
    """
    lines = text.splitlines()
    header = next(index for index, line in enumerate(lines) if "# MAPNAME" in line)
    return [line.split() for line in lines[header + 1 : header + 4]]


def _shell_commands(text: str) -> list[list[str]]:
    """Every command the text's bash blocks run, split as the shell would, comments dropped.

    A here-document's body is data, not commands, so it is skipped to its terminator.
    """
    commands: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        lines = iter(block.replace("\\\n", " ").splitlines())
        for line in lines:
            argv = shlex.split(line, comments=True)
            if not argv:
                continue
            commands.append(argv)
            heredoc = re.search(r"<<'(\w+)'", line)
            if heredoc:
                for body in lines:
                    if body.strip() == heredoc.group(1):
                        break
    return commands


def _documented_urls() -> list[str]:
    # Every place the guide shows the variable, so prose and code block cannot disagree.
    return re.findall(r"^PW_DATABASE_URL=(\S+)$", GUIDE.read_text(), re.MULTILINE)


def test_the_guide_shows_one_database_url_per_env_file() -> None:
    # The ingestion connects as the owner and the API as its own least-privilege role; the two
    # files must not silently converge on one role.
    urls = _documented_urls()
    assert urls, "docs/deploy/vps.md must show the PW_DATABASE_URL line of each env file"
    assert len(urls) == len(set(urls)) == 2, f"the guide shows conflicting URLs: {sorted(urls)}"
    assert {make_url(url).username for url in urls} == {"paperwatch", "paperwatch_api"}


def test_the_documented_database_urls_are_passwordless_socket_urls() -> None:
    for url in set(_documented_urls()):
        parsed = make_url(url)
        assert parsed.drivername == "postgresql+psycopg"
        assert parsed.password is None
        assert parsed.username in {"paperwatch", "paperwatch_api"}
        assert parsed.database == "paperwatch"
        # No TCP host and a host in the query string: psycopg reads an absolute path as the
        # directory holding the Unix socket, exactly as peer authentication expects.
        assert parsed.host is None
        assert parsed.query["host"] == "/var/run/postgresql"
        args, options = PGDialect_psycopg().create_connect_args(parsed)
        assert args == []  # positional args would mean a DSN string, not the keyword form
        assert options["host"] == "/var/run/postgresql"
        assert options["dbname"] == "paperwatch" and options["user"] == parsed.username
        assert "password" not in options


# The unit is the deployment contract for the follow-up window: it must enable the loop, and its
# timeout must stay above the loop's own window cap, or systemd would kill a healthy wait.
UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-ingest.service"


def test_the_ingest_unit_enables_follow_ups(monkeypatch: pytest.MonkeyPatch) -> None:
    # Run the unit's own arguments through the real command: the flags must parse and the loop
    # must reach the pipeline switched on.
    [line] = [line for line in UNIT.read_text().splitlines() if line.startswith("ExecStart=")]
    words = shlex.split(line.removeprefix("ExecStart="))
    args = words[words.index("paperwatch") + 1 :]
    assert args[0] == "ingest"

    seen: dict[str, object] = {}

    def stub(*_args: object, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ()))

    for name in [n for n in os.environ if n.startswith("PW_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(cli, "load_profiles", lambda path: [])
    monkeypatch.setattr(cli, "load_feeds", lambda path: [])
    monkeypatch.setattr(cli, "load_admission", lambda path: object())
    monkeypatch.setattr(cli, "make_http_client", lambda settings: object())
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: object())
    monkeypatch.setattr(cli, "run_ingestion", stub)
    result = CliRunner().invoke(cli.app, args, catch_exceptions=False)
    assert result.exit_code == 0
    assert seen["followup"] is True


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


BACKUP_UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-backup.service"


def _service_section(unit: Path) -> list[str]:
    # The [Service] lines only: systemd ignores a setting in the wrong section with a warning, so a
    # NoNewPrivileges line under [Unit] would silently vanish.
    lines = unit.read_text().splitlines()
    start = lines.index("[Service]") + 1
    section: list[str] = []
    for line in lines[start:]:
        if line.startswith("[") and line.endswith("]"):
            break
        section.append(line)
    return section


API_UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-api.service"
TUNNEL_UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-tunnel.service"
SOAK_UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-soak-stream.service"
SLICE_FILE = REPO_ROOT / "deploy" / "vps" / "paperwatch.slice"
INGEST_DROP_IN = REPO_ROOT / "deploy" / "vps" / "paperwatch-ingest.service.d" / "slice.conf"


@pytest.mark.parametrize(
    "unit",
    [UNIT, BACKUP_UNIT, API_UNIT, TUNNEL_UNIT, SOAK_UNIT],
    ids=["ingest", "backup", "api", "tunnel", "soak-stream"],
)
def test_each_unit_refuses_to_gain_privileges(unit: Path) -> None:
    # The units run code from the account's home. None needs a setuid or setgid binary, so the
    # setting that stops one from raising its own privileges costs nothing to keep.
    assert "NoNewPrivileges=yes" in _service_section(unit)


def _assert_no_unit_relationships(unit: Path) -> None:
    # The measurements showed that Conflicts= stops the ingestion, and any ordering would couple
    # the API to a run that must finish. The only shared bound is the slice (ADR 0013). The keys
    # live in [Unit], so the whole file is scanned, not only [Service].
    lines = [
        line
        for line in unit.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for key in ("Conflicts=", "Requires=", "BindsTo=", "PartOf=", "After=", "Wants="):
        assert not any(line.startswith(key) for line in lines), f"{unit.name} uses {key}"


def test_the_api_unit_never_names_the_ingestion() -> None:
    _assert_no_unit_relationships(API_UNIT)
    lines = [
        line
        for line in API_UNIT.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any("paperwatch-ingest" in line for line in lines)
    assert not any(line.startswith("MemoryHigh=") for line in lines), (
        "MemoryHigh without swap stalls (the measurements)"
    )


def test_the_tunnel_unit_never_creates_a_relationship() -> None:
    _assert_no_unit_relationships(TUNNEL_UNIT)


def test_the_api_unit_caps_memory_and_volunteers_for_the_oom_killer() -> None:
    section = _service_section(API_UNIT)
    assert "MemoryMax=700M" in section
    assert "MemorySwapMax=0" in section
    # No Slice=: the API is another Unix account now, and a systemd slice cannot span two managers.
    # The API still dies first because it volunteers to the OOM killer and the ingestion runs at
    # Nice=10; each unit's own no-swap cap bounds it absolutely (ADR 0017).
    assert not any(line.startswith("Slice=") for line in section)
    assert "OOMScoreAdjust=900" in section
    # A child OOM kill must leave the API answering: systemd's default would stop the whole unit.
    assert "OOMPolicy=continue" in section
    assert "Restart=on-failure" in section


def test_the_api_runs_as_its_own_account() -> None:
    # An RCE in the API must not be the account pg_ident maps to the owner role: no corpus writes,
    # no backups, no ingestion env file (ADR 0017).
    section = _service_section(API_UNIT)
    assert "User=paperwatch-api" in section
    assert "Group=paperwatch-api" in section
    # A system account needs no lingering user manager, so its unit is the system manager's.
    assert "WantedBy=multi-user.target" in API_UNIT.read_text().split("[Install]", 1)[1]
    [exec_line] = [line for line in section if line.startswith("ExecStart=")]
    [workdir] = [line for line in section if line.startswith("WorkingDirectory=")]
    [env_file] = [line for line in section if line.startswith("EnvironmentFile=")]
    for line in (exec_line, workdir, env_file):
        assert "paperwatch-api" in line
        # Never the old account's home, which holds the owner role's env file and the backups.
        assert not line.endswith("/home/paperwatch/paper-watch")
    assert "/home/paperwatch/" not in exec_line.replace("/home/paperwatch-api/", "")


def test_the_api_unit_is_sandboxed() -> None:
    # The cheap half of the hardening: the system tree, the other account's home and everything
    # else outside the API's own cache are read-only, and the process gets no device nodes, no
    # capabilities and no new namespaces.
    section = _service_section(API_UNIT)
    assert "ProtectSystem=strict" in section
    assert "PrivateTmp=yes" in section
    assert "PrivateDevices=yes" in section
    assert "ProtectKernelTunables=yes" in section
    assert "ProtectKernelModules=yes" in section
    assert "ProtectControlGroups=yes" in section
    assert "LockPersonality=yes" in section
    assert "CapabilityBoundingSet=" in section
    for name in ("RestrictAddressFamilies", "RestrictNamespaces", "RestrictSUIDSGID"):
        assert any(line.startswith(f"{name}=") for line in section), name
    # The model cache and the venv are the only writable places, and both are the API's own.
    writable = [line for line in section if line.startswith("ReadWritePaths=")]
    assert writable and all("paperwatch-api" in line for line in writable)
    assert not any("backups" in line or "paperwatch/.config" in line for line in writable)


def test_the_guide_documents_the_api_account_migration() -> None:
    # The server is already running, so the change is a migration, not an install: an ordered
    # procedure with a rollback, or the captain has to improvise on a live machine. The steps are
    # read from the migration section itself, since every command also appears earlier as an
    # install step.
    guide = GUIDE.read_text()
    heading = "## 18. Moving an existing server onto the API account"
    assert heading in guide
    section = guide[guide.index(heading) :]
    # One step per numbered bold heading, so a command is attributed to the step that runs it.
    steps = re.split(r"\n\d\. \*\*", section)[1:]
    assert len(steps) >= 7

    def step_with(*needles: str) -> int:
        for index, step in enumerate(steps):
            if all(needle in step for needle in needles):
                return index
        raise AssertionError(f"no step runs {needles}")

    assert step_with("useradd", "/home/paperwatch-api") == 0
    assert step_with("pg_ident.conf", "paperwatch-api", "psql") == 1
    # The checkout, uv and venv steps are section 15.1's, so there is one copy of them to drift.
    assert step_with("checkout", "section 15.1") == 2
    assert step_with("rm ~/.config/paperwatch/api.env") == 3
    # The port is never held twice: the old unit stops in the same step the new one starts.
    assert (
        step_with(
            "systemctl --user disable --now paperwatch-api", "systemctl enable --now paperwatch-api"
        )
        == 4
    )
    assert step_with("systemctl status paperwatch-api") == 5
    # The owner account keeps its path to the API's role until the last step, so every earlier
    # step can be undone without a database edit.
    assert step_with("rm ~/.config/systemd/user/paperwatch-api.service") == 6
    assert section.index("### Rolling back") > section.index("systemctl status paperwatch-api")

    # The tidy step removes the owner account's path to the API's role; the migration writes the
    # whole map out, so the final block is the two identity lines plus the API account's.
    assert _map_block(section) == [
        ["paperwatch", "paperwatch", "paperwatch"],
        ["paperwatch", "paperwatch", "paperwatch_ro"],
        ["paperwatch", "paperwatch-api", "paperwatch_api"],
    ]


def test_the_guide_keeps_the_api_env_file_secret() -> None:
    # The key moves to the new account's home, and the old copy is removed: leaving it in
    # `paperwatch`'s home would put the chat key back next to the ingestion that needs none. Read
    # from the commands the guide's shell blocks run, so a sentence cannot stand in for a step.
    guide = GUIDE.read_text()
    env_file = "/home/paperwatch-api/.config/paperwatch/api.env"
    install = guide[guide.index("### 15.2 The env file") : guide.index("### 15.3 The unit")]
    commands = _shell_commands(install)
    assert ["umask", "077"] in commands
    assert ["chown", "paperwatch-api:paperwatch-api", env_file] in commands
    assert ["chmod", "600", env_file] in commands
    migration = guide[guide.index("## 18. Moving an existing server onto the API account") :]
    assert ["rm", "~/.config/paperwatch/api.env"] in _shell_commands(migration)


def test_the_slice_bounds_the_ingestion_now() -> None:
    lines = SLICE_FILE.read_text().splitlines()
    start = lines.index("[Slice]") + 1
    section = [line for line in lines[start:] if line and not line.startswith("#")]
    assert "MemoryMax=1100M" in section
    assert "MemorySwapMax=0" in section


def test_the_ingest_drop_in_only_sets_the_slice() -> None:
    # The ingestion's own unit is not edited, so a release does not overwrite this line and this
    # line cannot acquire a second setting.
    lines = [
        line
        for line in INGEST_DROP_IN.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == ["[Service]", "Slice=paperwatch.slice"]


def test_the_api_binds_loopback_only() -> None:
    [line] = [line for line in _service_section(API_UNIT) if line.startswith("ExecStart=")]
    assert "--host 127.0.0.1" in line
    assert (
        "/home/paperwatch-api/.local/bin/uv run --frozen --no-dev paperwatch serve --live" in line
    )


def test_the_soak_stream_binds_loopback_only() -> None:
    # The stream is published by the tunnel's second mapping for the soak only, so the server
    # itself never listens on the machine's other interfaces.
    [line] = [line for line in _service_section(SOAK_UNIT) if line.startswith("ExecStart=")]
    argv = shlex.split(line.split("=", 1)[1])
    assert Path(argv[1]).name == "soak-stream.py"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8443"


def test_the_guide_shows_how_to_read_the_unit_security_score() -> None:
    # The number belongs to the deployed unit on the server, so the guide names the command and how
    # to read it rather than quoting a score that would go stale.
    guide = GUIDE.read_text()
    assert "systemd-analyze --user security paperwatch-ingest" in guide
    assert "Overall exposure level" in guide


def test_the_guide_binds_its_tunnel_to_loopback() -> None:
    # `-L 5433:...` listens on every interface, so on a shared wifi the database would be reachable
    # through the Mac; the guide's forward must name 127.0.0.1 and say when to close it.
    forwards = re.findall(r"^ssh -N -L (\S+) shared-vps$", GUIDE.read_text(), re.MULTILINE)
    assert forwards == ["127.0.0.1:5433:/var/run/postgresql/.s.PGSQL.5432"]
    assert "Close it when you are done" in GUIDE.read_text()


def test_the_guide_gives_each_of_the_two_roles_its_grants() -> None:
    # Beekeeper only reads, but the API and agent are not read-only: the daily chat cap records one
    # row per call in `chat_calls`, so a SELECT-only role fails the first answer. The owning role
    # would let a stray query write the corpus. Both roles and every grant they need are named, so
    # an admin can paste them.
    guide = GUIDE.read_text()
    # The sentence itself, so an unrelated mention of a read-only mode cannot satisfy the test.
    assert "the read-only role `paperwatch_ro` (SELECT only)" in guide
    assert "recommended connection for Beekeeper" in guide
    assert "The M3 API and agent are not read-only" in guide
    # The write the API needs is the cap's own row, and `id` comes from the sequence the migration
    # that creates `chat_calls` adds, so USAGE on that exact name is part of the grant.
    assert "GRANT INSERT ON chat_calls TO paperwatch_api;" in guide
    assert "GRANT USAGE ON SEQUENCE chat_calls_id_seq TO paperwatch_api;" in guide
    # Peer authentication sees the ssh account, so each role needs an ident map (or a password),
    # and default privileges keep the roles reading tables that later migrations create.
    assert "`pg_ident.conf`" in guide
    assert "ALTER DEFAULT PRIVILEGES FOR ROLE paperwatch" in guide

    # Letting those roles in takes a dedicated pg_hba.conf line, and the role field must list all
    # three: the second field is the database and the third the requested role, so a line naming
    # only `paperwatch` would send the other two to the generic local line without a map.
    assert [
        line.split() for line in guide.splitlines() if line.split()[:2] == ["local", "paperwatch"]
    ] == [
        [
            "local",
            "paperwatch",
            "paperwatch,paperwatch_ro,paperwatch_api",
            "peer",
            "map=paperwatch",
        ]
    ]
    # That line must come first, and never carry the map itself: `local all all peer` also covers
    # the other project's database, and a map there would gate that project's connections too.
    assert "placed above the generic" in guide
    assert "Never put `map=` on `local all all peer`" in guide

    # The map's first two lines keep the owner, hence the ingestion, and Beekeeper in; the third is
    # the whole point of the split: the API's own Unix account reaches `paperwatch_api` and nothing
    # else (ADR 0017). The map itself is read from the block, so prose cannot satisfy it.
    assert _map_block(guide) == [
        ["paperwatch", "paperwatch", "paperwatch"],
        ["paperwatch", "paperwatch", "paperwatch_ro"],
        ["paperwatch", "paperwatch-api", "paperwatch_api"],
    ]

    # The admin reloads the config and proves the owner still connects before trying the two
    # roles, because a bad line in either file would lock out the ingestion.
    assert "pg_reload_conf()" in guide and "pg_ctl reload" in guide
    assert "psql -d paperwatch -c 'select 1'" in guide
