"""`deploy/vps/tunnel-soak.sh` and `deploy/vps/soak-stream.py`: the 72-hour tunnel soak.

The soak itself runs against the public tunnel for 72 hours, so what is worth testing is the part
that only runs once, at the end: the arithmetic over the probe log. What counts as a failed
handshake, how long an outage lasted, and whether the verdict matches the pass rule (no failed
handshake, no failed probe, under two minutes to recover) are decisions a reader cannot check by
eye from 1,800 log lines. The probe itself is run once against a local stub so the line the
summary parses is the line the probe writes, and the stub stream server is the real one.
"""

import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from paperwatch.config import REPO_ROOT

SCRIPT = REPO_ROOT / "deploy" / "vps" / "tunnel-soak.sh"
STREAM_SERVER = REPO_ROOT / "deploy" / "vps" / "soak-stream.py"
UNIT = REPO_ROOT / "deploy" / "vps" / "paperwatch-soak-stream.service"
GUIDE = REPO_ROOT / "docs" / "deploy" / "vps.md"
BASH = shutil.which("bash")

T0 = 1792915200  # 2026-10-25T08:00:00Z, an hour boundary: the hourly stream starts here


def _at(seconds: int) -> int:
    return T0 + seconds


def _stamp(seconds: int) -> str:
    # The fixture stamps are hand-written, so this keeps them a real ISO UTC second.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_at(seconds)))


def _probe(seconds: int, probe: str, result: str, http: str, ms: int, **extra: str) -> str:
    fields = {"ts": _stamp(seconds), "at": str(_at(seconds)), "probe": probe, "result": result}
    fields["http"] = http
    fields["ms"] = str(ms)
    fields.update(extra)
    return " ".join(f"{key}={value}" for key, value in fields.items())


def _tick(seconds: int, *, result: str = "ok", http: str = "200", **extra: str) -> list[str]:
    """One five-minute tick: both probes through the tunnel."""
    return [
        _probe(seconds, "healthz", result, http, 312, **extra),
        _probe(seconds + 1, "status", result, http if http != "200" else "401", 298, **extra),
    ]


def _write_log(tmp_path: Path, lines: list[str], name: str = "soak.log") -> Path:
    log = tmp_path / name
    log.write_text("\n".join(lines) + "\n")
    return log


def _summary(log: Path) -> subprocess.CompletedProcess[str]:
    assert BASH, "the soak probe is a bash script"
    return subprocess.run(  # noqa: S603 (a fixed script path, no shell)
        [BASH, str(SCRIPT), "summary", str(log)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_a_clean_soak_passes(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        [
            *_tick(0),
            _probe(0, "stream", "ok", "200", 300004, frames="20", gap="15", hb="15"),
            *_tick(300),
            *_tick(605),
        ],
    )
    result = _summary(log)
    assert result.returncode == 0, result.stderr
    assert f"log: {log}" in result.stdout
    assert "window: 2026-10-25T08:00:00Z to 2026-10-25T08:10:06Z (10m 6s)" in result.stdout
    assert "probes: 6, 6 ok, 0 failed (healthz 3, status 3)" in result.stdout
    assert "failed handshakes: 0" in result.stdout
    assert "streams: 1, 1 ok, 0 failed" in result.stdout
    assert "longest outage: 0s" in result.stdout
    assert "longest gap between probes: 5m 5s" in result.stdout
    assert "verdict: PASS" in result.stdout


def test_failed_handshakes_are_counted_and_fail_the_soak(tmp_path: Path) -> None:
    # curl exits before any HTTP response when the tunnel's listener is gone, so the log has
    # http=000 and why=handshake: that is the one number the pass rule is built on.
    log = _write_log(
        tmp_path,
        [
            *_tick(0, result="fail", http="000", why="handshake", curl="7"),
            *_tick(300),
        ],
    )
    result = _summary(log)
    assert result.returncode == 1
    assert "failed handshakes: 2" in result.stdout
    assert "probes: 4, 2 ok, 2 failed (healthz 2, status 2)" in result.stdout
    assert "longest outage: 5m 0s" in result.stdout
    assert "verdict: FAIL - 2 failed handshakes, 2 failed probes" in result.stdout


def test_the_longest_outage_runs_to_the_next_successful_probe(tmp_path: Path) -> None:
    # Health is confirmed only by a probe, so an outage is measured from the first failed probe to
    # the next successful one of the same kind. Two failed probes five minutes apart and a success
    # five minutes later is the only reading the log supports.
    log = _write_log(
        tmp_path,
        [
            _probe(0, "healthz", "fail", "000", 20000, why="handshake"),
            _probe(300, "healthz", "fail", "000", 20000, why="handshake"),
            _probe(600, "healthz", "ok", "200", 400),
            _probe(600, "status", "ok", "401", 300),
        ],
    )
    result = _summary(log)
    assert "longest outage: 10m 0s" in result.stdout
    assert "verdict: FAIL" in result.stdout


def test_a_stream_gap_is_an_outage_and_an_interrupted_stream_is_a_failure(tmp_path: Path) -> None:
    # A stream sees a break to the nearest heartbeat, which is what measures the two-minute
    # recovery: 15 s apart, a 4-minute gap is a real outage the probes would round to five.
    log = _write_log(
        tmp_path,
        [
            *_tick(0),
            _probe(0, "stream", "ok", "200", 300004, frames="20", gap="15", hb="15"),
            *_tick(300),
            _probe(
                3600, "stream", "fail", "200", 300004, frames="4", gap="240", hb="15", why="gap"
            ),
        ],
    )
    result = _summary(log)
    assert result.returncode == 1
    assert "failed handshakes: 0" in result.stdout
    assert "streams: 2, 1 ok, 1 failed" in result.stdout
    assert "longest outage: 4m 0s" in result.stdout
    assert "verdict: FAIL - longest outage 4m 0s is over 2m" in result.stdout


def test_an_empty_log_is_an_error_not_a_pass(tmp_path: Path) -> None:
    # A missing probe run must never read as a clean 72 hours.
    log = _write_log(tmp_path, [])
    result = _summary(log)
    assert result.returncode == 2
    assert "no probes" in result.stderr


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class _ApiStub(BaseHTTPRequestHandler):
    """The two routes the soak probes, answering exactly as the API does without a token."""

    def do_GET(self) -> None:  # http.server's spelling, not ours
        body, status = {
            "/healthz": (b'{"status":"ok"}', 200),
            "/status": (b'{"detail":"unauthorized"}', 401),
        }.get(self.path, (b"not found", 404))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def api_stub() -> Iterator[str]:
    server, port = _serve(_ApiStub)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _start_stream_server() -> subprocess.Popen[str]:
    assert BASH, "the probe script is bash"
    proc = subprocess.Popen(  # noqa: S603 (the repository's own server, no shell)
        [
            sys.executable,
            str(STREAM_SERVER),
            "--port",
            "0",
            "--heartbeat",
            "0.2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def _stream_url(proc: subprocess.Popen[str]) -> str:
    assert proc.stdout is not None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        match = re.search(r"http://\S+/events", line)
        if match:
            return match.group(0)
    raise AssertionError("the soak stream server never printed its address")


def _wait_for_lines(log: Path, count: int, timeout: float = 30) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists():
            lines = [line for line in log.read_text().splitlines() if line.strip()]
            if len(lines) >= count:
                return lines
        time.sleep(0.2)
    raise AssertionError(f"{log} never collected {count} probe lines")


def test_the_probe_logs_one_line_per_probe_with_the_fields_the_summary_reads(
    tmp_path: Path, api_stub: str
) -> None:
    # The summary is only as good as the format: every field it parses is asserted here, against
    # the real ApiStub and the real stub stream, with a one-second cadence.
    stream = _start_stream_server()
    log = tmp_path / "soak.log"
    probe = subprocess.Popen(  # noqa: S603 (a fixed script path, no shell)
        [
            BASH,
            str(SCRIPT),
            "run",
            api_stub,
            "--stream-url",
            _stream_url(stream),
            "--log",
            str(log),
            "--interval",
            "1",
            "--stream-window",
            "1",
            "--stream-every",
            "3600",
            "--max-time",
            "5",
            "--hours",
            "0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        lines = _wait_for_lines(log, 3)
    finally:
        probe.terminate()
        probe.wait(timeout=15)
        stream.terminate()
        stream.wait(timeout=15)

    assert lines[0].split()[1].startswith("at=")
    assert re.fullmatch(r"ts=\S+ at=\d+ probe=healthz result=ok http=200 ms=\d+", lines[0]), lines[
        0
    ]
    assert re.fullmatch(r"ts=\S+ at=\d+ probe=status result=ok http=401 ms=\d+", lines[1]), lines[1]
    assert re.fullmatch(
        r"ts=\S+ at=\d+ probe=stream result=ok http=200 ms=\d+ frames=\d+ gap=\d+ hb=\d+",
        lines[2],
    ), lines[2]

    clean = _summary(log)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "failed handshakes: 0" in clean.stdout
    assert "verdict: PASS" in clean.stdout


def test_the_soak_stream_sends_heartbeat_frames() -> None:
    stream = _start_stream_server()
    curl = shutil.which("curl")
    assert curl, "curl is the probe's only network tool"
    try:
        url = _stream_url(stream)
        response = subprocess.run(  # noqa: S603 (a fixed binary and URL, no shell)
            [curl, "-sS", "-N", "--max-time", "3", url],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        stream.terminate()
        stream.wait(timeout=15)
    frames = [line for line in response.stdout.splitlines() if line.startswith("data:")]
    assert len(frames) >= 2, response.stdout
    assert frames[0].startswith("data: {")


def test_the_stream_unit_binds_loopback_and_never_gains_privileges() -> None:
    # The stream exists for the soak only, so it is published by the tunnel's own mapping and
    # never listens on the machine's other interfaces.
    text = UNIT.read_text()
    assert "--host 127.0.0.1" in text
    assert "--port 8443" in text
    assert "NoNewPrivileges=yes" in text
    assert STREAM_SERVER.name in text


def test_the_guide_shows_how_to_run_read_and_remove_the_soak() -> None:
    # The soak is run by hand, once, months from now: the commands in the guide are the interface.
    guide = GUIDE.read_text()
    assert "./deploy/vps/tunnel-soak.sh run https://<PW_API_PUBLIC_HOST>" in guide
    assert "./deploy/vps/tunnel-soak.sh summary ~/paperwatch-soak.log" in guide
    assert "funnel --bg 8443 http://127.0.0.1:8443" in guide
    assert "funnel --https=8443 off" in guide
    assert "systemctl --user restart paperwatch-tunnel" in guide
    assert "paperwatch-soak-stream" in guide
