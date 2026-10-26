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
import socket
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
    assert "verdict: FAIL - 1 failed streams, longest outage 4m 0s is over 2m" in result.stdout


def test_a_dropped_stream_fails_the_soak_and_its_outage_is_the_recovery(tmp_path: Path) -> None:
    # A tunnel restart closes the stream's connection. The probe reconnects, so the line carries
    # the recovery as outage=, and a drop is a failed stream even when it recovered in time: an
    # hourly stream that keeps dropping must never read as a clean soak.
    log = _write_log(
        tmp_path,
        [
            *_tick(0),
            _probe(
                0,
                "stream",
                "fail",
                "200",
                300004,
                frames="18",
                gap="40",
                drops="1",
                outage="40",
                hb="15",
                why="drop",
            ),
            *_tick(300),
        ],
    )
    result = _summary(log)
    assert result.returncode == 1
    assert "streams: 1, 0 ok, 1 failed" in result.stdout
    assert "longest outage: 40s" in result.stdout
    assert "verdict: FAIL - 1 failed streams" in result.stdout


def test_a_stream_outage_over_two_minutes_fails_the_rule(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path,
        [
            *_tick(0),
            _probe(
                0,
                "stream",
                "fail",
                "200",
                300004,
                frames="12",
                gap="150",
                drops="1",
                outage="150",
                hb="15",
                why="drop",
            ),
        ],
    )
    result = _summary(log)
    assert "longest outage: 2m 30s" in result.stdout
    assert "verdict: FAIL - 1 failed streams, longest outage 2m 30s is over 2m" in result.stdout


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


# One bound for both ends of a probe test, the lines it collects and the stop it asks for, so a
# loaded runner is treated the same way at both. The probe holds its stream for the window the test
# passes (seven seconds at most here) and answers a stop at once, whatever it was waiting for
# (`test_a_running_probe_stops_promptly_with_a_probe_in_flight`), so this leaves a slow runner
# seconds to spare while still failing a probe that waits on a window that has already closed.
PROBE_TIMEOUT = 30


def _wait_for_lines(log: Path, count: int, timeout: float = PROBE_TIMEOUT) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists():
            lines = [line for line in log.read_text().splitlines() if line.strip()]
            if len(lines) >= count:
                return lines
        time.sleep(0.2)
    raise AssertionError(f"{log} never collected {count} probe lines")


def _stop(proc: subprocess.Popen[str], what: str) -> None:
    """Stop a child, and report a signal it ignored as that instead of as a timeout."""
    proc.terminate()
    try:
        proc.wait(timeout=PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise AssertionError(f"{what} was still running {PROBE_TIMEOUT}s after SIGTERM") from None


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
        try:
            _stop(probe, "the soak probe")
        finally:
            stream.terminate()
            stream.wait(timeout=15)

    assert lines[0].split()[1].startswith("at=")
    assert re.fullmatch(r"ts=\S+ at=\d+ probe=healthz result=ok http=200 ms=\d+", lines[0]), lines[
        0
    ]
    assert re.fullmatch(r"ts=\S+ at=\d+ probe=status result=ok http=401 ms=\d+", lines[1]), lines[1]
    assert re.fullmatch(
        r"ts=\S+ at=\d+ probe=stream result=ok http=200 ms=\d+ frames=\d+ gap=\d+ drops=0"
        r" outage=0 hb=\d+",
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


class _DroppingStream(BaseHTTPRequestHandler):
    """A stream whose first connection breaks, as a tunnel restart breaks it.

    The first connection sends frames for a second and closes, or with `stall` goes silent and
    keeps the socket open, as a connection to an ingress that lost its tunnel does. Every later one
    answers 503 (the tunnel is still coming back) until `refusals` have been answered, then streams
    again; with `refusals=None` it never comes back.
    """

    refusals: int | None = 1
    stall = False
    connections = 0

    def do_GET(self) -> None:  # http.server's spelling, not ours
        cls = type(self)
        cls.connections += 1
        attempt = cls.connections
        if attempt > 1 and (cls.refusals is None or attempt <= 1 + cls.refusals):
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        deadline = time.monotonic() + (1.0 if attempt == 1 else 30.0)
        try:
            while time.monotonic() < deadline:
                self.wfile.write(b'data: {"tick": 0}\n\n')
                self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            return
        if attempt == 1 and cls.stall:
            time.sleep(30)

    def log_message(self, *_args: object) -> None:
        pass


def _stream_line(
    tmp_path: Path, api_stub: str, refusals: int | None, *, stall: bool = False, window: int = 7
) -> str:
    attrs = {"refusals": refusals, "stall": stall, "connections": 0}
    handler = type("Stream", (_DroppingStream,), attrs)
    server, port = _serve(handler)
    log = tmp_path / "soak.log"
    probe = subprocess.Popen(  # noqa: S603 (a fixed script path, no shell)
        [
            BASH,
            str(SCRIPT),
            "run",
            api_stub,
            "--stream-url",
            f"http://127.0.0.1:{port}/events",
            "--log",
            str(log),
            "--interval",
            "1",
            "--stream-window",
            str(window),
            "--stream-every",
            "3600",
            "--heartbeat",
            "1",
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
        try:
            _stop(probe, "the soak probe")
        finally:
            server.shutdown()
            server.server_close()
    [line] = [line for line in lines if " probe=stream " in line]
    return line


def _fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split())


def test_the_stream_reconnects_after_a_drop_and_records_the_recovery(
    tmp_path: Path, api_stub: str
) -> None:
    fields = _fields(_stream_line(tmp_path, api_stub, refusals=1))
    assert fields["result"] == "fail"
    assert fields["why"] == "drop"
    assert fields["http"] == "200"
    assert fields["drops"] == "1"
    # One second of pause after the drop, a 503, another second, then frames again.
    assert 1 <= int(fields["outage"]) <= 5, fields
    assert int(fields["frames"]) > 6, fields


def test_a_stream_that_never_comes_back_is_unrecovered(tmp_path: Path, api_stub: str) -> None:
    line = _stream_line(tmp_path, api_stub, refusals=None)
    fields = _fields(line)
    assert fields["why"] == "unrecovered"
    assert fields["drops"] == "1"
    # The outage runs from the last frame to the window's end: about six of its seven seconds.
    assert int(fields["outage"]) >= 4, fields

    result = _summary(_write_log(tmp_path, [line], name="unrecovered.log"))
    assert result.returncode == 1
    assert "verdict: FAIL - 1 failed streams" in result.stdout


def test_a_stalled_stream_is_abandoned_and_its_recovery_timed(
    tmp_path: Path, api_stub: str
) -> None:
    # A silent connection that never closes would otherwise hold the window to its end and read ok.
    # Three heartbeats of silence (3 s at --heartbeat 1) plus curl's five-second speed average
    # abandon it, then the reconnect streams.
    fields = _fields(_stream_line(tmp_path, api_stub, refusals=0, stall=True, window=16))
    assert fields["result"] == "fail"
    assert fields["why"] == "drop"
    assert fields["drops"] == "1"
    # The outage is timed from the last frame, so it covers the stall, the pause and the reconnect.
    assert 3 <= int(fields["outage"]) <= 14, fields
    assert int(fields["frames"]) > 6, fields


def test_a_stalled_stream_that_never_comes_back_is_unrecovered(
    tmp_path: Path, api_stub: str
) -> None:
    line = _stream_line(tmp_path, api_stub, refusals=None, stall=True, window=14)
    fields = _fields(line)
    assert fields["why"] == "unrecovered"
    assert fields["drops"] == "1"
    # The outage runs from the last frame at about one second to the window's end at fourteen.
    assert int(fields["outage"]) >= 10, fields
    assert int(fields["gap"]) >= 10, fields

    result = _summary(_write_log(tmp_path, [line], name="stalled.log"))
    assert result.returncode == 1
    assert "verdict: FAIL - 1 failed streams" in result.stdout


def test_a_running_probe_stops_promptly_with_a_probe_in_flight(tmp_path: Path) -> None:
    # The soak runs for 72 hours and is stopped by hand or by systemd, so the stop itself is worth
    # testing. bash runs a trapped signal only once the command it is waiting for returns, so a
    # probe blocked in curl defers SIGTERM for curl's whole --max-time and the sleep between ticks
    # defers it for up to --interval; the probe waits on its children in the background for that
    # reason. The five seconds here are far under the 30 s max-time given below, so a stop that
    # still waits on the tunnel fails.
    log = tmp_path / "soak.log"
    assert BASH, "the soak probe is bash"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        probe = subprocess.Popen(  # noqa: S603 (a fixed script path, no shell)
            [
                BASH,
                str(SCRIPT),
                "run",
                f"http://127.0.0.1:{port}",
                "--stream-url",
                f"http://127.0.0.1:{port}/events",
                "--log",
                str(log),
                "--interval",
                "300",
                "--stream-window",
                "1",
                "--max-time",
                "30",
                "--hours",
                "0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # A listener that accepts and never answers holds the first healthz probe inside curl
            # from the moment the connection lands.
            listener.settimeout(30)
            connection, _ = listener.accept()
            with connection:
                probe.terminate()
                probe.wait(timeout=5)
        finally:
            probe.kill()
            probe.wait(timeout=15)
    assert probe.returncode == 130, probe.stderr


def _run_args(*args: str) -> subprocess.CompletedProcess[str]:
    assert BASH, "the soak probe is a bash script"
    return subprocess.run(  # noqa: S603 (a fixed script path, no shell)
        [BASH, str(SCRIPT), "run", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["https://example.test"], "run needs --stream-url URL"),
        (["example.test", "--stream-url", "https://example.test:8443/events"], "BASE_URL must"),
        (
            ["https://example.test", "--stream-url", "https://example.test:8443/events", "--hours"],
            "--hours needs a value",
        ),
        (
            [
                "https://example.test",
                "--stream-url",
                "https://example.test:8443/events",
                "--interval",
                "5m",
            ],
            "--interval must be a whole number of seconds, got 5m",
        ),
    ],
    ids=["no-stream-url", "no-scheme", "missing-value", "not-a-number"],
)
def test_a_wrong_run_command_exits_2_before_probing(args: list[str], message: str) -> None:
    # The soak is started by hand once: a typo must stop at once with the reason, not 72 hours of
    # probes against the wrong address or a shell arithmetic error at the first tick.
    result = _run_args(*args)
    assert result.returncode == 2
    assert message in result.stderr
