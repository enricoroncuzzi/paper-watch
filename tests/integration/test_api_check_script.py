"""`deploy/vps/api-check.sh` against a real uvicorn server, gated and ungated.

The gated app is the real `create_app` with an HMAC verifier and the test database; the token is
minted fresh per authenticated check by a tiny command, because an HMAC token is single use. The
ungated app is a plain FastAPI app with its default docs routes, which the script must fail on.
"""

import contextlib
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from sqlalchemy.engine import Engine

from paperwatch.api.app import AgentDeps, create_app
from paperwatch.api.auth import HmacVerifier
from paperwatch.api.limits import RequestLimiter
from paperwatch.api.search_mode import ModeDecision
from paperwatch.config import (
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    Settings,
)
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.store.db import make_session_factory

pytestmark = pytest.mark.integration
SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "vps" / "api-check.sh"
SH = "/bin/sh"
SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


class _Store:
    def record_chat_call(self, **kwargs: Any) -> None:
        raise AssertionError("the acceptance check makes no chat call")

    def chat_calls_since(self, at: datetime) -> int:
        raise AssertionError("the acceptance check makes no chat call")


class _HybridMode:
    def current(self) -> ModeDecision:
        return ModeDecision("hybrid", "ok")


def _deps(engine: Engine) -> AgentDeps:
    factory = make_session_factory(engine)
    return AgentDeps(
        session_factory=factory,
        embedder=FakeEmbedder(),
        models=[],
        settings=Settings(),
        clock=lambda: now(),
        chat_calls=_Store(),
        profile_names=frozenset(),
        verifier=HmacVerifier(SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: now(),
        ),
        search_mode=_HybridMode(),
        ready_session_factory=factory,
    )


def now() -> datetime:
    return NOW


@contextlib.contextmanager
def _served(app: Any) -> Iterator[str]:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - a server that never starts is a broken test, not a product path
        server.should_exit = True
        raise AssertionError("uvicorn did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _run(base: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - a fixed script path and a fixed shell
        [SH, str(SCRIPT), base],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_api_check_passes_against_a_gated_app(engine: Engine, tmp_path: Path) -> None:
    mint = tmp_path / "mint.py"
    mint.write_text(
        "import time\n"
        "from paperwatch.api.auth import mint_hmac_token\n"
        f"print(mint_hmac_token({SECRET!r}, 'check', time.time()))\n"
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(mint))}"
    # A stub that answers the script's systemd probe with failure, so the unit checks are the WARN
    # path on a laptop and on a CI runner alike.
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "systemctl").write_text("#!/bin/sh\nexit 1\n")
    (stub / "systemctl").chmod(0o755)
    env = {
        **os.environ,
        "PW_CHECK_TOKEN_CMD": command,
        "PATH": f"{stub}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    env.pop("PW_CHECK_TOKEN", None)
    with _served(create_app(_deps(engine))) as base:
        result = _run(base, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "all checks passed" in result.stdout
    # The captain's requirement: health is the one route without a credential.
    assert "ok: healthz answers without a token" in result.stdout
    assert "ok: docs need a credential" in result.stdout
    assert "ok: a body over 16 KB is refused" in result.stdout
    assert "ok: a foreign Host is refused on loopback" in result.stdout


def test_api_check_fails_when_docs_are_open() -> None:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    env = {key: value for key, value in os.environ.items() if not key.startswith("PW_CHECK_")}
    with _served(app) as base:
        result = _run(base, env)
    assert result.returncode == 1
    assert "FAIL: docs need a credential" in result.stdout
    assert "all checks passed" not in result.stdout
