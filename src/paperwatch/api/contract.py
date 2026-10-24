"""Build the committed OpenAPI contract without a database, a key or a model.

`docs/api/openapi.json` is the client's reference, so it must be reproducible from the code alone:
this module wires the app with stub dependencies and writes the document. The dependencies are
stubs because the contract depends only on the routes and their models.

Rewrite it after an intended API change:

    uv run python -m paperwatch.api.contract
"""

import json
from datetime import UTC, datetime
from typing import Any

from paperwatch.api.app import AgentDeps, create_app
from paperwatch.api.auth import HmacVerifier
from paperwatch.api.limits import RequestLimiter
from paperwatch.api.search_mode import SearchModeGate
from paperwatch.config import (
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    REPO_ROOT,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    Settings,
)
from paperwatch.llm.embedder import FakeEmbedder

# The contract's own file, resolved from the package so the command works from any directory.
CONTRACT_PATH = REPO_ROOT / "docs" / "api" / "openapi.json"


class _NoCalls:
    """The `ChatCallStore` stub: building the document never asks about a call."""

    def record_chat_call(self, **kwargs: Any) -> None:
        raise AssertionError("the contract build makes no chat call")

    def chat_calls_since(self, at: datetime) -> int:
        raise AssertionError("the contract build makes no chat call")


class _NoRun:
    """A session whose one query finds no running pass, so the gate stays hybrid."""

    def __enter__(self) -> "_NoRun":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def scalar(self, statement: Any) -> None:
        return None


def build_contract() -> dict[str, Any]:
    """The OpenAPI document for the served app, built with stub dependencies."""
    deps = AgentDeps(
        session_factory=lambda: None,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        models=[],
        settings=Settings(),
        clock=lambda: datetime.now(UTC),
        chat_calls=_NoCalls(),
        profile_names=frozenset(),
        verifier=HmacVerifier("x" * 32, lambda: 0.0),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: datetime.now(UTC),
        ),
        search_mode=SearchModeGate(
            lambda: _NoRun(),  # type: ignore[arg-type,return-value]
            clock=lambda: datetime.now(UTC),
            read_meminfo=lambda: None,
            read_psi=lambda: None,
        ),
    )
    return create_app(deps).openapi()


if __name__ == "__main__":
    CONTRACT_PATH.write_text(json.dumps(build_contract(), indent=2) + "\n")
