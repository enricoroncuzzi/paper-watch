"""The gate in front of every API route: credential, host, origin, method and size.

The app here is a tiny Starlette application, so a failure is the gate's and not FastAPI's. The
tests drive it through a `TestClient` whose base URL is `localhost`, and through raw ASGI where the
order of the checks (the body is never read before the credential) is what must be proven.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from paperwatch.api.app import AgentDeps, create_app
from paperwatch.api.auth import Principal
from paperwatch.api.gate import ApiGate, _host_name
from paperwatch.api.limits import RequestLimiter
from paperwatch.config import (
    API_MAX_BODY_BYTES,
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    Settings,
)
from paperwatch.llm.embedder import FakeEmbedder

GOOD = "good"
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


class StubVerifier:
    """Accepts one literal token, so the gate tests never mint or expire a real credential."""

    def verify(self, token: str) -> Principal | None:
        return Principal("hmac", "test") if token == GOOD else None


async def _echo(request: Request) -> JSONResponse:
    body = await request.body()
    return JSONResponse({"size": len(body)})


def _inner() -> Starlette:
    return Starlette(
        routes=[
            Route("/", lambda request: PlainTextResponse("ok")),
            Route("/healthz", lambda request: JSONResponse({"status": "ok"})),
            Route("/echo", _echo, methods=["POST"]),
        ]
    )


def _gated(
    allowed_hosts: frozenset[str] = LOOPBACK_HOSTS,
    max_body_bytes: int = API_MAX_BODY_BYTES,
) -> ApiGate:
    return ApiGate(
        _inner(),
        verifier=StubVerifier(),
        allowed_hosts=allowed_hosts,
        max_body_bytes=max_body_bytes,
    )


def _client(
    allowed_hosts: frozenset[str] = LOOPBACK_HOSTS,
    max_body_bytes: int = API_MAX_BODY_BYTES,
) -> TestClient:
    return TestClient(_gated(allowed_hosts, max_body_bytes), base_url="http://localhost")


def _auth(token: str = GOOD) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_every_bad_token_gets_the_same_401() -> None:
    # One body and one status for every reason, so a caller learns nothing about why it failed.
    client = _client()
    responses = [
        client.get("/"),
        client.get("/", headers={"Authorization": "Basic eA=="}),
        client.get("/", headers={"Authorization": "Bearer bad"}),
        client.get("/", headers={"Authorization": "Bearer "}),
    ]
    assert {r.status_code for r in responses} == {401}
    assert len({r.content for r in responses}) == 1
    assert json.loads(responses[0].content) == {"detail": "unauthorized"}
    assert {r.headers["www-authenticate"] for r in responses} == {"Bearer"}


def test_the_body_is_not_read_before_the_credential() -> None:
    # If the receive channel were touched, a 16 KB cap would be applied only after an anonymous
    # caller could already make the server read a large body.
    gate = _gated()
    read = False

    async def receive() -> dict[str, Any]:
        nonlocal read
        read = True
        raise AssertionError("the body must not be read before the credential")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/echo",
        "raw_path": b"/echo",
        "query_string": b"",
        "headers": [(b"host", b"localhost"), (b"content-type", b"application/json")],
        "server": ("localhost", 80),
        "client": ("testclient", 1),
        "scheme": "http",
        "root_path": "",
    }
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    import asyncio

    asyncio.run(gate(scope, receive, send))
    assert not read
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 401


def test_credential_comes_before_host_and_origin() -> None:
    response = _client().get("/", headers={"Host": "evil.example", "Origin": "https://example.org"})
    assert response.status_code == 401


def test_a_public_host_must_be_allowed() -> None:
    assert _client().get("/", headers=_auth()).status_code == 200
    assert _client().get("/", headers={**_auth(), "Host": "abc.ts.net"}).status_code == 400
    allowed = _client(frozenset({"localhost", "abc.ts.net"}))
    assert allowed.get("/", headers={**_auth(), "Host": "abc.ts.net"}).status_code == 200
    # A browser writes an IPv6 host in brackets, and the old loopback check is now the gate's.
    assert _client().get("/", headers={**_auth(), "Host": "[::1]:8000"}).status_code == 200


def test_any_origin_is_refused() -> None:
    # The page's server calls the API, and it sends no Origin; a browser does, so any Origin came
    # from a page that is not allowed to talk to this API.
    response = _client().get("/", headers={**_auth(), "Origin": "https://example.org"})
    assert response.status_code == 403


def test_post_needs_json() -> None:
    response = _client().post(
        "/echo", headers={**_auth(), "Content-Type": "text/plain"}, content=b"hello"
    )
    assert response.status_code == 415


def test_a_large_body_is_refused_by_header_and_by_stream() -> None:
    client = _client()
    payload = b"x" * 20_000
    by_header = client.post(
        "/echo",
        headers={**_auth(), "Content-Type": "application/json"},
        content=payload,
    )
    assert by_header.status_code == 413

    # No Content-Length: a chunked body must hit the same cap, or the header check is trivial to
    # bypass.
    def chunks() -> Any:
        for _ in range(20):
            yield b"x" * 1_000

    by_stream = client.post(
        "/echo",
        headers={**_auth(), "Content-Type": "application/json"},
        content=chunks(),
    )
    assert by_stream.status_code == 413


def test_the_health_route_is_the_one_route_without_a_credential() -> None:
    # The captain's requirement: a liveness probe must answer without a token, and nothing else
    # does. A health body is fixed, touches no database and no model.
    client = _client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 401
    assert client.post("/echo", content=b"{}").status_code == 401


def test_docs_routes_are_gone() -> None:
    # Served only to an authenticated request would still publish the contract to anyone who can
    # mint a token; the served app simply has no docs or OpenAPI routes (the contract is committed
    # separately).
    client = TestClient(create_app(_deps()), base_url="http://localhost", headers=_auth())
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_a_health_probe_through_the_tunnel_still_needs_the_public_host() -> None:
    # The tunnel forwards with its own Host, so the operator must name it in PW_API_PUBLIC_HOST;
    # an unauthenticated route must not become a host-check bypass.
    response = _client().get("/healthz", headers={"Host": "abc.ts.net"})
    assert response.status_code == 400


def test_the_app_always_installs_the_gate() -> None:
    app = create_app(_deps())
    assert any(middleware.cls is ApiGate for middleware in app.user_middleware)


@pytest.mark.parametrize(
    "header, expected",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:8000", "127.0.0.1"),
        ("localhost", "localhost"),
        ("LOCALHOST:8000", "localhost"),
        ("[::1]:8000", "::1"),
        ("[::1]", "::1"),
        ("evil.example", "evil.example"),
        ("127.0.0.1.evil.example", "127.0.0.1.evil.example"),
        ("localhost.", "localhost."),
        ("localhost.:8000", "localhost."),
        ("[::1].evil.example", ""),
        ("[::1", ""),
    ],
)
def test_the_host_header_is_read_the_way_a_browser_writes_it(header: str, expected: str) -> None:
    assert _host_name(header) == expected


def test_a_loopback_server_refuses_a_foreign_host() -> None:
    # Moved from tests/unit/test_cli.py: the check now lives in the gate, not in a CLI middleware.
    client = _client()
    for host in ("evil.example", "127.0.0.1.evil.example", "localhost.", "localhost.:8000"):
        assert client.get("/healthz", headers={"Host": host}).status_code == 400, host


class _Store:
    def record_chat_call(self, **kwargs: Any) -> None:
        raise AssertionError("no chat call in a gate test")

    def chat_calls_since(self, at: datetime) -> int:
        raise AssertionError("no chat call in a gate test")


def _deps() -> AgentDeps:
    return AgentDeps(
        session_factory=lambda: None,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=[],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=_Store(),
        profile_names=frozenset(),
        verifier=StubVerifier(),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
    )


def test_a_gated_fastapi_app_keeps_its_response_shape() -> None:
    # A sanity check that the gate is transparent to a normal route: the body, status and headers
    # the route writes are what the client receives.
    app = FastAPI()
    app.add_middleware(ApiGate, verifier=StubVerifier(), allowed_hosts=LOOPBACK_HOSTS)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "1"}

    client = TestClient(app, base_url="http://localhost")
    assert client.get("/ping", headers=_auth()).json() == {"pong": "1"}


def test_the_gate_is_asgi_so_streaming_is_untouched() -> None:
    # Plain ASGI, never a BaseHTTPMiddleware: a middleware that wrapped the response would break
    # the disconnect path `_ClosingStream` depends on.
    from starlette.middleware.base import BaseHTTPMiddleware

    assert not issubclass(ApiGate, BaseHTTPMiddleware)
    assert isinstance(_gated(), ApiGate)
