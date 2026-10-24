"""The first thing every request meets, before FastAPI parses it or any model work can start.

The order is the point: credential, host, `Origin`, method and content type, size. Each refusal is
decided before the next check runs, so an unauthenticated caller cannot learn the host allow-list,
cause a body read, or reach a parsed field. `/healthz` is the one exception to the credential (the
captain's slice 2 requirement): a liveness probe must answer without a token, and that route
touches neither the database nor a model.

Plain ASGI rather than a `BaseHTTPMiddleware`: `/ask` streams for seconds, and a middleware that
buffered or wrapped the response would change the disconnect path `_ClosingStream` depends on.
"""

from typing import Any

import anyio
import structlog
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from paperwatch.api.auth import Verifier
from paperwatch.config import API_MAX_BODY_BYTES

log = structlog.get_logger()

# The one route without a credential. The match is exact and GET-only, so a POST elsewhere never
# slips through as a health check.
PUBLIC_PATH = "/healthz"


def _host_name(header: str) -> str:
    """The host part of a `Host` header, lowercased, or "" when it is not a host at all.

    A browser writes an IPv6 host in brackets (`[::1]:8000`), a port may follow any host, and a
    trailing dot or embedded credentials make a name that is not one of the allowed ones. A plain
    string comparison against `::1` can never match a real header, which is why a naive check made
    `serve --host ::1` refuse every request.
    """
    if header.startswith("["):
        end = header.find("]")
        if end < 0:
            return ""
        name, rest = header[1:end].lower(), header[end + 1 :]
        # Only `[host]` and `[host]:port` are a host header; `[::1].evil.example` is not.
        if rest and (not rest.startswith(":") or not rest[1:].isdigit()):
            return ""
        return name
    name, sep, port = header.partition(":")
    return name.lower() if not sep or port.isdigit() else ""


class _BodyTooLarge(BaseException):
    """A body outgrew the cap while being streamed in.

    A `BaseException`, like `AnswerAborted`: Starlette's error middleware catches `Exception` and
    would answer 500 for a request the gate must refuse with 413.
    """


class _CappedReceive:
    """Count the bytes of a body the app reads, so a chunked request hits the same cap.

    A `Content-Length` header can be absent or a lie, so the header check alone is not a limit.
    """

    def __init__(self, receive: Receive, max_bytes: int) -> None:
        self._receive = receive
        self._max_bytes = max_bytes
        self._seen = 0

    async def __call__(self) -> Any:
        message = await self._receive()
        if message["type"] == "http.request":
            self._seen += len(message.get("body", b""))
            if self._seen > self._max_bytes:
                raise _BodyTooLarge
        return message


def _bearer(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


class ApiGate:
    """Refuse, in a fixed order, every request the API should not see."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        verifier: Verifier,
        allowed_hosts: frozenset[str],
        max_body_bytes: int = API_MAX_BODY_BYTES,
    ) -> None:
        self._app = app
        self._verifier = verifier
        self._allowed_hosts = allowed_hosts
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # No websocket or lifespan route exists; only the HTTP gate is this gate's business.
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        path = scope["path"]
        method = scope["method"]

        if not (path == PUBLIC_PATH and method == "GET"):
            token = _bearer(headers.get("authorization"))
            if token is None:
                await self._refuse(scope, receive, send, 401, "unauthorized", "credential")
                return
            # In a worker thread: a key fetch is a network call, and the event loop serves every
            # other request meanwhile.
            principal = await anyio.to_thread.run_sync(self._verifier.verify, token)
            if principal is None:
                await self._refuse(scope, receive, send, 401, "unauthorized", "credential")
                return
            scope.setdefault("state", {})["principal"] = principal

        if _host_name(headers.get("host", "")) not in self._allowed_hosts:
            await self._refuse(scope, receive, send, 400, "invalid host", "host")
            return

        # The page's server calls the API and sends no `Origin`; a browser always does, so any
        # request carrying one came from a page that must not talk to this API.
        if headers.get("origin") is not None:
            await self._refuse(scope, receive, send, 403, "origin refused", "origin")
            return

        if method == "POST":
            media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
            if media_type != "application/json":
                await self._refuse(scope, receive, send, 415, "json required", "content_type")
                return

        length = headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                await self._refuse(scope, receive, send, 400, "invalid content-length", "size")
                return
            if declared > self._max_body_bytes:
                await self._refuse(scope, receive, send, 413, "request too large", "size")
                return

        response_started = False

        async def guard_send(message: Any) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self._app(scope, _CappedReceive(receive, self._max_body_bytes), guard_send)
        except _BodyTooLarge:
            if not response_started:
                await self._refuse(scope, receive, send, 413, "request too large", "size")

    async def _refuse(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        status: int,
        detail: str,
        reason: str,
    ) -> None:
        # The reason code only: never the header, the token or the host.
        log.info("api_request_refused", reason=reason, status=status)
        extra = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        await JSONResponse({"detail": detail}, status_code=status, headers=extra)(
            scope, receive, send
        )
