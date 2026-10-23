"""Shared pieces for every source client: the HTTP client factory and retries."""

import ipaddress
import math
import socket
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from typing import Any
from urllib.parse import urlparse

import httpx
from publicsuffixlist import PublicSuffixList
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from paperwatch.config import Settings

# A server's Retry-After cannot be allowed to stall a run for hours.
MAX_RETRY_AFTER_S = 60.0

# Statuses a later attempt can clear. 429 and 503 are the server itself saying "not now"; 500,
# 502 and 504 are the same outage seen through a proxy or a dying backend (a gateway that timed
# out waiting, a worker that dropped the connection), which source sites send during congestion
# as readily as a 503. Every other 4xx answers this exact request: a bug, never a retry.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# A body is held in memory whole and then parsed whole, so these caps are a memory bound, not only a
# transfer bound: measured on 2026-10-23, one 49 MiB arXivRaw page (21 360 records) grew RSS by
# about 380 MiB, while an ingestion peaks near 707 MB against the unit's MemoryMax=950M, leaving
# about 243 MB. Each cap is therefore about 5x to 10x the real page size (an arXiv day page of a few
# MB, a 500-id Semantic Scholar answer or a 100-row Hugging Face page under 1 MB), which keeps the
# parse growth under about 130 MiB. Over the cap is a fetch failure for that source, dead-lettered
# like any other, and never an OOM that kills a process sharing the machine.
MAX_FEED_BODY_BYTES = 10 * 1024 * 1024
MAX_ARXIV_BODY_BYTES = 16 * 1024 * 1024
MAX_HF_BODY_BYTES = 8 * 1024 * 1024
MAX_S2_BODY_BYTES = 8 * 1024 * 1024
# A request that names no source gets the smallest cap: a cap is a bound, never a promise.
MAX_BODY_BYTES = min(
    MAX_FEED_BODY_BYTES, MAX_ARXIV_BODY_BYTES, MAX_HF_BODY_BYTES, MAX_S2_BODY_BYTES
)
# The 30 s read timeout bounds one read, not one request: a server that drips a byte every 20 s
# never trips it, and h11 allows a 16 KiB response header, so a drip at the header stage could hold
# a run for days. This deadline covers the whole request (connect, headers and body) and is
# enforced from outside the request: see `_send_once`.
REQUEST_DEADLINE_S = 120.0

# Redirects are followed by hand, never by httpx, so a target can be checked before the request
# goes out: a hop to plain http would send the request in the clear, and a feed that redirects off
# its own site is no longer the official feed that was configured. Five hops is more than any real
# feed or API needs, and it stops a redirect loop instead of spending the deadline on it.
MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# The public suffix list ships inside the package as a snapshot (refreshed with the dependency), so
# nothing here does a DNS lookup or a download at run time, and a run never depends on the network
# to decide whether a host is the configured feed's own site.
_PUBLIC_SUFFIXES = PublicSuffixList()


class RedirectRefused(httpx.HTTPError):
    """A redirect this client will not follow: a downgrade, another site, or too many hops.

    Deliberately not an `httpx.TransportError`: the same redirect will arrive next time, so a
    retry would only repeat the refusal.
    """


def registrable_domain(url: str) -> str:
    """The eTLD+1 of a URL's host, so one site's own hosts compare equal.

    `blog.example.co.uk` and `www.example.co.uk` both give `example.co.uk`, while `evil.co.uk`
    gives `evil.co.uk` and `a.github.io` gives `a.github.io` (a shared public suffix is not a
    site). A host the suffix list does not know is its own domain, and an IP literal is returned
    as it is: the suffix list would read `127.0.0.1` as the name `0.1`.
    """
    host = (urlparse(url).hostname or "").lower()
    if not host or _is_ip_literal(host):
        return host
    return _PUBLIC_SUFFIXES.privatesuffix(host) or host


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _redirect_target(current: httpx.URL, location: str, *, allowed: str | None) -> httpx.URL:
    """The URL a redirect points at, or `RedirectRefused` if this client will not follow it."""
    target = current.join(location)
    if target.scheme not in ("http", "https"):
        raise RedirectRefused(f"{current} redirected to a non-HTTP target: {target}")
    if current.scheme == "https" and target.scheme == "http":
        raise RedirectRefused(f"{current} redirected to plain http: {target}")
    if allowed is not None and registrable_domain(str(target)) != allowed:
        raise RedirectRefused(f"{current} redirected off its site ({allowed}): {target}")
    return target


class BodyTooLarge(httpx.HTTPError):
    """A response body above the per-source cap: a fetch failure for that source, never an OOM.

    Deliberately not an `httpx.TransportError`, so the retry policy does not spend attempts on a
    body that will be exactly as large next time.
    """


class RequestDeadlineExceeded(httpx.TimeoutException):
    """One request outlived `REQUEST_DEADLINE_S`: a slow server, retried like a read timeout."""


def make_http_client(settings: Settings) -> httpx.Client:
    # One client reused across sources; the User-Agent identifies us to source sites
    # and the timeout stops a hung source from stalling the whole weekly run.
    # arXiv's terms allow one connection per client, so the pool is capped at one.
    # Redirects are not followed here but in `send_with_retries`, which can check each target.
    return httpx.Client(
        timeout=30.0,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )


def is_retryable(exc: BaseException) -> bool:
    """Whether a later attempt can clear this failure: overload, outage or transport blip.

    Public because a caller that ran out of attempts must tell "the server refused us" (the
    request is still due) from "this request is wrong" (a deterministic 4xx is a bug, never a
    retry).
    """
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRYABLE_STATUSES


def _read_body_capped(resp: httpx.Response, *, max_bytes: int) -> bytes:
    """Read a streamed body, aborting as soon as it passes the cap.

    Bytes are counted as they arrive and the read is abandoned at the first byte over the cap, so
    an endless body costs the cap and not the run's memory. The stream's own chunks are joined once
    at the end, so the peak is the body plus that one joined copy, not a body built up and copied
    again from scratch.
    """
    chunks: list[bytes] = []
    size = 0
    for chunk in resp.iter_bytes():
        size += len(chunk)
        if size > max_bytes:
            raise BodyTooLarge(f"{resp.request.url} sent more than {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


_DECODED_HEADERS = frozenset({"content-encoding", "content-length"})


def _body_headers(resp: httpx.Response) -> httpx.Headers:
    # The body was already decoded by the stream, so the encoding and length the server sent no
    # longer describe it; everything else (Retry-After, content-type) the callers read is kept.
    return httpx.Headers(
        [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in _DECODED_HEADERS]
    )


# The connection each client holds per origin, remembered from the trace hook below. httpcore
# reports the network stream only when it *creates* a connection, and the run reuses one pooled
# connection across every request (arXiv's terms cap the pool at one), so without this a request
# that lands on a reused connection would have no handle on the socket a header drip is stuck on.
_CONNECTIONS: weakref.WeakKeyDictionary[httpx.Client, dict[str, Any]] = weakref.WeakKeyDictionary()
_CONNECTIONS_LOCK = threading.Lock()


def _origin(url: httpx.URL) -> str:
    return f"{url.scheme}://{url.host}:{url.port}"


# The credentials a hop to another origin must not carry: Authorization, which httpx also drops
# across origins, plus Cookie and x-api-key. httpx never drops x-api-key, and it drops Cookie on
# every hop, same-origin included, so it is the origin check here that keeps a caller's Cookie on a
# same-origin hop. `send_with_retries` follows hops by hand, so it does this itself, and only when
# the origin changes.
_CROSS_ORIGIN_CREDENTIALS = ("authorization", "cookie", "x-api-key")


def _without_credentials(headers: httpx.Headers) -> httpx.Headers:
    """The same headers minus the credentials, whatever case the caller wrote them in."""
    # Copied, not rebuilt, so the headers that stay keep their own casing and any repetition.
    kept = httpx.Headers(headers)
    for name in _CROSS_ORIGIN_CREDENTIALS:
        kept.pop(name, None)
    return kept


def _remember_connection(client: httpx.Client, origin: str, network: Any) -> None:
    with _CONNECTIONS_LOCK:
        _CONNECTIONS.setdefault(client, {})[origin] = network


def _pooled_connection(client: httpx.Client, origin: str) -> Any:
    with _CONNECTIONS_LOCK:
        return _CONNECTIONS.get(client, {}).get(origin)


class _InFlight:
    """One request of one attempt, shared between the caller and the worker running it."""

    def __init__(self) -> None:
        # The response as it arrives (its body unread) and the one handed back to the caller.
        self.streamed: httpx.Response | None = None
        self.response: httpx.Response | None = None
        self.error: BaseException | None = None
        # The connection's network stream, handed over by the trace hook below.
        self.network: Any = None
        self.abandoned = threading.Event()

    def abandon(self) -> None:
        # Tell the worker to stop, then break the connection itself. A header drip leaves the worker
        # blocked in `recv` with no response object to close, and that is the one case that would
        # otherwise keep the client's single pooled connection for hours; shutting the socket down
        # wakes the read at once on both Linux and macOS (closing alone does not).
        self.abandoned.set()
        network = self.network
        if network is not None:
            with suppress(Exception):
                network.get_extra_info("socket").shutdown(socket.SHUT_RDWR)
        streamed = self.streamed
        if streamed is not None:
            with suppress(Exception):
                streamed.close()


def _send_once(
    client: httpx.Client,
    method: str,
    target: httpx.URL,
    *,
    kwargs: dict[str, Any],
    max_bytes: int,
    join_s: float,
    deadline_s: float,
) -> httpx.Response:
    """Send one request and read its body, giving up `join_s` seconds after starting it.

    None of httpx's timeouts bounds a whole request: the read timeout restarts on every byte, so a
    server dripping response headers (h11 allows 16 KiB of them) or body bytes outlasts all of them.
    So the request runs in a worker thread while the caller waits `join_s`, and on that timeout the
    caller shuts its connection down and closes the response the worker is reading, then raises
    `RequestDeadlineExceeded`. An abandoned request therefore stops at the deadline and hands its
    pool slot back instead of getting in the way of the next source. `join_s` is the wall-clock
    budget left to this request, so the hops of one redirect chain share it.
    """
    state = _InFlight()
    origin = _origin(target)
    # The connection this client already holds for the target, which the request will reuse if it
    # can; a new one is reported by the trace hook before the headers start arriving.
    state.network = _pooled_connection(client, origin)

    def trace(event: str, info: dict[str, Any]) -> None:
        # httpcore reports the connection's network stream as soon as it exists, which is before the
        # headers arrive; that handle is the only way to break a request that is stuck there.
        if event in ("connection.connect_tcp.complete", "connection.start_tls.complete"):
            _remember_connection(client, origin, info["return_value"])
            state.network = info["return_value"]
            if state.abandoned.is_set():
                state.abandon()

    def work() -> None:
        try:
            # Redirects are never followed here: the caller checks every target itself, so a client
            # built with follow_redirects=True cannot make this skip the policy by accident.
            with client.stream(
                method,
                target,
                follow_redirects=False,
                extensions={"trace": trace},
                **kwargs,
            ) as resp:
                state.streamed = resp
                if state.abandoned.is_set():
                    return
                body = _read_body_capped(resp, max_bytes=max_bytes)
                if state.abandoned.is_set():
                    return
                # Rebuilt as a plain response so callers keep using .text/.json()/.content on an
                # already-read body, and raise_for_status still carries the server's headers.
                state.response = httpx.Response(
                    resp.status_code,
                    headers=_body_headers(resp),
                    content=body,
                    request=resp.request,
                )
        except BaseException as exc:  # noqa: BLE001 - handed back for the caller to re-raise
            state.error = exc

    # A daemon thread, so a worker still waiting on a dead socket never holds the process open.
    worker = threading.Thread(target=work, name="paperwatch-http", daemon=True)
    worker.start()
    worker.join(join_s)
    if worker.is_alive():
        state.abandon()
        raise RequestDeadlineExceeded(f"{target} took longer than {deadline_s:.0f} s")
    if state.error is not None:
        raise state.error
    if state.response is None:
        # Only reachable if the worker saw the abandonment flag, which only the timeout sets.
        raise RequestDeadlineExceeded(f"{target} took longer than {deadline_s:.0f} s")
    return state.response


def _wait_for_retry(state: RetryCallState, *, min_wait_s: float, max_wait_s: float) -> float:
    # Honor a sane Retry-After, otherwise back off exponentially between the given bounds.
    exc = state.outcome.exception() if state.outcome is not None else None
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            with suppress(ValueError):
                seconds = float(retry_after)
                # Negative, NaN and infinite values are not usable waits.
                if math.isfinite(seconds) and seconds >= 0:
                    return min(seconds, MAX_RETRY_AFTER_S)
    return wait_exponential(multiplier=min_wait_s, min=min_wait_s, max=max_wait_s)(state)


def send_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    attempts: int = 3,
    min_wait_s: float = 1.0,
    max_wait_s: float = 30.0,
    max_bytes: int = MAX_BODY_BYTES,
    request_deadline_s: float = REQUEST_DEADLINE_S,
    follow_redirects: bool = True,
    stay_on_site: str | None = None,
    before_request: Callable[[], None] | None = None,
    before_sleep: Callable[[RetryCallState], None] | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Send one request, retrying a retryable status or a transport error, and raise on failure.

    `RETRYABLE_STATUSES` (429, 500, 502, 503, 504) and transport errors are retried under the
    same bounded policy; any other status, a 4xx in particular, fails on the first attempt. The
    defaults (three attempts, a 1 s exponential floor capped at 30 s) keep every source that does
    not override them; Semantic Scholar asks for five attempts from a 5 s floor capped at 60 s.
    Retry-After is honored when present. After the last attempt the last exception is re-raised
    so the caller can degrade visibly. `before_sleep` sees the tenacity state before every wait,
    so a caller can log each refusal with the wait it is about to take.

    The body is streamed and read here, never by a bare `client.request`: a body over `max_bytes`
    raises `BodyTooLarge`, and `request_deadline_s` bounds the whole request from connect to the
    last body byte, headers included, raising `RequestDeadlineExceeded` (see `_send_once`). The
    returned response is therefore always complete in memory and bounded.

    Compressed bodies are asked for uncompressed (`Accept-Encoding: identity`), so the cap counts
    the bytes that are actually held; a server that compresses anyway is still capped, on the
    decoded size.

    A redirect is followed only under this client's own rules (see `_redirect_target`): never to
    plain http, and with `stay_on_site` (a URL) given, never off that URL's registrable domain.
    `follow_redirects=False` refuses every redirect, which raises like any other answer this client
    will not accept, so a caller never parses a page from a hop it did not agree to take. A hop to
    another origin (scheme, host or port) drops the caller's credentials first. Credentials must
    therefore travel in `headers`: an `auth=` or `cookies=` kwarg rides every hop untouched.
    """
    # The caller's params ride on the first hop only: a later hop's own Location carries them, so
    # they are folded into the URL once, and a retry asks for exactly the same URL as the first try.
    params = kwargs.pop("params", None)
    first_target = httpx.URL(url)
    if params is not None:
        first_target = first_target.copy_merge_params(params)
    allowed = registrable_domain(stay_on_site) if stay_on_site is not None else None
    # Ask for the bytes as they are (see the docstring): a caller's own header wins.
    headers = httpx.Headers(kwargs.pop("headers", None))
    headers.setdefault("Accept-Encoding", "identity")
    for attempt in Retrying(
        stop=stop_after_attempt(attempts),
        wait=partial(_wait_for_retry, min_wait_s=min_wait_s, max_wait_s=max_wait_s),
        retry=retry_if_exception(is_retryable),
        before_sleep=before_sleep,
        reraise=True,
    ):
        with attempt:
            if before_request is not None:
                # A caller that must pace every attempt (arXiv's one-request-per-3 s
                # rule) hooks in here, so a retry is paced too, not only the first try.
                before_request()
            # The deadline covers this request only: a pacing wait before it is not the server's,
            # and a redirect chain is one request, so the hops share the budget left by the others.
            started = time.monotonic()
            target = first_target
            hop_headers = headers
            hops = 0
            while True:
                # Each hop gets what is left of one attempt's budget, so a chain cannot multiply
                # the deadline, and a spent budget ends the chain on the very next hop.
                result = _send_once(
                    client,
                    method,
                    target,
                    kwargs={**kwargs, "headers": hop_headers},
                    max_bytes=max_bytes,
                    join_s=request_deadline_s - (time.monotonic() - started),
                    deadline_s=request_deadline_s,
                )
                location = result.headers.get("location")
                if not (follow_redirects and result.status_code in _REDIRECT_STATUSES and location):
                    result.raise_for_status()
                    return result
                hops += 1
                if hops > MAX_REDIRECTS:
                    raise RedirectRefused(f"{url} redirected more than {MAX_REDIRECTS} times")
                # The sources that follow redirects only ever GET, so the method is kept; a 303
                # from a POST source (S2, which refuses redirects) never reaches this line.
                next_target = _redirect_target(target, location, allowed=allowed)
                # Credentials belong to the origin the caller named, so another origin (a new
                # scheme, host or port) gets the headers without them, and a later hop back does
                # not get them again.
                if _origin(next_target) != _origin(target):
                    hop_headers = _without_credentials(hop_headers)
                target = next_target
    # Unreachable: tenacity re-raises after the final attempt.
    raise RuntimeError("retry loop exhausted without a result or an exception")
