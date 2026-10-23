"""Shared pieces for every source client: the HTTP client factory and retries."""

import math
import time
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from typing import Any

import httpx
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

# A body is read into memory whole before it is parsed, so a broken or hostile server must not be
# able to make a run allocate gigabytes: a real lab feed is under a megabyte even with full post
# bodies, and a 500-record arXivRaw OAI page or a 500-id Semantic Scholar batch answer is a few
# megabytes. Over the cap is a fetch failure for that source, dead-lettered like any other, and
# never an OOM that kills a process sharing the machine.
MAX_FEED_BODY_BYTES = 10 * 1024 * 1024
# The API cap covers arXiv, Hugging Face and Semantic Scholar, whose pages carry the most bytes.
MAX_API_BODY_BYTES = 50 * 1024 * 1024
# The 30 s read timeout bounds one read, not one request: a server that drips a byte every 20 s
# never trips it and could hold a run open until the unit's 17 h timeout. This deadline covers a
# whole request (connect plus every read), on top of the read timeout that catches a silent stall.
REQUEST_DEADLINE_S = 120.0


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
    return httpx.Client(
        timeout=30.0,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
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


def _read_body_capped(
    resp: httpx.Response, *, max_bytes: int, deadline: float, clock: Callable[[], float]
) -> bytes:
    """Read a streamed body, aborting as soon as it passes the cap or the deadline.

    Bytes are counted as they arrive and the read is abandoned at the first byte over the cap, so
    an endless body costs the cap and not the run's memory. The deadline is checked on every read
    chunk, which the 30 s read timeout makes at least as often as it fires.
    """
    body = bytearray()
    for chunk in resp.iter_bytes():
        body += chunk
        if len(body) > max_bytes:
            raise BodyTooLarge(f"{resp.request.url} sent more than {max_bytes} bytes")
        if clock() > deadline:
            raise RequestDeadlineExceeded(
                f"{resp.request.url} took longer than {REQUEST_DEADLINE_S:.0f} s"
            )
    return bytes(body)


_DECODED_HEADERS = frozenset({"content-encoding", "content-length"})


def _body_headers(resp: httpx.Response) -> httpx.Headers:
    # The body was already decoded by the stream, so the encoding and length the server sent no
    # longer describe it; everything else (Retry-After, content-type) the callers read is kept.
    return httpx.Headers(
        [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in _DECODED_HEADERS]
    )


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
    max_bytes: int = MAX_API_BODY_BYTES,
    request_deadline_s: float = REQUEST_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
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
    raises `BodyTooLarge` and one that outlives the deadline raises `RequestDeadlineExceeded`, so
    the returned response is always complete in memory and bounded.
    """
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
            # The deadline covers this request only: a pacing wait before it is not the server's.
            deadline = clock() + request_deadline_s
            with client.stream(method, url, **kwargs) as resp:
                body = _read_body_capped(resp, max_bytes=max_bytes, deadline=deadline, clock=clock)
                # Rebuilt as a plain response so callers keep using .text/.json().content on an
                # already-read body, and raise_for_status still carries the server's headers.
                result = httpx.Response(
                    resp.status_code,
                    headers=_body_headers(resp),
                    content=body,
                    request=resp.request,
                )
            result.raise_for_status()
            return result
    # Unreachable: tenacity re-raises after the final attempt.
    raise RuntimeError("retry loop exhausted without a result or an exception")
