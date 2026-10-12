"""Shared pieces for every source client: the HTTP client factory and retries."""

from contextlib import suppress
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


def _is_retryable(exc: BaseException) -> bool:
    # Retry overload and transport blips; a deterministic 4xx is not retried.
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 503)


def _wait_for_retry(state: RetryCallState) -> float:
    # Honor Retry-After when the server sends it, otherwise back off exponentially.
    exc = state.outcome.exception() if state.outcome is not None else None
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            with suppress(ValueError):
                return float(retry_after)
    return wait_exponential(multiplier=1, min=1, max=30)(state)


def send_with_retries(client: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Send one request, retrying 429/503 and transport errors, and raise on failure.

    Other 4xx responses fail on the first attempt. After three attempts the last
    exception is re-raised so the caller can degrade visibly.
    """
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=_wait_for_retry,
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
    # Unreachable: tenacity re-raises after the final attempt.
    raise RuntimeError("retry loop exhausted without a result or an exception")
