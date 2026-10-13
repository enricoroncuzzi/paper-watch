"""Shared pieces for every source client: the HTTP client factory and retries."""

import math
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
    **kwargs: Any,
) -> httpx.Response:
    """Send one request, retrying 429/503 and transport errors, and raise on failure.

    Other 4xx responses fail on the first attempt. The defaults (three attempts, a 1 s
    exponential floor capped at 30 s) keep every source that does not override them;
    Semantic Scholar asks for five attempts from a 5 s floor capped at 60 s. Retry-After
    is honored when present. After the last attempt the last exception is re-raised so the
    caller can degrade visibly.
    """
    for attempt in Retrying(
        stop=stop_after_attempt(attempts),
        wait=partial(_wait_for_retry, min_wait_s=min_wait_s, max_wait_s=max_wait_s),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
    # Unreachable: tenacity re-raises after the final attempt.
    raise RuntimeError("retry loop exhausted without a result or an exception")
