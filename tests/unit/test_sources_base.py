"""unit tests for the shared source HTTP client factory and retry waits."""

import time
from collections.abc import Callable, Iterator

import httpx
import pytest
import respx

from paperwatch.config import Settings
from paperwatch.sources.base import (
    MAX_RETRY_AFTER_S,
    BodyTooLarge,
    RequestDeadlineExceeded,
    is_retryable,
    make_http_client,
    send_with_retries,
)


def test_make_http_client_sets_user_agent_and_timeout() -> None:
    # The User-Agent identifies us to sources; the timeout bounds a hung response.
    settings = Settings()
    client = make_http_client(settings)
    try:
        assert client.headers["User-Agent"] == settings.user_agent
        assert client.timeout == httpx.Timeout(30.0)
    finally:
        client.close()


def _sleeps_for_retry_after(monkeypatch, retry_after: str) -> list[float]:
    # Two 503s then a 200: capture the wait before each retry attempt.
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    respx.get("https://example.test/x").mock(
        side_effect=[
            httpx.Response(503, headers={"Retry-After": retry_after}),
            httpx.Response(503, headers={"Retry-After": retry_after}),
            httpx.Response(200),
        ]
    )
    send_with_retries(httpx.Client(), "GET", "https://example.test/x")
    return sleeps


@respx.mock
def test_retry_after_normal_value_is_honored(monkeypatch) -> None:
    assert _sleeps_for_retry_after(monkeypatch, "7") == [7.0, 7.0]


@respx.mock
def test_retry_after_huge_value_is_capped(monkeypatch) -> None:
    assert _sleeps_for_retry_after(monkeypatch, "9999") == [
        MAX_RETRY_AFTER_S,
        MAX_RETRY_AFTER_S,
    ]


@respx.mock
def test_retry_after_negative_falls_back_to_backoff(monkeypatch) -> None:
    assert _sleeps_for_retry_after(monkeypatch, "-5") == [1.0, 2.0]


@respx.mock
def test_retry_after_nan_falls_back_to_backoff(monkeypatch) -> None:
    assert _sleeps_for_retry_after(monkeypatch, "nan") == [1.0, 2.0]


@respx.mock
def test_retry_after_infinite_falls_back_to_backoff(monkeypatch) -> None:
    assert _sleeps_for_retry_after(monkeypatch, "inf") == [1.0, 2.0]


@respx.mock
def test_retry_after_http_date_falls_back_to_backoff(monkeypatch) -> None:
    assert _sleeps_for_retry_after(monkeypatch, "Wed, 21 Oct 2026 07:28:00 GMT") == [1.0, 2.0]


@respx.mock
def test_send_with_retries_defaults_to_three_attempts(monkeypatch) -> None:
    # Other sources keep today's behaviour unless they ask for more attempts.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.get("https://example.test/x").mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(httpx.Client(), "GET", "https://example.test/x")
    assert route.call_count == 3


@respx.mock
def test_send_with_retries_honors_a_higher_attempt_floor(monkeypatch) -> None:
    # The S2 client asks for five attempts and a 5 s floor; here only the limits matter.
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    route = respx.get("https://example.test/x").mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(
            httpx.Client(),
            "GET",
            "https://example.test/x",
            attempts=5,
            min_wait_s=5.0,
            max_wait_s=60.0,
        )
    assert route.call_count == 5
    assert sleeps == [5.0, 10.0, 20.0, 40.0]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
@respx.mock
def test_a_retryable_status_gets_every_attempt(monkeypatch, status: int) -> None:
    # A throttle (429/503) and a fronted outage (500/502/504) are the same "come back later" to
    # us, so they share the one bounded policy.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.get("https://example.test/x").mock(return_value=httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(httpx.Client(), "GET", "https://example.test/x")
    assert route.call_count == 3


@pytest.mark.parametrize("status", [400, 403, 404])
@respx.mock
def test_a_deterministic_4xx_is_a_bug_not_a_retry(monkeypatch, status: int) -> None:
    # The request itself is wrong: asking again would only spend the same refusal.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.get("https://example.test/x").mock(return_value=httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(httpx.Client(), "GET", "https://example.test/x")
    assert route.call_count == 1


def _streaming_client(
    chunk_source: Callable[[], Iterator[bytes]],
) -> tuple[httpx.Client, list[httpx.Request]]:
    # A transport that hands the body over chunk by chunk, so a test can watch the read stop, and
    # that counts the requests it saw, so a test can tell one attempt from a retry.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=chunk_source())

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_a_body_over_the_cap_is_aborted_mid_stream() -> None:
    # The cap is a memory bound, not a size check after the fact: the read stops at the first byte
    # over it, so an endless body costs the cap and never the whole run's memory.
    pulled: list[int] = []

    def chunks() -> Iterator[bytes]:
        for i in range(10):
            pulled.append(i)
            yield b"x" * 1024

    client, _requests = _streaming_client(chunks)
    with pytest.raises(BodyTooLarge):
        send_with_retries(client, "GET", "https://example.test/x", max_bytes=4 * 1024)
    assert len(pulled) == 5  # the fifth kilobyte went over the 4 KiB cap and nothing followed


def test_a_body_exactly_at_the_cap_is_read() -> None:
    client, _ = _streaming_client(lambda: iter([b"x" * 512, b"x" * 512]))
    resp = send_with_retries(client, "GET", "https://example.test/x", max_bytes=1024)
    assert resp.content == b"x" * 1024


def test_a_body_over_the_cap_is_not_retried(monkeypatch) -> None:
    # The same body would arrive again, so a retry only spends time on a deterministic answer.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    client, requests = _streaming_client(lambda: iter([b"x" * 4096]))
    with pytest.raises(BodyTooLarge):
        send_with_retries(client, "GET", "https://example.test/x", max_bytes=1024)
    assert len(requests) == 1


def test_a_slow_drip_body_trips_the_request_deadline() -> None:
    # The 30 s read timeout bounds one read, so a server dripping a chunk at a time never trips it.
    # The clock is injected and advanced per read, so no test ever really waits.
    client, _ = _streaming_client(lambda: iter([b"x"] * 10))
    now = [0.0]

    def clock() -> float:
        now[0] += 40.0
        return now[0]

    with pytest.raises(RequestDeadlineExceeded):
        send_with_retries(
            client,
            "GET",
            "https://example.test/x",
            request_deadline_s=120.0,
            clock=clock,
            attempts=1,
        )
    # A slow server is a transport failure like a read timeout, so a later attempt may still help.
    assert is_retryable(RequestDeadlineExceeded("slow"))


def test_a_request_within_the_deadline_is_returned() -> None:
    client, _ = _streaming_client(lambda: iter([b"ok"]))
    now = [0.0]

    def clock() -> float:
        now[0] += 10.0
        return now[0]

    resp = send_with_retries(client, "GET", "https://example.test/x", clock=clock)
    assert resp.text == "ok"
