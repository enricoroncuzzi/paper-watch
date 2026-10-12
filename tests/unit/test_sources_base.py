"""unit tests for the shared source HTTP client factory and retry waits."""

import time

import httpx
import respx

from paperwatch.config import Settings
from paperwatch.sources.base import MAX_RETRY_AFTER_S, make_http_client, send_with_retries


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
