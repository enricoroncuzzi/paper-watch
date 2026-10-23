"""unit tests for the shared source HTTP client factory and retry waits."""

import http.server
import itertools
import threading
import time
from collections.abc import Callable, Iterator

import httpx
import pytest
import respx

from paperwatch.config import Settings
from paperwatch.sources.base import (
    MAX_ARXIV_BODY_BYTES,
    MAX_BODY_BYTES,
    MAX_FEED_BODY_BYTES,
    MAX_HF_BODY_BYTES,
    MAX_REDIRECTS,
    MAX_RETRY_AFTER_S,
    MAX_S2_BODY_BYTES,
    BodyTooLarge,
    RedirectRefused,
    RequestDeadlineExceeded,
    is_retryable,
    make_http_client,
    registrable_domain,
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


@pytest.mark.parametrize("status", [400, 403, 404, 501, 505])
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
    # The deadline is real, so the test keeps it tiny and the drip fast: it must end at the
    # deadline, not after the 20 chunks the server had ready.
    def chunks() -> Iterator[bytes]:
        for _ in range(20):
            time.sleep(0.02)
            yield b"x"

    client, _ = _streaming_client(chunks)
    started = time.monotonic()
    with pytest.raises(RequestDeadlineExceeded):
        send_with_retries(
            client, "GET", "https://example.test/x", request_deadline_s=0.05, attempts=1
        )
    assert time.monotonic() - started < 1.0
    # A slow server is a transport failure like a read timeout, so a later attempt may still help.
    assert is_retryable(RequestDeadlineExceeded("slow"))


def test_the_request_deadline_covers_the_response_headers() -> None:
    # httpx's read timeout restarts on every byte, so a server dripping headers (h11 allows 16 KiB
    # of them) outlasts it. The caller's own watchdog does not care what the request waits for, so
    # the answer never arrives inside the deadline and the caller stops at the deadline.
    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.5)  # stands in for headers that keep trickling in
        return httpx.Response(200, text="finally")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    started = time.monotonic()
    with pytest.raises(RequestDeadlineExceeded):
        send_with_retries(
            client, "GET", "https://example.test/x", request_deadline_s=0.05, attempts=1
        )
    assert time.monotonic() - started < 0.3  # the deadline, not the server, ended the request


def test_a_request_within_the_deadline_is_returned() -> None:
    client, _ = _streaming_client(lambda: iter([b"ok"]))
    resp = send_with_retries(client, "GET", "https://example.test/x", request_deadline_s=5.0)
    assert resp.text == "ok"


def test_the_request_deadline_bounds_the_whole_redirect_chain() -> None:
    # Hops share one budget: a chain of slow redirects ends at the deadline, instead of every hop
    # getting the whole deadline again and the hop cap being what finally stops it.
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        time.sleep(0.06)
        return httpx.Response(302, headers={"location": f"/hop{len(seen)}"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    started = time.monotonic()
    with pytest.raises(RequestDeadlineExceeded):
        send_with_retries(
            client,
            "GET",
            "https://lab.example/x",
            request_deadline_s=0.12,
            attempts=1,
        )
    # Two hops of 0.06 s use the budget, so the third is cut off there rather than walked to the
    # hop cap (which would raise RedirectRefused several deadlines later).
    assert len(seen) <= 3
    assert time.monotonic() - started < 0.6


def test_the_second_hop_may_not_leave_the_site() -> None:
    # The check runs on every hop, not only on the first: a same-site hop that redirects off the
    # site is refused, and the off-site URL is never asked for.
    client, seen = _redirecting_client(
        {
            "https://lab.example/feed.xml": httpx.Response(
                302, headers={"location": "https://lab.example/next"}
            ),
            "https://lab.example/next": httpx.Response(
                302, headers={"location": "https://evil.test/feed.xml"}
            ),
        }
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(
            client,
            "GET",
            "https://lab.example/feed.xml",
            stay_on_site="https://lab.example/feed.xml",
        )
    assert [str(url) for url in seen] == [
        "https://lab.example/feed.xml",
        "https://lab.example/next",
    ]


def test_the_second_hop_may_not_downgrade_to_plain_http() -> None:
    client, seen = _redirecting_client(
        {
            "https://lab.example/x": httpx.Response(
                302, headers={"location": "https://lab.example/y"}
            ),
            "https://lab.example/y": httpx.Response(
                302, headers={"location": "http://lab.example/y"}
            ),
        }
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(client, "GET", "https://lab.example/x")
    assert len(seen) == 2


def test_make_http_client_never_follows_redirects_itself() -> None:
    # The policy lives in send_with_retries; a client that followed hops itself would silently skip
    # every check, whatever send_with_retries asks for.
    client = make_http_client(Settings())
    try:
        assert client.follow_redirects is False
    finally:
        client.close()


def test_capped_requests_ask_for_an_uncompressed_body() -> None:
    # The cap counts the bytes held in memory, so a compressed body must not be expanded on the way
    # in: one 64 KiB chunk can decode to far more than the cap. A caller's own header still wins.
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, text="ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    send_with_retries(client, "GET", "https://example.test/x")
    send_with_retries(client, "GET", "https://example.test/x", headers={"accept-encoding": "gzip"})
    assert [headers["accept-encoding"] for headers in seen] == ["identity", "gzip"]


def test_each_source_has_its_own_body_cap() -> None:
    # A body is parsed whole after the cap, and a 49 MiB page grew RSS by about 380 MiB against the
    # unit's 950M limit, so these are the sizes that measurement supports (the numbers are on the
    # constants in `base.py`).
    assert MAX_FEED_BODY_BYTES == 10 * 1024 * 1024
    assert MAX_ARXIV_BODY_BYTES == 16 * 1024 * 1024
    assert MAX_HF_BODY_BYTES == 8 * 1024 * 1024
    assert MAX_S2_BODY_BYTES == 8 * 1024 * 1024
    # A request that names no source is still bounded by the smallest cap.
    assert (
        min(MAX_FEED_BODY_BYTES, MAX_ARXIV_BODY_BYTES, MAX_HF_BODY_BYTES, MAX_S2_BODY_BYTES)
        == MAX_BODY_BYTES
    )


def _redirecting_client(
    hops: dict[str, httpx.Response],
) -> tuple[httpx.Client, list[httpx.URL]]:
    # A transport that answers a known URL with a redirect and counts every request it served, so a
    # test can tell a followed hop from a request that never went out.
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return hops.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


# Mixed case on purpose: the strip must match these names case-insensitively.
_CALLER_HEADERS = {
    "Authorization": "Bearer secret",
    "COOKIE": "session=1",
    "x-api-key": "key",
    "X-Trace": "1",
}
_CREDENTIAL_NAMES = ("authorization", "cookie", "x-api-key")


def _credentials_in(headers: httpx.Headers) -> dict[str, str]:
    # What this request would disclose: the credential headers it actually carries.
    return {name: headers[name] for name in _CREDENTIAL_NAMES if name in headers}


def _redirect_chain(start: str, chain: list[str]) -> dict[str, httpx.Response]:
    # Every URL but the last answers with a redirect to the next one; the last is left out, so the
    # client below answers it with a plain 200.
    urls = [start, *chain]
    return {
        url: httpx.Response(302, headers={"location": next_url})
        for url, next_url in itertools.pairwise(urls)
    }


def _chain_client(
    responses: dict[str, httpx.Response],
) -> tuple[httpx.Client, list[tuple[str, httpx.Headers]]]:
    # Records every request's URL and headers in order, so a test can assert what the last one
    # carried; a URL the map does not know answers 200.
    seen: list[tuple[str, httpx.Headers]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers))
        return responses.get(str(request.url), httpx.Response(200, text="ok"))

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


@pytest.mark.parametrize(
    ("start", "chain", "kept"),
    [
        ("https://lab.example/x", ["https://lab.example/y"], True),
        ("https://lab.example/x", ["https://other.example/y"], False),
        ("https://lab.example/x", ["https://lab.example:8443/y"], False),
        ("http://lab.example/x", ["https://lab.example/y"], False),
        (
            "https://lab.example/x",
            ["https://other.example/y", "https://lab.example/z"],
            False,
        ),
    ],
    ids=["same-origin", "other-host", "other-port", "other-scheme", "back-to-the-first-origin"],
)
def test_only_a_same_origin_chain_keeps_the_credentials(
    start: str, chain: list[str], kept: bool
) -> None:
    # One chain per way an origin can change, each asserting what the LAST request carries: a rule
    # that ignored the port, ignored the scheme or restored the credentials on the way back could
    # hide behind a hop that had already stripped them. A credential belongs to the origin the
    # caller named, and leaving it is final.
    client, seen = _chain_client(_redirect_chain(start, chain))
    resp = send_with_retries(client, "GET", start, headers=dict(_CALLER_HEADERS))
    assert resp.text == "ok"
    assert [url for url, _ in seen] == [start, *chain]
    named = {"authorization": "Bearer secret", "cookie": "session=1", "x-api-key": "key"}
    assert _credentials_in(seen[-1][1]) == (named if kept else {})
    # The body cap counts the bytes held in memory, so every request asks for the body as it is:
    # the strip must not take Accept-Encoding with it.
    assert [headers["accept-encoding"] for _, headers in seen] == ["identity"] * len(seen)


def test_a_retry_after_a_cross_origin_hop_starts_over_with_the_credentials() -> None:
    # The strip is per attempt, not per call: a retry re-asks the caller's URL, which is its own
    # origin again, so it must carry what the previous attempt dropped on the way off the origin.
    client, seen = _chain_client(
        {
            "https://lab.example/x": httpx.Response(
                302, headers={"location": "https://other.example/y"}
            ),
            # The off-origin hop refuses with a retryable status, ending this attempt there.
            "https://other.example/y": httpx.Response(503),
        }
    )
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(
            client,
            "GET",
            "https://lab.example/x",
            headers=dict(_CALLER_HEADERS),
            attempts=2,
            min_wait_s=0.0,
            max_wait_s=0.0,
        )
    named = {"authorization": "Bearer secret", "cookie": "session=1", "x-api-key": "key"}
    assert [(url, _credentials_in(headers)) for url, headers in seen] == [
        ("https://lab.example/x", named),
        ("https://other.example/y", {}),
        ("https://lab.example/x", named),
        ("https://other.example/y", {}),
    ]


def test_registrable_domain_is_the_etld_plus_one() -> None:
    # One site's hosts compare equal; a shared public suffix is not a site.
    assert registrable_domain("https://blog.example.co.uk/feed.xml") == "example.co.uk"
    assert registrable_domain("https://www.example.co.uk/feed.xml") == "example.co.uk"
    assert registrable_domain("https://evil.co.uk/feed.xml") == "evil.co.uk"
    assert registrable_domain("https://a.github.io/x") == "a.github.io"
    assert registrable_domain("https://lab.example/feed.xml") == "lab.example"
    # An IP literal is its own host, not the suffix list's reading of it as a name ("0.1").
    assert registrable_domain("https://127.0.0.1/feed.xml") == "127.0.0.1"
    assert registrable_domain("https://[::1]/feed.xml") == "::1"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_every_redirect_status_is_followed_under_the_same_rules(status: int) -> None:
    # The policy is the same whatever the server calls the hop, so none of the five slips past.
    client, seen = _redirecting_client(
        {
            "https://lab.example/x": httpx.Response(status, headers={"location": "/y"}),
            "https://lab.example/y": httpx.Response(200, text="ok"),
        }
    )
    resp = send_with_retries(client, "GET", "https://lab.example/x")
    assert resp.text == "ok"
    assert len(seen) == 2


def test_a_redirect_to_a_non_http_scheme_is_refused() -> None:
    # httpx would raise UnsupportedProtocol for this, which is a transport error and so retried;
    # refusing it here makes the answer a policy decision, not a retryable blip.
    client, seen = _redirecting_client(
        {"https://lab.example/x": httpx.Response(302, headers={"location": "ftp://lab.example/x"})}
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(client, "GET", "https://lab.example/x")
    assert len(seen) == 1


def test_a_client_that_follows_redirects_itself_cannot_skip_the_policy() -> None:
    # A client built to follow hops on its own still cannot: send_with_retries asks for no
    # following, so the downgrade is refused instead of followed silently.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://lab.example/y"})

    client = httpx.Client(follow_redirects=True, transport=httpx.MockTransport(handler))
    with pytest.raises(RedirectRefused):
        send_with_retries(client, "GET", "https://lab.example/x")


def test_a_redirect_to_plain_http_is_refused() -> None:
    # The request would go out in the clear, so it must not go out at all.
    client, seen = _redirecting_client(
        {
            "https://lab.example/feed.xml": httpx.Response(
                302, headers={"location": "http://lab.example/feed.xml"}
            ),
        }
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(client, "GET", "https://lab.example/feed.xml")
    assert [str(url) for url in seen] == ["https://lab.example/feed.xml"]


def test_a_redirect_within_the_site_is_followed() -> None:
    # A feed that moves between its own hosts is still the same official feed.
    client, seen = _redirecting_client(
        {
            "https://blog.example.co.uk/feed.xml": httpx.Response(
                301, headers={"location": "https://www.example.co.uk/feed.xml"}
            ),
            "https://www.example.co.uk/feed.xml": httpx.Response(200, text="<rss/>"),
        }
    )
    resp = send_with_retries(
        client,
        "GET",
        "https://blog.example.co.uk/feed.xml",
        stay_on_site="https://blog.example.co.uk/feed.xml",
    )
    assert resp.text == "<rss/>"
    assert [str(url) for url in seen] == [
        "https://blog.example.co.uk/feed.xml",
        "https://www.example.co.uk/feed.xml",
    ]


def test_a_feed_redirect_off_its_site_is_refused() -> None:
    client, seen = _redirecting_client(
        {
            "https://lab.example/feed.xml": httpx.Response(
                302, headers={"location": "https://evil.test/feed.xml"}
            ),
        }
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(
            client,
            "GET",
            "https://lab.example/feed.xml",
            stay_on_site="https://lab.example/feed.xml",
        )
    assert [str(url) for url in seen] == ["https://lab.example/feed.xml"]


def test_a_redirect_loop_is_refused() -> None:
    client, seen = _redirecting_client(
        {"https://lab.example/x": httpx.Response(302, headers={"location": "/x"})}
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(client, "GET", "https://lab.example/x")
    assert len(seen) == MAX_REDIRECTS + 1


def test_a_refused_redirect_is_not_retried() -> None:
    # The same hop arrives next time, so a retry only repeats the refusal.
    client, seen = _redirecting_client(
        {}
        | {
            "https://lab.example/x": httpx.Response(
                302, headers={"location": "http://lab.example/x"}
            )
        }
    )
    with pytest.raises(RedirectRefused):
        send_with_retries(client, "GET", "https://lab.example/x")
    assert len(seen) == 1


def test_params_ride_on_the_first_hop_only() -> None:
    # A redirect's own Location carries the query from then on: appending the caller's params to
    # every hop would ask a different question and drop the server's token.
    client, seen = _redirecting_client(
        {
            "https://api.example.test/records?page=1&size=100": httpx.Response(
                302, headers={"location": "/records?token=abc"}
            ),
            "https://api.example.test/records?token=abc": httpx.Response(200, text="ok"),
        }
    )
    resp = send_with_retries(
        client, "GET", "https://api.example.test/records", params={"page": 1, "size": 100}
    )
    assert resp.text == "ok"
    assert [str(url) for url in seen] == [
        "https://api.example.test/records?page=1&size=100",
        "https://api.example.test/records?token=abc",
    ]


def test_a_source_that_refuses_redirects_does_not_follow_one() -> None:
    # Semantic Scholar asks with follow_redirects=False: a moved endpoint is a hop this client does
    # not take, so the answer raises instead of the target being fetched.
    client, seen = _redirecting_client(
        {
            "https://example.test/x": httpx.Response(
                302, headers={"location": "https://example.test/y"}
            ),
        }
    )
    with pytest.raises(httpx.HTTPStatusError):
        send_with_retries(client, "GET", "https://example.test/x", follow_redirects=False)
    assert [str(url) for url in seen] == ["https://example.test/x"]


class _DripServer(http.server.ThreadingHTTPServer):
    """A loopback server that answers three ways, so a test can time a real socket.

    `/drip-headers` writes the status line and then trickles header bytes without ever finishing
    the headers, `/drip-body` finishes the headers and trickles the body, and `/ok` answers with one
    complete response. Nothing leaves the loopback interface.
    """

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _DripHandler)
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        host, port = self.server_address[:2]
        self.base_url = f"http://{host}:{port}"

    def handle_error(self, request: object, client_address: object) -> None:
        # A shut-down drip resets its socket, which is what this test is about, not a failure.
        pass

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join(timeout=5)


class _DripHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path in ("/drip-headers", "/drip-body"):
            self._drip(headers_only=self.path == "/drip-headers")
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drip(self, *, headers_only: bool) -> None:
        if headers_only:
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
        else:
            self.send_response(200)
            self.send_header("Content-Length", "2000")
            self.end_headers()
        for _ in range(200):
            try:
                self.wfile.write(b"X-Drip: b\r\n" if headers_only else b"b")
                self.wfile.flush()
            except OSError:
                # The client shut the connection down: nothing left to write to.
                self.close_connection = True
                return
            time.sleep(0.02)

    def log_message(self, format: str, *args: object) -> None:
        pass  # a test needs no request log


@pytest.mark.parametrize("path", ["/drip-headers", "/drip-body"])
@pytest.mark.parametrize("reused", [False, True], ids=["fresh", "reused"])
def test_an_abandoned_request_frees_the_shared_connection(path: str, reused: bool) -> None:
    # The client the whole run shares has one pooled connection (arXiv's terms), so an abandoned
    # request that keeps it makes every later source fail with a pool timeout. Both a header drip
    # and a body drip must end at the deadline and hand the connection back before the next
    # request, whether the drip is on a fresh connection or on the pooled one the run reuses.
    server = _DripServer()
    client = httpx.Client(timeout=5.0, limits=httpx.Limits(max_connections=1))
    try:
        if reused:
            assert send_with_retries(client, "GET", f"{server.base_url}/ok").text == "ok"
        started = time.monotonic()
        with pytest.raises(RequestDeadlineExceeded):
            send_with_retries(
                client,
                "GET",
                f"{server.base_url}{path}",
                request_deadline_s=0.3,
                attempts=1,
            )
        assert time.monotonic() - started < 1.0
        # The same client, right after: the abandoned request must not still hold its connection,
        # so this returns at once instead of waiting for the drip to end or the pool to time out.
        started = time.monotonic()
        resp = send_with_retries(client, "GET", f"{server.base_url}/ok", request_deadline_s=5.0)
        assert resp.text == "ok"
        assert time.monotonic() - started < 1.0
    finally:
        client.close()
        server.close()
