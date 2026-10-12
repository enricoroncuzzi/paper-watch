"""unit tests for the shared source HTTP client factory."""

import httpx

from paperwatch.config import Settings
from paperwatch.sources.base import make_http_client


def test_make_http_client_sets_user_agent_and_timeout() -> None:
    # The User-Agent identifies us to sources; the timeout bounds a hung response.
    settings = Settings()
    client = make_http_client(settings)
    try:
        assert client.headers["User-Agent"] == settings.user_agent
        assert client.timeout == httpx.Timeout(30.0)
    finally:
        client.close()
