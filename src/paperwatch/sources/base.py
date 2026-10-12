"""Shared pieces for every source client: the HTTP client factory."""

import httpx

from paperwatch.config import Settings


def make_http_client(settings: Settings) -> httpx.Client:
    # One client reused across sources; the User-Agent identifies us to source sites
    # and the timeout stops a hung source from stalling the whole weekly run.
    return httpx.Client(
        timeout=30.0,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )
