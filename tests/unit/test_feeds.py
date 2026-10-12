"""unit tests for the tier-3 official lab feed source."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

from paperwatch.sources.feeds import Feed, FeedSource, load_feeds

FIX = Path("tests/fixtures/feed_lab.xml").read_text()
FEED = Feed(lab="TestLab", url="https://lab.example/feed.xml", region="us-eu")


@respx.mock
def test_items_inside_window_only() -> None:
    # The fixture holds three real items, one dated before the window below.
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=FIX))
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert failures == []
    assert len(items) == 2
    assert all(i.source == "feed" and i.licence == "copyright (private search only)" for i in items)
    assert all(i.authors == ("TestLab",) for i in items)


@respx.mock
def test_broken_feed_is_reported_not_raised() -> None:
    # One down feed must not abort the others; the failure carries the feed URL.
    respx.get(FEED.url).mock(return_value=httpx.Response(500))
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert items == [] and failures[0][0] == FEED.url


def test_load_feeds_reads_the_configured_list() -> None:
    # The one verified feed is present; the rest of the list arrives later.
    feeds = load_feeds(Path("config/feeds.yaml"))
    assert any(f.lab == "Hugging Face" and "huggingface.co" in f.url for f in feeds)
    assert all(f.region for f in feeds)
