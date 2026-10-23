"""unit tests for the tier-3 official lab feed source."""

import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from paperwatch.domain import RejectedEntry
from paperwatch.sources.feeds import Feed, FeedSource, load_feeds

FIX = Path("tests/fixtures/feed_lab.xml").read_text()
FEED = Feed(lab="TestLab", url="https://lab.example/feed.xml", region="us-eu")
OTHER = Feed(lab="Other", url="https://other.example/feed.xml", region="us")
SINCE = datetime(2026, 1, 1, tzinfo=UTC)
UNTIL = datetime(2030, 1, 1, tzinfo=UTC)


def _item(guid: str, title: str | None, pub: str | None, link: str | None = None) -> str:
    # Minimal RSS item; isPermaLink=false keeps the guid independent of the link.
    parts = [f'<guid isPermaLink="false">{guid}</guid>']
    if title is not None:
        parts.append(f"<title>{title}</title>")
    if pub is not None:
        parts.append(f"<pubDate>{pub}</pubDate>")
    if link is not None:
        parts.append(f"<link>{link}</link>")
    return "<item>" + "".join(parts) + "</item>"


def _rss(*items: str) -> str:
    # Minimal wrapper so feedparser sees an RSS root.
    return '<rss version="2.0"><channel>' + "".join(items) + "</channel></rss>"


@respx.mock
def test_items_inside_window_only() -> None:
    # The fixture holds three real items, one dated before the window below.
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=FIX))
    src = FeedSource(httpx.Client(), [FEED])
    items, failures = src.fetch(SINCE, UNTIL)
    assert failures == []
    assert len(items) == 2
    ids = {i.external_id for i in items}
    # The feed URL scopes each id so two labs cannot collide on a shared guid.
    assert f"{FEED.url} https://huggingface.co/blog/allenai/impactful-scheduling" in ids
    assert f"{FEED.url} https://huggingface.co/blog/building-with-ml-intern" in ids
    assert all("open-d1" not in i.external_id for i in items)
    assert all(i.source == "feed" and i.licence == "copyright (private search only)" for i in items)
    assert all(i.authors == ("TestLab",) for i in items)
    text = next(i.text for i in items if i.external_id.endswith("impactful-scheduling"))
    # HTML tags are stripped, entities decoded, and the image alt text kept.
    assert text == "Hello & goodbye link a diagram"
    assert src.rejected == []


@respx.mock
def test_broken_feed_is_reported_not_raised() -> None:
    # One down feed must not abort the others; the failure carries the feed URL.
    respx.get(FEED.url).mock(return_value=httpx.Response(500))
    respx.get(OTHER.url).mock(return_value=httpx.Response(200, text=FIX))
    items, failures = FeedSource(httpx.Client(), [FEED, OTHER]).fetch(SINCE, UNTIL)
    assert failures[0][0] == FEED.url
    assert len(items) == 2
    assert all(i.authors == ("Other",) for i in items)


@respx.mock
def test_non_feed_200_is_reported_and_other_feeds_continue() -> None:
    # A maintenance HTML page is not a quiet day; it must not advance a cursor.
    respx.get(FEED.url).mock(
        return_value=httpx.Response(
            200, text="<html><body>maintenance</body></html>", headers={"content-type": "text/html"}
        )
    )
    respx.get(OTHER.url).mock(return_value=httpx.Response(200, text=FIX))
    items, failures = FeedSource(httpx.Client(), [FEED, OTHER]).fetch(SINCE, UNTIL)
    assert failures[0][0] == FEED.url
    assert len(items) == 2
    assert all(i.authors == ("Other",) for i in items)


@respx.mock
def test_error_document_200_is_reported() -> None:
    # A 200 error document that is not a feed is treated the same as a down feed.
    respx.get(FEED.url).mock(
        return_value=httpx.Response(200, text="<?xml version='1.0'?><error>nope</error>")
    )
    respx.get(OTHER.url).mock(return_value=httpx.Response(200, text=FIX))
    items, failures = FeedSource(httpx.Client(), [FEED, OTHER]).fetch(SINCE, UNTIL)
    assert failures[0][0] == FEED.url
    assert len(items) == 2


@respx.mock
def test_empty_feed_is_a_success_with_no_items() -> None:
    # An empty but well-formed feed is a valid, if quiet, day.
    respx.get(FEED.url).mock(
        return_value=httpx.Response(200, text='<rss version="2.0"><channel></channel></rss>')
    )
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(SINCE, UNTIL)
    assert items == [] and failures == []


@respx.mock
def test_fetch_one_records_the_newest_entry_date_outside_the_window() -> None:
    # A feed that answers but has not published lately must be visible as stale.
    respx.get(FEED.url).mock(
        return_value=httpx.Response(
            200,
            text=_rss(
                _item("a", "Old post", "Mon, 01 Jun 2026 10:00:00 +0000", "https://lab.example/a"),
                _item(
                    "b", "Older post", "Fri, 01 May 2026 10:00:00 +0000", "https://lab.example/b"
                ),
            ),
        )
    )
    src = FeedSource(httpx.Client(), [FEED])
    items = src.fetch_one(FEED, datetime(2026, 10, 1, tzinfo=UTC), UNTIL)
    assert items == []
    assert src.newest == {FEED.url: datetime(2026, 6, 1, 10, 0, tzinfo=UTC)}


@respx.mock
def test_bad_entry_is_rejected_and_other_entries_continue() -> None:
    # A title-less entry is isolated; the good entries and the next feed survive.
    doc = _rss(
        _item(
            "https://lab.example/1", "One", "Fri, 09 Oct 2026 10:00:00 GMT", "https://lab.example/1"
        ),
        _item(
            "https://lab.example/2", None, "Fri, 09 Oct 2026 11:00:00 GMT", "https://lab.example/2"
        ),
        _item(
            "https://lab.example/3",
            "Three",
            "Fri, 09 Oct 2026 12:00:00 GMT",
            "https://lab.example/3",
        ),
    )
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    respx.get(OTHER.url).mock(return_value=httpx.Response(200, text=FIX))
    src = FeedSource(httpx.Client(), [FEED, OTHER])
    items, failures = src.fetch(SINCE, UNTIL)
    assert failures == []
    assert len([i for i in items if i.authors == ("TestLab",)]) == 2
    assert len([i for i in items if i.authors == ("Other",)]) == 2
    assert len(src.rejected) == 1
    assert src.rejected[0].source == "feed"
    assert "https://lab.example/2" in src.rejected[0].ref
    assert src.rejected[0].error


@respx.mock
def test_timezone_less_pubdate_is_treated_as_utc() -> None:
    # RFC 822 without a zone is the lab's clock; treat it as UTC rather than drop it.
    doc = _rss(
        _item("https://lab.example/9", "Nine", "Fri, 09 Oct 2026 15:20:29", "https://lab.example/9")
    )
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    src = FeedSource(httpx.Client(), [FEED])
    items, failures = src.fetch(SINCE, UNTIL)
    assert failures == []
    assert len(items) == 1
    assert items[0].published_at == datetime(2026, 10, 9, 15, 20, 29, tzinfo=UTC)
    assert src.rejected == []


@respx.mock
def test_undated_entry_is_rejected() -> None:
    # No parsed stamp and no raw date string: reject the entry, never skip it silently.
    doc = _rss(_item("https://lab.example/10", "Ten", None, "https://lab.example/10"))
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    src = FeedSource(httpx.Client(), [FEED])
    items, failures = src.fetch(SINCE, UNTIL)
    assert items == []
    assert failures == []
    assert len(src.rejected) == 1


@respx.mock
def test_entry_without_link_uses_absolute_id_as_url() -> None:
    # A link-less entry with an absolute-URL id can still be stored.
    doc = _rss(_item("https://lab.example/11", "Eleven", "Fri, 09 Oct 2026 10:00:00 GMT"))
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(SINCE, UNTIL)
    assert failures == []
    assert len(items) == 1
    assert items[0].url == "https://lab.example/11"
    assert items[0].external_id == f"{FEED.url} https://lab.example/11"


@respx.mock
def test_entry_without_link_and_non_url_id_is_rejected() -> None:
    # A link-less entry whose id is not a URL has nothing usable for `url`.
    doc = _rss(_item("42", "Forty two", "Fri, 09 Oct 2026 10:00:00 GMT"))
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    src = FeedSource(httpx.Client(), [FEED])
    items, failures = src.fetch(SINCE, UNTIL)
    assert items == []
    assert failures == []
    assert len(src.rejected) == 1


@respx.mock
def test_two_feeds_with_same_guid_do_not_collide() -> None:
    # Two labs reusing guid "42" must still produce two distinct store keys.
    doc = _rss(_item("42", "Same Guid", "Fri, 09 Oct 2026 10:00:00 GMT", "https://lab.example/a"))
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    respx.get(OTHER.url).mock(return_value=httpx.Response(200, text=doc))
    items, failures = FeedSource(httpx.Client(), [FEED, OTHER]).fetch(SINCE, UNTIL)
    assert failures == []
    assert len(items) == 2
    assert {i.external_id for i in items} == {f"{FEED.url} 42", f"{OTHER.url} 42"}


@respx.mock
def test_overlong_scoped_id_is_rejected() -> None:
    # The store column is 512 chars; an overlong scoped id fails rather than truncates.
    long_id = "https://lab.example/" + "x" * 600
    doc = _rss(_item(long_id, "Long", "Fri, 09 Oct 2026 10:00:00 GMT", long_id))
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=doc))
    src = FeedSource(httpx.Client(), [FEED])
    items, failures = src.fetch(SINCE, UNTIL)
    assert items == []
    assert failures == []
    assert len(src.rejected) == 1


@respx.mock
def test_rejected_is_reset_each_fetch() -> None:
    # rejected belongs to one fetch; a later clean fetch must not inherit it.
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=FIX))
    src = FeedSource(httpx.Client(), [FEED])
    src.rejected.append(RejectedEntry(source="feed", ref="old", error="stale"))
    src.fetch(SINCE, UNTIL)
    assert src.rejected == []


@respx.mock
def test_feed_fetch_retries_503(monkeypatch) -> None:
    # Two 503s then the fixture: the shared retry helper absorbs the blip.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.get(FEED.url).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, text=FIX),
        ]
    )
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(SINCE, UNTIL)
    assert route.call_count == 3
    assert failures == []
    assert len(items) == 2


def test_naive_window_raises() -> None:
    # A naive cursor would compare offset-naive with offset-aware; fail loudly.
    with pytest.raises(ValueError, match="timezone-aware"):
        FeedSource(httpx.Client(), [FEED]).fetch(datetime(2026, 1, 1), datetime(2030, 1, 1))


def _write_feeds(tmp_path: Path, body: str) -> Path:
    # Write a small feed list so validation failures can be exercised offline.
    path = tmp_path / "feeds.yaml"
    path.write_text(body)
    return path


def test_load_feeds_rejects_a_bad_region(tmp_path: Path) -> None:
    path = _write_feeds(
        tmp_path, "feeds:\n  - lab: X\n    region: mars\n    url: https://example.com/feed\n"
    )
    with pytest.raises(ValueError, match="region"):
        load_feeds(path)


def test_load_feeds_rejects_a_non_url(tmp_path: Path) -> None:
    path = _write_feeds(tmp_path, "feeds:\n  - lab: X\n    region: us-eu\n    url: not-a-url\n")
    with pytest.raises(ValueError, match="url"):
        load_feeds(path)


def test_load_feeds_rejects_a_non_https_url(tmp_path: Path) -> None:
    # Official feeds are verified over TLS; http must fail the same rule as the test below.
    path = _write_feeds(
        tmp_path, "feeds:\n  - lab: X\n    region: us-eu\n    url: http://example.com/feed\n"
    )
    with pytest.raises(ValueError, match="https"):
        load_feeds(path)


def test_load_feeds_rejects_an_extra_key(tmp_path: Path) -> None:
    path = _write_feeds(
        tmp_path,
        "feeds:\n  - lab: X\n    region: us-eu\n    url: https://example.com/feed\n    extra: 1\n",
    )
    with pytest.raises(ValueError, match="extra"):
        load_feeds(path)


def test_load_feeds_rejects_a_duplicate_url(tmp_path: Path) -> None:
    path = _write_feeds(
        tmp_path,
        "feeds:\n"
        "  - lab: X\n    region: us-eu\n    url: https://example.com/feed\n"
        "  - lab: Y\n    region: cn\n    url: https://example.com/feed\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_feeds(path)


def test_load_feeds_reads_the_configured_list() -> None:
    # Properties of the real file, not its size: enabling a wave-2 feed must not break this.
    feeds = load_feeds(Path("config/feeds.yaml"))
    assert feeds
    assert all(f.url.startswith("https://") for f in feeds)
    assert len({f.url for f in feeds}) == len(feeds)
    assert all(f.region in {"us-eu", "cn"} for f in feeds)


@respx.mock
def test_a_body_over_the_feed_cap_fails_that_feed(monkeypatch) -> None:
    # A feed body is held in memory whole, so its cap is the tightest of the sources: a feed that
    # streams past it is a failed feed (dead-lettered, cursor pinned), never a run out of memory.
    monkeypatch.setattr("paperwatch.sources.feeds.MAX_FEED_BODY_BYTES", 64 * 1024)
    huge = FIX + "<!--" + "x" * (64 * 1024) + "-->"
    respx.get(FEED.url).mock(return_value=httpx.Response(200, text=huge))
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(SINCE, UNTIL)
    assert items == []
    assert [url for url, _ in failures] == [FEED.url]
    assert "more than" in failures[0][1]


@respx.mock
def test_a_feed_may_redirect_between_its_own_hosts() -> None:
    # blog.example.co.uk and www.example.co.uk are the same registrable domain (co.uk is a public
    # suffix), so a feed that moves between them still fetches.
    respx.get(FEED.url).mock(
        return_value=httpx.Response(301, headers={"location": "https://www.lab.example/feed.xml"})
    )
    respx.get("https://www.lab.example/feed.xml").mock(return_value=httpx.Response(200, text=FIX))
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(SINCE, UNTIL)
    assert failures == []
    assert items and all(item.source == "feed" for item in items)


@respx.mock
def test_a_feed_redirect_off_its_site_fails_that_feed() -> None:
    # The configured feed is an official lab site; a redirect anywhere else is someone else's page,
    # so the feed fails (letter, pinned cursor) instead of importing a stranger's posts.
    respx.get(FEED.url).mock(
        return_value=httpx.Response(302, headers={"location": "https://evil.example/feed.xml"})
    )
    evil = respx.get("https://evil.example/feed.xml")
    items, failures = FeedSource(httpx.Client(), [FEED]).fetch(SINCE, UNTIL)
    assert items == []
    assert [url for url, _ in failures] == [FEED.url]
    assert "redirected off its site" in failures[0][1]
    assert not evil.called
