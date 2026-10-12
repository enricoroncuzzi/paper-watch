"""Tier-3 lab blog feed client: official RSS/Atom only, never scraped HTML."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Annotated, Any, Literal

import feedparser
import httpx
import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, StringConstraints, ValidationError

from paperwatch.domain import RejectedEntry, SourceItem
from paperwatch.sources.base import send_with_retries

# Feed bodies arrive as HTML fragments; strip tags so only readable text is stored.
_TAGS = re.compile(r"<[^>]+>")
# Keep an image's alt text before the generic strip deletes the whole tag.
_IMG = re.compile(r"""<img[^>]*\balt=["']([^"']*)["'][^>]*>""", re.IGNORECASE)
# A real feed must contain one of these roots; anything else is not a feed.
_FEED_ROOTS = (b"<rss", b"<feed", b"<rdf")
_FEED_PROBE_BYTES = 8192
# The planned store column for external_id; fail rather than truncate a long id.
_MAX_ID_LEN = 512


@dataclass(frozen=True)
class Feed:
    lab: str
    url: str
    region: str


# A blank lab name would make tier-3 items anonymous, so strip and require one char.
_LabName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _FeedConfig(BaseModel):
    # extra="forbid" turns a typo'd or stray key into a clear error, not a silent ignore.
    model_config = ConfigDict(extra="forbid")

    lab: _LabName
    # Only the two regions the corpus uses; a new region is a deliberate code change.
    region: Literal["us-eu", "cn"]
    url: HttpUrl


class _FeedsConfig(BaseModel):
    # The file has one top-level key, so a stray key there is a mistake too.
    model_config = ConfigDict(extra="forbid")

    feeds: list[_FeedConfig]


def load_feeds(path: Path) -> list[Feed]:
    # Feeds are configuration, not code, so the lab list is validated YAML data.
    try:
        config = _FeedsConfig.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as exc:
        raise ValueError(f"invalid feed list in {path}: {exc}") from exc
    urls = [str(feed.url) for feed in config.feeds]
    if len(urls) != len(set(urls)):
        # A duplicate URL would fetch and store the same posts twice.
        duplicates = sorted({url for url in urls if urls.count(url) > 1})
        raise ValueError(f"duplicate feed URLs in {path}: {duplicates}")
    return [Feed(lab=feed.lab, url=str(feed.url), region=feed.region) for feed in config.feeds]


def _looks_like_feed(content: bytes) -> bool:
    # A 200 that is not a feed is a failure, not a quiet day: check the root tag.
    head = content[:_FEED_PROBE_BYTES].lower()
    return any(root in head for root in _FEED_ROOTS)


def _when(e: Any) -> datetime:
    # feedparser pre-parses most stamps; fall back to the raw date string.
    stamp = e.get("published_parsed")
    if not stamp and "updated_parsed" in e:
        # Membership avoids feedparser's deprecated updated->published mapping.
        stamp = e.get("updated_parsed")
    if stamp:
        year, month, day, hour, minute, second = stamp[:6]
        return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    raw = e.get("published")
    if not raw and "updated" in e:
        raw = e.get("updated")
    if not raw:
        raise ValueError("entry has no date")
    when = parsedate_to_datetime(raw)
    if when.tzinfo is None:
        # A zone-less stamp is the lab's local clock; store it as UTC.
        when = when.replace(tzinfo=UTC)
    return when.astimezone(UTC)


def _text(body: str) -> str:
    # Keep image alt text, strip remaining tags, decode entities, collapse spaces.
    return " ".join(unescape(_TAGS.sub(" ", _IMG.sub(r"\1", body))).split())


class FeedSource:
    name = "feed"

    def __init__(self, client: httpx.Client, feeds: list[Feed]) -> None:
        self._client, self._feeds = client, feeds
        # Task 9 reads this to dead-letter a bad entry without pinning the cursor.
        self.rejected: list[RejectedEntry] = []

    def fetch(
        self, since: datetime, until: datetime
    ) -> tuple[list[SourceItem], list[tuple[str, str]]]:
        if since.tzinfo is None or until.tzinfo is None:
            # A naive cursor would raise later; reject it before any HTTP call.
            raise ValueError("since and until must be timezone-aware")
        # rejected describes one fetch; clear it so callers never read stale rows.
        self.rejected = []
        items: list[SourceItem] = []
        failures: list[tuple[str, str]] = []
        for feed in self._feeds:
            try:
                resp = send_with_retries(self._client, "GET", feed.url)
                if not _looks_like_feed(resp.content):
                    # A maintenance page or error document is a failure, not a quiet day.
                    content_type = resp.headers.get("content-type", "unknown")
                    raise ValueError(f"{feed.url} returned {content_type}, not a feed")
            except (httpx.HTTPError, ValueError) as exc:
                # A down feed is recorded and skipped; the other feeds still run.
                failures.append((feed.url, str(exc)))
                continue
            for e in feedparser.parse(resp.content).entries:
                ref = e.get("id") or e.get("link") or "<missing id>"
                try:
                    when = _when(e)
                    if not since <= when < until:
                        continue
                    title = e.get("title")
                    if not title:
                        raise ValueError("entry has no title")
                    link = e.get("link")
                    raw_id = e.get("id") or link
                    if not raw_id:
                        raise ValueError("entry has no id or link")
                    url = link or raw_id
                    if not str(url).startswith(("http://", "https://")):
                        raise ValueError(f"entry has no usable URL: {url!r}")
                    # Scope the id with the feed URL: two labs can reuse a guid.
                    external_id = f"{feed.url} {raw_id}"
                    if len(external_id) > _MAX_ID_LEN:
                        raise ValueError(
                            f"scoped id is {len(external_id)} chars (max {_MAX_ID_LEN})"
                        )
                    content = e.get("content")
                    body = (content[0].get("value") if content else None) or e.get("summary") or ""
                    items.append(
                        SourceItem(
                            source=self.name,
                            external_id=external_id,
                            title=" ".join(str(title).split()),
                            # Tier 3 has no per-author data; the lab is the author.
                            authors=(feed.lab,),
                            text=_text(body),
                            url=str(url),
                            published_at=when,
                            licence="copyright (private search only)",
                        )
                    )
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    # One bad entry must not fail the feed or pin its cursor.
                    self.rejected.append(
                        RejectedEntry(source=self.name, ref=str(ref), error=str(exc))
                    )
        return items, failures
