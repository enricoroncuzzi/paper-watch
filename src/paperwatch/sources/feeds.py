"""Tier-3 lab blog feed client: official RSS/Atom only, never scraped HTML."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

import feedparser
import httpx
import yaml

from paperwatch.domain import SourceItem

# Feed bodies arrive as HTML fragments; strip tags so only readable text is stored.
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Feed:
    lab: str
    url: str
    region: str


def load_feeds(path: Path) -> list[Feed]:
    # Feeds are configuration, not code, so the lab list is data in YAML.
    return [Feed(**f) for f in yaml.safe_load(path.read_text())["feeds"]]


class FeedSource:
    name = "feed"

    def __init__(self, client: httpx.Client, feeds: list[Feed]) -> None:
        self._client, self._feeds = client, feeds

    def fetch(
        self, since: datetime, until: datetime
    ) -> tuple[list[SourceItem], list[tuple[str, str]]]:
        items: list[SourceItem] = []
        failures: list[tuple[str, str]] = []
        for feed in self._feeds:
            try:
                resp = self._client.get(feed.url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # A down feed is recorded and skipped; the other feeds still run.
                failures.append((feed.url, str(exc)))
                continue
            for e in feedparser.parse(resp.text).entries:
                # Some feeds carry only updated; both are parsed UTC stamps.
                stamp = e.get("published_parsed") or e.get("updated_parsed")
                if not stamp:
                    continue
                year, month, day, hour, minute, second = stamp[:6]
                when = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
                if not since <= when < until:
                    continue
                body = e.get("content", [{}])[0].get("value") or e.get("summary", "")
                items.append(
                    SourceItem(
                        source=self.name,
                        external_id=e.get("id") or e.link,
                        title=" ".join(e.title.split()),
                        # Tier 3 has no per-author data; the lab is the author.
                        authors=(feed.lab,),
                        text=" ".join(unescape(_TAGS.sub(" ", body)).split()),
                        url=e.link,
                        published_at=when,
                        licence="copyright (private search only)",
                    )
                )
        return items, failures
