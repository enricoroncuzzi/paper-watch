"""arXiv Atom API client.

Terms: https://info.arxiv.org/help/api/tou.html - at most one request every
three seconds on one connection, and only metadata/abstracts are stored (CC0),
never the PDF.
"""

import re
import time
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from paperwatch.domain import Profile, RejectedEntry, SourceItem

ARXIV_URL = "https://export.arxiv.org/api/query"
# arXiv appends a version to every id (v1, v2, ...); identity is the base id.
_VERSION = re.compile(r"v\d+$")
# The API refuses start values past this ceiling, so paging must stop before it.
_MAX_START = 30_000
log = structlog.get_logger()


def arxiv_base_id(raw: str) -> str:
    """Return the version-free arXiv id, accepting a URL or a bare id."""
    return _VERSION.sub("", raw.rstrip("/").split("/abs/")[-1])


def _version_of(raw: str) -> int:
    # Trailing "vN" ranks versions so the newest body wins; a bare id ranks 0.
    match = _VERSION.search(raw.rstrip("/"))
    return int(match.group()[1:]) if match else 0


def _utc(parsed: tuple[int, ...]) -> datetime:
    # feedparser hands back a naive struct_time; arXiv timestamps are UTC.
    year, month, day, hour, minute, second = parsed[:6]
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _stamp(value: datetime) -> str:
    # submittedDate is GMT: an aware value converts, a naive one is already UTC.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return f"{value.astimezone(UTC):%Y%m%d%H%M}"


def _profile_query(p: Profile, since: datetime, until: datetime) -> str:
    # Build one query per profile: category filter AND term filter AND date window.
    cats = " OR ".join(f"cat:{c}" for c in p.categories)
    terms = " OR ".join(f'all:"{t}"' for t in p.terms)
    window = f"submittedDate:[{_stamp(since)} TO {_stamp(until)}]"
    return f"({cats}) AND ({terms}) AND {window}"


def _total_results(feed: Any) -> int | None:
    # opensearch:totalResults is how many entries the whole window holds.
    raw = feed.get("opensearch_totalresults")
    return int(raw) if raw else None


class ArxivSource:
    # Matches the source field written onto every SourceItem.
    name = "arxiv"

    def __init__(
        self,
        client: httpx.Client,
        profiles: list[Profile],
        page_size: int = 100,
        pause_s: float = 3.0,
    ) -> None:
        self._client, self._profiles = client, profiles
        self._page_size, self._pause_s = page_size, pause_s
        # Entries skipped during the most recent fetch, for the run to dead-letter.
        self.rejected: list[RejectedEntry] = []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=30), reraise=True)
    def _get(self, params: dict[str, str | int]) -> str:
        # Retries keep at least 3 s between attempts, which doubles as the rate limit.
        resp = self._client.get(ARXIV_URL, params=params)
        resp.raise_for_status()
        return resp.text

    def _item(self, e: Any) -> SourceItem:
        # Feedparser entries are dict-like; a missing field raises and is re-labelled.
        base = arxiv_base_id(e.id)
        return SourceItem(
            source=self.name,
            external_id=base,
            # Normalise whitespace: Atom carries newlines in titles/summaries.
            title=" ".join(e.title.split()),
            authors=tuple(a.name for a in e.get("authors", [])),
            text=" ".join(e.summary.split()),
            url=f"https://arxiv.org/abs/{base}",
            published_at=_utc(e.published_parsed),
            categories=tuple(t.term for t in e.get("tags", [])),
            licence="CC0-1.0 (metadata)",
            version=_version_of(e.id),
        )

    def fetch(self, since: datetime, until: datetime) -> list[SourceItem]:
        # rejected describes one fetch; clear it so callers never read stale rows.
        self.rejected = []
        seen: dict[str, SourceItem] = {}
        # Highest version number stored per base id, to pick the newest body.
        versions: dict[str, int] = {}
        for profile in self._profiles:
            start = 0
            while True:
                text = self._get(
                    {
                        "search_query": _profile_query(profile, since, until),
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "start": start,
                        "max_results": self._page_size,
                    }
                )
                feed = feedparser.parse(text)
                entries = feed.entries
                total = _total_results(feed.feed)
                for e in entries:
                    try:
                        item = self._item(e)
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        # A deterministic bad entry must not block the whole source, so
                        # skip it and record the id: Task 9 writes a dead letter, marks
                        # the run partial, and still advances the cursor.
                        ref = e.get("id") or "<missing id>"
                        self.rejected.append(
                            RejectedEntry(source=self.name, ref=ref, error=str(exc))
                        )
                        log.warning("arxiv_entry_rejected", ref=ref, error=str(exc))
                        continue
                    # Replace the stored body when a higher version of the same base id
                    # arrives, independent of the order pages happen to return.
                    version = _version_of(e.id)
                    if item.external_id not in seen or version > versions[item.external_id]:
                        seen[item.external_id] = item
                        versions[item.external_id] = version
                # Pacing: one request every pause_s seconds, retries included.
                time.sleep(self._pause_s)
                if not entries:
                    # An empty page ends paging only when the total is unknown or
                    # reached; otherwise it is a transient gap and returning early
                    # would silently lose the rest of the window.
                    if total is not None and start < total:
                        raise ValueError(
                            f"arXiv returned an empty page at start={start} "
                            f"but totalResults={total}"
                        )
                    break
                if total is not None and start + len(entries) >= total:
                    # totalResults says we have the whole window.
                    break
                if len(entries) < self._page_size:
                    # A short page is an ending only when no total is known; with a
                    # total that says more exist it is a truncated response, so fail
                    # loudly and let the run dead-letter and replay the window.
                    if total is not None:
                        raise ValueError(
                            f"arXiv returned a short page ({len(entries)}) at "
                            f"start={start} but totalResults={total}"
                        )
                    break
                start += self._page_size
                if start >= _MAX_START:
                    # Past the API ceiling we cannot page further; stop rather than loop.
                    break
        return list(seen.values())
