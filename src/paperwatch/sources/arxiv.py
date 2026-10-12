"""arXiv Atom API client.

Terms: https://info.arxiv.org/help/api/tou.html - at most one request every
three seconds on one connection, and only metadata/abstracts are stored (CC0),
never the PDF.
"""

import re
import time
from datetime import UTC, datetime

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from paperwatch.domain import Profile, SourceItem

ARXIV_URL = "https://export.arxiv.org/api/query"
# arXiv appends a version to every id (v1, v2, ...); identity is the base id.
_VERSION = re.compile(r"v\d+$")


def arxiv_base_id(raw: str) -> str:
    """Return the version-free arXiv id, accepting a URL or a bare id."""
    return _VERSION.sub("", raw.rstrip("/").split("/abs/")[-1])


def _utc(parsed: tuple[int, ...]) -> datetime:
    # feedparser hands back a naive struct_time; arXiv timestamps are UTC.
    year, month, day, hour, minute, second = parsed[:6]
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _profile_query(p: Profile, since: datetime, until: datetime) -> str:
    # Build one query per profile: category filter AND term filter AND date window.
    cats = " OR ".join(f"cat:{c}" for c in p.categories)
    terms = " OR ".join(f'all:"{t}"' for t in p.terms)
    window = f"submittedDate:[{since:%Y%m%d%H%M} TO {until:%Y%m%d%H%M}]"
    return f"({cats}) AND ({terms}) AND {window}"


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=30), reraise=True)
    def _get(self, params: dict[str, str | int]) -> str:
        # Retries keep at least 3 s between attempts, which doubles as the rate limit.
        resp = self._client.get(ARXIV_URL, params=params)
        resp.raise_for_status()
        return resp.text

    def fetch(self, since: datetime, until: datetime) -> list[SourceItem]:
        # setdefault keeps the first (usually newest) version seen for a base id.
        seen: dict[str, SourceItem] = {}
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
                entries = feedparser.parse(text).entries
                for e in entries:
                    base = arxiv_base_id(e.id)
                    seen.setdefault(
                        base,
                        SourceItem(
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
                        ),
                    )
                # Pacing: one request every pause_s seconds, retries included.
                time.sleep(self._pause_s)
                if len(entries) < self._page_size:
                    break
                start += self._page_size
        return list(seen.values())
