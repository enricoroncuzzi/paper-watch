"""arXiv OAI-PMH bulk-harvest client storing arXivRaw metadata.

Terms: https://info.arxiv.org/help/api/tou.html and https://info.arxiv.org/help/oa/index.html.
OAI-PMH is arXiv's bulk-harvest interface and its preferred way to keep an up-to-date
copy: at most one request every three seconds on one connection, metadata/abstracts only
(CC0), never the PDF. A datestamp day is harvested at a time so a paper is seen the day it
is announced and seen again when it is revised, which the Atom query interface never did.
"""

import re
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx
import structlog

from paperwatch.config import ARXIV_OAI_PREFIX, ARXIV_OAI_URL
from paperwatch.domain import Profile, RejectedEntry, SourceItem
from paperwatch.sources.base import MAX_ARXIV_BODY_BYTES, send_with_retries

# The OAI-PMH envelope and the arXivRaw metadata body; element names carry both.
_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_RAW = "{http://arxiv.org/OAI/arXivRaw/}"
# arXiv appends a version to every id (v1, v2, ...); identity is the base id.
_VERSION = re.compile(r"v(\d+)$")
# Bound one (set, day) like hf_daily's MAX_PAGES_PER_DAY: a server that keeps handing out a
# resumption token, or repeats one, must not make a day's harvest run all night. A real day of one
# set is a couple of pages at the server's page size, so 200 is far above any honest answer.
MAX_PAGES_PER_DAY = 200
log = structlog.get_logger()


def arxiv_base_id(raw: str) -> str:
    """Return the version-free arXiv id, accepting a URL or a bare id."""
    return _VERSION.sub("", raw.rstrip("/").split("/abs/")[-1])


# arXiv's physics umbrella: these archives have no umbrella OAI set of their own, and their
# real sets live under `physics:` (for example `physics:astro-ph:CO`). Until that mapping is
# added, a profile using one must fail loudly rather than silently harvest nothing or fail each
# day on a set arXiv does not know.
_PHYSICS_UMBRELLA = frozenset(
    {
        "astro-ph",
        "cond-mat",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "quant-ph",
    }
)


def oai_sets(profiles: list[Profile]) -> list[str]:
    """The OAI sets to harvest for these profiles, sorted and deduplicated.

    The cs archive has one umbrella set; any other archive is queried at its sub-archive
    set, for example stat.ML -> stat:stat:ML. A category the mapping cannot name raises, so
    a profile edit cannot silently harvest nothing or break every day.
    """
    sets: set[str] = set()
    for profile in profiles:
        for category in profile.categories:
            archive, _, sub = category.partition(".")
            if not sub:
                raise ValueError(
                    f"arXiv category {category!r} has no sub-archive, so no OAI set is known"
                )
            if archive in _PHYSICS_UMBRELLA:
                raise ValueError(
                    f"arXiv category {category!r} is under the physics umbrella, whose OAI set "
                    "mapping is not implemented"
                )
            if archive == "cs":
                sets.add("cs")
            else:
                sets.add(f"{archive}:{archive}:{sub}")
    return sorted(sets)


def _term_pattern(term: str) -> re.Pattern[str]:
    # A phrase's internal spaces match any whitespace, and \b keeps "bot" out of "robot". The
    # last word also accepts a plural, because the arXiv all: query it replaces stemmed terms;
    # "tool use" matching "tool uses" (the verb) is accepted for the same reason.
    parts = [re.escape(part) for part in term.split()]
    last = parts[-1] + r"(?:s|es)?"
    return re.compile(r"\b" + r"\s+".join([*parts[:-1], last]) + r"\b", re.I)


def _version_number(version: ET.Element) -> int:
    # "v2" ranks above "v1"; an absent version ranks lowest.
    match = _VERSION.search(version.get("version") or "")
    return int(match.group(1)) if match else 0


def _split_authors(value: str) -> tuple[str, ...]:
    # arXivRaw separates authors with commas and a final " and "; an affiliation follows a name
    # in parentheses. Drop the groups first (repeatedly, for nesting), so a comma or "and"
    # inside an affiliation cannot split a name. Stop when a pass removes nothing: a stray ")"
    # before a stray "(" never pairs, so testing for the characters alone would loop forever.
    while (stripped := re.sub(r"\([^()]*\)", "", value)) != value:
        value = stripped
    authors: list[str] = []
    for part in re.split(r",|\band\b", value):
        name = part.strip()
        if name:
            authors.append(name)
    return tuple(authors)


def _published_at(version: ET.Element) -> datetime:
    # OAI dates are RFC 2822 with a "GMT" zone; store the v1 date in UTC.
    parsed = parsedate_to_datetime((version.findtext(f"{_RAW}date") or "").strip())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ArxivOaiSource:
    # Matches the source field written onto every SourceItem.
    name = "arxiv"

    def __init__(
        self,
        client: httpx.Client,
        profiles: list[Profile],
        pause_s: float = 3.0,
        retry_wait_s: float = 60.0,
    ) -> None:
        self._client, self._profiles = client, profiles
        self._pause_s, self._retry_wait_s = pause_s, retry_wait_s
        # Resolve the sets now, so a profile the mapping cannot name fails at startup rather
        # than silently harvesting nothing or failing every day on an unknown set.
        self._sets = oai_sets(profiles)
        # Start of the last request attempt, for the pacing rule; None before the first.
        self._last_request: float | None = None
        # (categories, term patterns) per profile, so a page filters without recompiling.
        self._match_terms = [
            (set(p.categories), [_term_pattern(t) for t in p.terms]) for p in profiles
        ]
        # Records skipped during the most recent harvest, for the run to dead-letter.
        self.rejected: list[RejectedEntry] = []

    def _pace(self) -> None:
        # Space request starts by pause_s on a monotonic clock, retries included; the
        # clock is read before the request, so the request's own duration counts too.
        now = time.monotonic()
        if self._last_request is not None:
            wait = self._pause_s - (now - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, params: dict[str, str]) -> str:
        # One retry after Retry-After or retry_wait_s, then the exception propagates so the
        # run degrades and the source stops for the day.
        resp = send_with_retries(
            self._client,
            "GET",
            ARXIV_OAI_URL,
            params=params,
            attempts=2,
            min_wait_s=self._retry_wait_s,
            max_wait_s=self._retry_wait_s,
            before_request=self._pace,
            max_bytes=MAX_ARXIV_BODY_BYTES,
        )
        return resp.text

    def _matches(self, categories: tuple[str, ...], text: str) -> bool:
        # A record is in scope when a profile's category and one of its terms both hit.
        cats = set(categories)
        return any(
            cats & profile_cats and any(pattern.search(text) for pattern in patterns)
            for profile_cats, patterns in self._match_terms
        )

    def _item(self, record: ET.Element) -> SourceItem | None:
        # None means the record is out of scope; a structural problem raises and is rejected.
        raw = record.find(f".//{_RAW}arXivRaw")
        if raw is None:
            raise ValueError("record has no arXivRaw metadata")
        raw_id = (raw.findtext(f"{_RAW}id") or "").strip()
        if not raw_id:
            raise ValueError("arXivRaw has no id")
        versions = raw.findall(f"{_RAW}version")
        if not versions:
            raise ValueError("arXivRaw has no version")
        ordered = sorted(versions, key=_version_number)
        if not (ordered[0].findtext(f"{_RAW}date") or "").strip():
            raise ValueError("arXivRaw v1 has no date")
        # Normalise whitespace in the title and abstract: OAI carries newlines and runs.
        title = " ".join((raw.findtext(f"{_RAW}title") or "").split())
        abstract = " ".join((raw.findtext(f"{_RAW}abstract") or "").split())
        categories = tuple((raw.findtext(f"{_RAW}categories") or "").split())
        if not self._matches(categories, f"{title} {abstract}"):
            return None
        base = arxiv_base_id(raw_id)
        return SourceItem(
            source=self.name,
            external_id=base,
            title=title,
            authors=_split_authors(raw.findtext(f"{_RAW}authors") or ""),
            text=abstract,
            url=f"https://arxiv.org/abs/{base}",
            # The v1 date is the first submission date; arXiv announces it one to three days later.
            published_at=_published_at(ordered[0]),
            categories=categories,
            licence="CC0-1.0 (metadata)",
            version=_version_number(ordered[-1]),
        )

    def _parse_page(self, text: str) -> tuple[list[SourceItem], str]:
        if "<!DOCTYPE" in text:
            # OAI-PMH never carries a DTD; refusing one blocks entity-amplification bodies
            # before expat sees them.
            raise ValueError("OAI-PMH response carries a DOCTYPE, which arXiv never sends")
        # ElementTree is safe here: expat >= 2.4.1 bounds entity amplification, external
        # entities are not resolved, and a DOCTYPE is refused above.
        root = ET.fromstring(text)  # noqa: S314 - see the comment above
        error = root.find(f"{_OAI}error")
        if error is not None:
            code = error.get("code") or "unknown"
            if code == "noRecordsMatch":
                return [], ""  # a quiet datestamp day, not a failure
            raise ValueError(f"OAI-PMH error {code}: {(error.text or '').strip()}")
        listed = root.find(f"{_OAI}ListRecords")
        if listed is None:
            raise ValueError("OAI-PMH response has no ListRecords element")
        items: list[SourceItem] = []
        for record in listed.findall(f"{_OAI}record"):
            header = record.find(f"{_OAI}header")
            if header is not None and header.get("status") == "deleted":
                continue  # a withdrawn paper is not a news item
            try:
                item = self._item(record)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                # A deterministic bad record must not block the page: skip it and record
                # the OAI identifier so Task 9 dead-letters it and still advances the day.
                ref = header.findtext(f"{_OAI}identifier") if header is not None else None
                ref = ref or "<missing id>"
                self.rejected.append(RejectedEntry(source=self.name, ref=ref, error=str(exc)))
                log.warning("arxiv_record_rejected", ref=ref, error=str(exc))
                continue
            if item is not None:
                items.append(item)
        token_el = listed.find(f"{_OAI}resumptionToken")
        token = (token_el.text or "").strip() if token_el is not None else ""
        return items, token

    def harvest(self, day: date) -> Iterator[list[SourceItem]]:
        """Yield one filtered page per OAI response for `day`, sets in sorted order.

        The caller receives each page before the next request goes out, so it can park
        the page durably and only then ask for the resumption token. A (set, day) is bounded by
        `MAX_PAGES_PER_DAY`, and a token the server already handed out for it ends the harvest
        with an error naming both, so a broken pager degrades arXiv instead of running forever.
        """
        # rejected describes one harvest; clear it so callers never read stale rows.
        self.rejected = []
        # Base ids already yielded in this call: a paper cross-listed into two sets is
        # yielded once, at the version its single arXivRaw record lists (its highest).
        seen: set[str] = set()
        for set_spec in self._sets:
            token = ""
            first = True
            pages = 0
            # Tokens this (set, day) already consumed: a repeat means the pager is looping.
            used: set[str] = set()
            while True:
                if pages >= MAX_PAGES_PER_DAY:
                    raise ValueError(
                        f"OAI-PMH page cap ({MAX_PAGES_PER_DAY}) reached for set {set_spec} "
                        f"on {day.isoformat()}"
                    )
                params = (
                    {
                        "verb": "ListRecords",
                        "metadataPrefix": ARXIV_OAI_PREFIX,
                        "set": set_spec,
                        "from": day.isoformat(),
                        "until": day.isoformat(),
                    }
                    if first
                    else {"verb": "ListRecords", "resumptionToken": token}
                )
                items, token = self._parse_page(self._get(params))
                pages += 1
                page: list[SourceItem] = []
                for item in items:
                    if item.external_id in seen:
                        continue
                    seen.add(item.external_id)
                    page.append(item)
                if page:
                    yield page
                first = False
                if not token:
                    break
                if token in used:
                    raise ValueError(
                        f"OAI-PMH resumption token {token!r} repeated for set {set_spec} "
                        f"on {day.isoformat()} at page {pages}"
                    )
                used.add(token)
