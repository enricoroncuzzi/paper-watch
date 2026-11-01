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
from datetime import UTC, date, datetime, timedelta
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


# Failed (set, day) requests in a row that stop the harvest for the run: one blip is tolerated,
# a second in a row looks like an outage, and asking for every later day would hammer it.
MAX_FAILED_IN_A_ROW = 2
# The server itself saying "not now": still refused after its Retry-After, it is never asked again
# in this run.
_REFUSALS = frozenset({429, 503})
# Connect and read timeout of one OAI request. Uncached ListRecords calls measured on 2026-11-01
# answered their first byte in at most 4.4 s (cs and stat:stat:ML for 2026-10-30), yet the 10-31
# and 11-01 runs lost the uncached stat:stat:ML request to two 30 s read timeouts in a row. 60 s is
# over ten times the measured worst and still half of REQUEST_DEADLINE_S.
OAI_TIMEOUT_S = 60.0
# How much older than its request a page's responseDate may be. A live answer is stamped while it
# is being served, seconds after the request; arXiv's CDN served a GET for 2026-10-29 on 11-01 with
# responseDate 2026-10-30T07:00:06Z, so a page this much older is a cached copy, never an answer.
MAX_RESPONSE_AGE = timedelta(minutes=15)


class HarvestStopped(RuntimeError):
    """arXiv refused or kept failing, so the rest of this run's harvest is not asked for."""


class UnreadablePage(ValueError):
    """A response that is not a usable OAI-PMH page: an HTML error page, a stale token or a
    cached copy of an earlier answer."""


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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _refuse_stale_answer(root: ET.Element, sent_at: datetime) -> None:
    """Raise `UnreadablePage` for a page answered more than `MAX_RESPONSE_AGE` before `sent_at`.

    A cached page of a day, the overlap day included, would mark it complete without the records
    that reached it after the copy was made. POST already keeps arXiv's CDN out of the way; this
    makes sure a copy that still arrives can never complete a day.
    """
    stamp = (root.findtext(f"{_OAI}responseDate") or "").strip()
    try:
        answered = datetime.fromisoformat(stamp)
    except ValueError:
        raise UnreadablePage(f"OAI-PMH response has no usable responseDate: {stamp!r}") from None
    if answered.tzinfo is None:
        answered = answered.replace(tzinfo=UTC)
    if answered < sent_at - MAX_RESPONSE_AGE:
        raise UnreadablePage(
            f"OAI-PMH page answered at {stamp} for a request sent at {sent_at.isoformat()}: "
            "a cached copy"
        )


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
        # Wall-clock start of the last request attempt, which its page's responseDate must match.
        self._sent_at = _utc_now()
        # (categories, term patterns) per profile, so a page filters without recompiling.
        self._match_terms = [
            (set(p.categories), [_term_pattern(t) for t in p.terms]) for p in profiles
        ]
        # Records skipped during the most recent harvest, for the run to dead-letter.
        self.rejected: list[RejectedEntry] = []
        # (set, day) pages that could not be read during the most recent harvest(day). The
        # caller keeps that day incomplete and the cursor before it, then retries next run.
        self.failures: list[str] = []
        # Spans harvest(day) calls, so an outage that starts on one day and continues into the
        # next still stops the run.
        self._failed_in_a_row = 0

    def _pace(self) -> None:
        # Space request starts by pause_s on a monotonic clock, retries included; the
        # clock is read before the request, so the request's own duration counts too.
        now = time.monotonic()
        if self._last_request is not None:
            wait = self._pause_s - (now - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()
        self._sent_at = _utc_now()

    def _post(self, params: dict[str, str]) -> str:
        # One retry after Retry-After or retry_wait_s, then the exception reaches the harvest,
        # which keeps this (set, day) incomplete and either moves on or stops for the run.
        # POST, which OAI-PMH allows for every verb: arXiv's CDN answered a GET for 2026-10-29 cs
        # on 2026-11-01 from a two-day-old cache entry (x-cache HIT, age 210460), while three POSTs
        # of the same query were x-cache MISS with fresh responseDates. A redirect would turn the
        # POST into another request, so none is followed.
        resp = send_with_retries(
            self._client,
            "POST",
            ARXIV_OAI_URL,
            data=params,
            follow_redirects=False,
            attempts=2,
            min_wait_s=self._retry_wait_s,
            max_wait_s=self._retry_wait_s,
            before_request=self._pace,
            max_bytes=MAX_ARXIV_BODY_BYTES,
            timeout=OAI_TIMEOUT_S,
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
            raise UnreadablePage("OAI-PMH response carries a DOCTYPE, which arXiv never sends")
        # ElementTree is safe here: expat >= 2.4.1 bounds entity amplification, external
        # entities are not resolved, and a DOCTYPE is refused above.
        root = ET.fromstring(text)  # noqa: S314 - see the comment above
        _refuse_stale_answer(root, self._sent_at)
        error = root.find(f"{_OAI}error")
        if error is not None:
            code = error.get("code") or "unknown"
            if code == "noRecordsMatch":
                return [], ""  # a quiet datestamp day, not a failure
            detail = f"OAI-PMH error {code}: {(error.text or '').strip()}"
            if code == "badResumptionToken":
                raise UnreadablePage(detail)
            raise ValueError(detail)
        listed = root.find(f"{_OAI}ListRecords")
        if listed is None:
            raise UnreadablePage("OAI-PMH response has no ListRecords element")
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
        self.failures = []
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
                try:
                    items, token = self._parse_page(self._post(params))
                except (httpx.HTTPError, OSError, ET.ParseError, UnreadablePage) as exc:
                    # A transport failure, a corrupt or HTML body (a CDN error page), a stale
                    # resumption token or a cached copy of an earlier answer must not abandon the
                    # other sets or the later days: the day stays incomplete and the cursor stays
                    # before it for the next run. A refusal or a second failure in a row stops the
                    # harvest instead. Any other OAI error still raises, because a request this
                    # server rejects is a bug, not a blip.
                    self._failed_in_a_row += 1
                    refused = (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code in _REFUSALS
                    )
                    if refused or self._failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                        raise HarvestStopped(
                            f"arXiv harvest stopped at {set_spec} on {day.isoformat()}: {exc!r}"
                        ) from exc
                    self._mark_failed(day, set_spec, exc)
                    break
                self._failed_in_a_row = 0
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

    def _mark_failed(self, day: date, set_spec: str, exc: BaseException) -> None:
        """Record one unreadable (set, day) page without making the whole harvest fatal."""
        message = f"{set_spec} on {day.isoformat()}: {exc!r}"
        self.failures.append(message)
        log.warning("arxiv_page_failed", set=set_spec, day=day.isoformat(), error=repr(exc))
