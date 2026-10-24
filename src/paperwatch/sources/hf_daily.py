"""Hugging Face daily papers: community attention and the extras the listing carries."""

import re
import unicodedata
from datetime import UTC, date, datetime, timedelta
from typing import cast
from urllib.parse import urlsplit

import httpx
import structlog

from paperwatch.config import HF_ORG_MAX_CHARS, INT4_MAX
from paperwatch.domain import HfPaper as HfPaper
from paperwatch.sources.arxiv import arxiv_base_id
from paperwatch.sources.base import MAX_HF_BODY_BYTES, is_retryable, send_with_retries

HF_URL = "https://huggingface.co/api/daily_papers"
# The endpoint caps one page at 100 rows, so a full page means there is a next one.
PAGE_SIZE = 100
# Bound one day so a server that ignores p cannot loop forever.
MAX_PAGES_PER_DAY = 20
# Two fully failed days in a row mean the listing endpoint is down, not that these days are odd:
# asking the rest of the 31-day window would spend three attempts and two waits per day on the same
# outage, which a gateway answering 504 after a minute stretches to hours. The remaining days are
# re-read by every run anyway, so stopping loses nothing.
HF_FAILED_DAYS = 2
# Only an https GitHub repository URL becomes a stored link: everything else (a javascript:
# scheme, a look-alike host) is dropped at the source, so nothing downstream can render it.
GITHUB_REPO = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?")
# Bounds a project-page link the way the daily page itself does; anything longer is a payload.
PROJECT_PAGE_MAX_CHARS = 2048
# GitHub allows a 39-character owner and a 100-character repo; the extra room covers the prefix
# and the slash, and a value past this width is not a repository link.
GITHUB_REPO_MAX_CHARS = 200
# Only a real arXiv id may key the listing. It has to match a stored external_id exactly, and a
# hostile or enormous value would cost the whole write for every paper: new style 2510.12345, old
# style hep-th/9901001 (arxiv_base_id has already stripped the version).
ARXIV_ID = re.compile(r"[0-9]{4}\.[0-9]{4,5}|[a-z][a-z-]*/[0-9]{7}")
# The daily listing began in 2023; the floor leaves a margin below that. A stamp outside this span
# is a bad extra, and one at the edge of years 1..9999 cannot be read back whenever the database
# session time zone moves it across the edge, which would fail every later run and search.
LISTED_AT_FLOOR = datetime(2020, 1, 1, tzinfo=UTC)
LISTED_AT_CEILING = datetime(2100, 1, 1, tzinfo=UTC)
log = structlog.get_logger()


def _json_int(text: str) -> int:
    # json.loads refuses an integer of more than 4,300 digits, which would fail the whole page for
    # one value; anything this long is past int4 and is dropped as a bad count.
    return int(text) if len(text) <= 64 else INT4_MAX + 1


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = int(cast("str | int | float", value))
    except (TypeError, ValueError, OverflowError):
        # Infinity (JSON's "1e999") overflows int; it is a bad extra, not a failed day.
        return None
    # The columns are int4: a number they cannot hold would fail the whole listing write, so it
    # is dropped like any other malformed extra.
    return number if 0 <= number <= INT4_MAX else None


def _github_repo(value: object) -> str | None:
    if (
        isinstance(value, str)
        and len(value) <= GITHUB_REPO_MAX_CHARS
        and GITHUB_REPO.fullmatch(value)
    ):
        return value
    return None


def _has_unstorable_characters(value: str) -> bool:
    # Postgres text cannot hold a control character and a lone surrogate cannot be encoded at
    # all, so a name or a link carrying either is a bad extra, not a value to store.
    return any(unicodedata.category(character) in ("Cc", "Cs") for character in value)


def _project_page(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > PROJECT_PAGE_MAX_CHARS:
        return None
    if not value.startswith("https://"):
        return None
    # A stored link is only ever as safe as its shape: no whitespace (the shared check catches
    # control and surrogate characters), and urlsplit must name an https host with no credentials.
    if _has_unstorable_characters(value) or any(character.isspace() for character in value):
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        # A malformed IPv6 or normalization-hostile host raises here; it is a bad page, not a
        # failed day.
        return None
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return None
    return value


def _organization(paper: dict[str, object], entry: dict[str, object]) -> str | None:
    # The paper's own organisation is the one the listing attaches; the entry's is the fallback
    # for a paper row that carries none, and `fullname` reads better than the slug.
    for holder in (paper, entry):
        org = holder.get("organization")
        if isinstance(org, dict):
            for key in ("fullname", "name"):
                value = org.get(key)
                if isinstance(value, str) and value and not _has_unstorable_characters(value):
                    # The column holds HF_ORG_MAX_CHARS; clipping here keeps a long name from
                    # failing the write, and the store clips again for a caller that skips us.
                    return value[:HF_ORG_MAX_CHARS]
    return None


def _listed_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        # The listing only ever sends UTC, so that is the honest reading of a naive stamp.
        stamp = stamp.replace(tzinfo=UTC)
    try:
        stamp = stamp.astimezone(UTC)
    except OverflowError:
        return None
    return stamp if LISTED_AT_FLOOR <= stamp <= LISTED_AT_CEILING else None


def _paper(entry: object) -> tuple[str, HfPaper] | None:
    # A row (or a paper) of the wrong shape is a bad row to skip, not a failed page. Only a real
    # arXiv id becomes a key: it is matched against a stored external_id, and a hostile or
    # enormous one would cost the whole write for every paper.
    if not isinstance(entry, dict):
        return None
    paper = entry.get("paper")
    if not isinstance(paper, dict):
        return None
    base = arxiv_base_id(str(paper.get("id") or ""))
    if not ARXIV_ID.fullmatch(base):
        return None
    upvotes = _as_int(paper.get("upvotes") or 0)
    if upvotes is None:
        # A single malformed row is skipped, not the whole day.
        return None
    return base, HfPaper(
        upvotes=upvotes,
        comments=_as_int(entry.get("numComments")),
        github_repo=_github_repo(paper.get("githubRepo")),
        github_stars=_as_int(paper.get("githubStars")),
        project_page=_project_page(paper.get("projectPage")),
        organization=_organization(paper, entry),
        listed_at=_listed_at(paper.get("submittedOnDailyAt")),
    )


def _highest(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _merge(old: HfPaper, new: HfPaper) -> HfPaper:
    # Counts only grow across a window; an extra the newest page has not filled in yet keeps the
    # value an older page carried, and the listing time is the earliest one seen.
    listed = [stamp for stamp in (old.listed_at, new.listed_at) if stamp is not None]
    return HfPaper(
        upvotes=max(old.upvotes, new.upvotes),
        comments=_highest(old.comments, new.comments),
        github_repo=old.github_repo or new.github_repo,
        github_stars=_highest(old.github_stars, new.github_stars),
        project_page=old.project_page or new.project_page,
        organization=old.organization or new.organization,
        listed_at=min(listed) if listed else None,
    )


class HfDailySource:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        # Task 9 reads these to record a dead letter and mark the run partial.
        self.degraded = False
        self.failure: str | None = None

    def papers(self, since: date, until: date) -> dict[str, HfPaper]:
        # The health flags describe this call only.
        self.degraded = False
        self.failure = None
        out: dict[str, HfPaper] = {}
        failed_in_a_row = 0
        # Newest first, so a day that fails for its own reason costs the least valuable days: the
        # listing that carries a fresh paper's upvotes is the newest one, and `out` merges with
        # max(), so the order never changes the answer.
        day = until
        while day >= since:
            try:
                self._merge_day(out, day)
                failed_in_a_row = 0
            except (
                httpx.HTTPError,
                ValueError,
                TypeError,
                AttributeError,
                RecursionError,
            ) as exc:
                # A failed day must not discard earlier days; record the outage so
                # the run can dead-letter it and stay visible without blocking.
                self.degraded = True
                self.failure = f"hf {day.isoformat()}: {exc}"
                # Only a retryable failure says the endpoint is down; a bad answer (a page cap, a
                # shape this client does not understand) is that day's problem and ends the streak.
                failed_in_a_row = failed_in_a_row + 1 if is_retryable(exc) else 0
                if failed_in_a_row >= HF_FAILED_DAYS:
                    self.failure = (
                        f"hf {day.isoformat()}: {failed_in_a_row} days in a row failed with a "
                        f"retryable error, stopped asking: {exc}"
                    )
                    log.warning(
                        "hf_stopped_asking",
                        day=day.isoformat(),
                        failed_days=failed_in_a_row,
                        error=str(exc),
                    )
                    break
            day -= timedelta(days=1)
        return out

    def _merge_day(self, out: dict[str, HfPaper], day: date) -> None:
        page = 0
        seen_today: set[str] = set()
        while True:
            if page >= MAX_PAGES_PER_DAY:
                # More full pages than any real day has: treat the day as failed.
                raise ValueError(f"hf page cap reached for {day.isoformat()} at p={page}")
            resp = send_with_retries(
                self._client,
                "GET",
                HF_URL,
                params={"date": day.isoformat(), "limit": PAGE_SIZE, "p": page},
                max_bytes=MAX_HF_BODY_BYTES,
            )
            rows = resp.json(parse_int=_json_int)
            if not isinstance(rows, list):
                # An error envelope or a wrapped list is API drift, not a day with no papers: it
                # must stay a visible failed day rather than look like a healthy empty one.
                raise TypeError(f"hf answer is not a list: {type(rows).__name__}")
            new_ids = 0
            for entry in rows:
                # A null paper is a bad row to skip, not a failed page.
                parsed = _paper(entry)
                if parsed is None:
                    continue
                base, record = parsed
                if base not in seen_today:
                    seen_today.add(base)
                    new_ids += 1
                # Key by base id so a versioned id joins the stored arXiv item.
                out[base] = _merge(out[base], record) if base in out else record
            if len(rows) < PAGE_SIZE:
                break
            if new_ids == 0:
                # A full page with nothing new means the server is ignoring p.
                raise ValueError(f"hf page at p={page} added no new ids for {day.isoformat()}")
            page += 1
