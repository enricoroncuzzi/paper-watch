"""Semantic Scholar: venue and citation signals for arXiv papers."""

import random
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from tenacity import RetryCallState

from paperwatch.domain import Signals
from paperwatch.sources.base import is_retryable, send_with_retries

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# The batch endpoint rejects more than 500 ids in one request body.
BATCH_SIZE = 500
# Keyless first attempts are refused about a third of the time with no Retry-After, so a
# fourth refusal is common and five attempts with a 5 s to 60 s backoff clears most chunks.
S2_ATTEMPTS = 5
S2_MIN_WAIT_S = 5.0
S2_MAX_WAIT_S = 60.0
# Keyless capacity is congestion-limited, not average-limited: a slow pacing floor avoids
# the shared API Gateway throttle. Jitter keeps concurrent runs from synchronizing.
S2_PAUSE_S = 10.0
S2_JITTER = 0.2
# Two fully refused chunks in a row mean the keyless pool is refusing this run, not this
# chunk: asking the rest of a 12 500-paper re-check would spend an hour on refusals alone. They
# stay refused (and so due) for the next pass instead.
S2_REFUSED_CHUNKS = 2
_VERSION = re.compile(r"v\d+$")
# S2's 400 body when no id in the chunk is indexed yet, routine for papers announced the
# same day: the chunk was checked and simply has no signals.
_NO_KNOWN_IDS = {"error": "No valid paper ids given"}
log = structlog.get_logger()


def _no_known_ids(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 400:
        return False
    try:
        return bool(exc.response.json() == _NO_KNOWN_IDS)
    except ValueError:
        return False


def _refusal_details(exc: BaseException | None) -> tuple[int | None, str | None]:
    # What the server said about a refusal, for the log: a transport error has neither.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code, exc.response.headers.get("Retry-After")
    return None, None


def _log_refusal(state: RetryCallState) -> None:
    # One line per refused attempt, before the retry waits: the morning check reads these to
    # tell a congested hour from a blocked key, so the status, the server's Retry-After and the
    # wait actually taken are all on the line.
    exc = state.outcome.exception() if state.outcome is not None else None
    status, retry_after = _refusal_details(exc)
    log.warning(
        "s2_attempt_refused",
        status=status,
        retry_after=retry_after,
        wait_s=round(state.upcoming_sleep, 1),
        at=datetime.now(UTC).isoformat(timespec="seconds"),
        error=type(exc).__name__ if exc is not None else None,
    )


def _base_id(arxiv_id: str) -> str:
    # S2 answers a versioned id with HTTP 400 for the whole chunk, so strip vN here too.
    return _VERSION.sub("", arxiv_id)


def _row_arxiv_id(paper: dict[str, Any]) -> str | None:
    # Each row says which paper it is, and that is the only safe way to join it back: measured
    # over the server's 1321 ids on 2026-10-22, an id S2 has no record of (2603.17771) comes
    # back neither as a null nor as a row, so the answer is one row short and every later row
    # sits one position early.
    external = paper.get("externalIds")
    if not isinstance(external, dict):
        return None
    value = external.get("ArXiv")
    return _base_id(value) if isinstance(value, str) and value else None


def _row_label(paper: dict[str, Any]) -> str:
    # A short, stable name for a row we could not attribute, so a logged anomaly says what came
    # back instead of only how much.
    external = paper.get("externalIds")
    if not isinstance(external, dict):
        return "no externalIds"
    arxiv = external.get("ArXiv")
    if isinstance(arxiv, str) and arxiv:
        return arxiv
    return f"externalIds={external!r}"


def _signals(paper: dict[str, Any]) -> Signals:
    venue = (paper.get("venue") or "").strip() or None
    pv = paper.get("publicationVenue") or {}
    # Some venues carry alternate_names as a bare string; only a list
    # is trustworthy, otherwise unpacking would split it into letters.
    alternate = pv.get("alternate_names")
    if not isinstance(alternate, list):
        alternate = []
    # Keep the venue name and every alternate name, e.g. "ICLR".
    aliases = tuple(a for a in [pv.get("name"), *alternate] if a)
    return Signals(
        venue=venue,
        venue_aliases=aliases,
        citation_count=int(paper.get("citationCount") or 0),
    )


class SemanticScholarSource:
    def __init__(
        self, client: httpx.Client, api_key: str | None, pause_s: float = S2_PAUSE_S
    ) -> None:
        self._client, self._pause_s = client, pause_s
        # An API key raises the rate limit; the endpoint works without one.
        self._headers = {"x-api-key": api_key} if api_key else {}
        # Task 9 reads these to record a dead letter and mark the run partial.
        self.degraded = False
        self.failure: str | None = None
        # Ids whose chunk S2 answered with 200, so Task 9 can tell "checked, nothing new"
        # from "not checked". Reset on every call.
        self.checked_ids: set[str] = set()
        # Ids whose chunk ran out of attempts on a retryable error: S2 said "come back later",
        # so these stay due. A deterministic 4xx, an omitted id and the all-unknown 400 are not
        # refusals. Reset on every call.
        self.refused_ids: set[str] = set()
        # A problem that is not a refusal (an answer whose shape this client does not understand,
        # or a deterministic 4xx): retrying cannot fix it, so a run may not let a later clean
        # retry of the refused ids erase it. Reset on every call.
        self.anomaly: str | None = None

    def lookup(self, arxiv_ids: list[str]) -> dict[str, Signals]:
        # The health flags describe this call only.
        self.degraded = False
        self.failure = None
        self.anomaly = None
        self.checked_ids = set()
        self.refused_ids = set()
        out: dict[str, Signals] = {}
        # Consecutive fully refused chunks: a chunk that ends any other way, answered or a bug,
        # ends the streak.
        refused_in_a_row = 0
        stopped = False
        for i in range(0, len(arxiv_ids), BATCH_SIZE):
            chunk = arxiv_ids[i : i + BATCH_SIZE]
            has_more = i + BATCH_SIZE < len(arxiv_ids)
            try:
                try:
                    resp = send_with_retries(
                        self._client,
                        "POST",
                        S2_BATCH_URL,
                        headers=self._headers,
                        params={"fields": "externalIds,venue,publicationVenue,citationCount"},
                        json={"ids": [f"ARXIV:{_base_id(a)}" for a in chunk]},
                        attempts=S2_ATTEMPTS,
                        min_wait_s=S2_MIN_WAIT_S,
                        max_wait_s=S2_MAX_WAIT_S,
                        before_sleep=_log_refusal,
                        # The batch endpoint redirects only where this client will not go (a
                        # moved API), so a redirect is an answer we do not follow, not a detour.
                        follow_redirects=False,
                    )
                except httpx.HTTPStatusError as exc:
                    if not _no_known_ids(exc):
                        raise
                    # The chunk was checked and has no signals, so it is not a refusal.
                    refused_in_a_row = 0
                    self.checked_ids.update(chunk)
                    continue
                # S2 normally answers one row per requested id, null for an id it knows but has no
                # record for. Measured 2026-10-22, an id S2 does not know at all (1 of the
                # server's 1321) is omitted from the answer rather than nulled: a known gap in its
                # batch endpoint. Rows are joined back by the arXiv id they carry, never by
                # position, and an id S2 left out stays unanswered so the next run asks again.
                rows = resp.json()
                if not isinstance(rows, list):
                    raise TypeError(f"S2 answer is not a list: {type(rows).__name__}")
                wanted = [_base_id(a) for a in chunk]
                known = set(wanted)
                by_id: dict[str, dict[str, Any]] = {}
                unpaired = 0
                unpaired_labels: list[str] = []  # the first few, for the message
                for paper in rows:
                    if paper is None:
                        continue
                    if not isinstance(paper, dict):
                        raise TypeError(f"S2 answer row is not an object: {paper!r}")
                    base = _row_arxiv_id(paper)
                    if base is None or base not in known:
                        unpaired += 1
                        if len(unpaired_labels) < 3:
                            unpaired_labels.append(_row_label(paper))
                        continue
                    by_id[base] = paper
                # Position is trusted for the normal null row only where the answer lines up with
                # the request: same length, every row naming the id at its position. A short
                # answer has already shifted, so a null there could not be attributed.
                lines_up = len(rows) == len(chunk) and all(
                    paper is None or _row_arxiv_id(paper) == wanted[pos]
                    for pos, paper in enumerate(rows)
                )
                for pos, aid in enumerate(chunk):
                    paper = by_id.get(wanted[pos])
                    if paper is not None:
                        out[aid] = _signals(paper)
                        self.checked_ids.add(aid)
                    elif lines_up and rows[pos] is None:
                        self.checked_ids.add(aid)
                problems: list[str] = []
                if unpaired:
                    # Up to three values, so a one-off answer is named without a wall of ids.
                    problems.append(
                        f"{unpaired} rows matched no requested id: " + ", ".join(unpaired_labels)
                    )
                # One omitted id is S2 not knowing that paper, but a much shorter answer is not
                # the shape this client understands: flag it and keep whatever rows did pair.
                if len(chunk) - len(rows) > max(2, len(chunk) // 20):
                    problems.append(f"{len(rows)} rows for {len(chunk)} ids")
                if problems:
                    self.degraded = True
                    self.failure = f"s2 chunk at {i}: " + "; ".join(problems)
                    self.anomaly = self.failure
                    log.warning(
                        "s2_answer_anomaly",
                        error=self.failure,
                        unpaired=unpaired,
                        rows=len(rows),
                        ids=len(chunk),
                    )
                # S2 answered this chunk, however odd the answer: the pool is not refusing us.
                refused_in_a_row = 0
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                # Optional enrichment: keep earlier chunks, record the failure, and
                # let ingestion continue. A 4xx is a bug, never a retry.
                self.degraded = True
                self.failure = f"s2 chunk at {i}: {exc}"
                if not is_retryable(exc):
                    # A bug, not the pool refusing us, so it also ends the streak.
                    refused_in_a_row = 0
                    self.anomaly = self.failure
                    log.warning("s2_lookup_failed", error=str(exc), count=len(chunk))
                else:
                    # Every attempt was refused: the ids are still due, so the next pass asks
                    # again instead of them being recorded as checked with no signals.
                    status, retry_after = _refusal_details(exc)
                    self.refused_ids.update(chunk)
                    refused_in_a_row += 1
                    log.warning(
                        "s2_chunk_refused",
                        error=str(exc),
                        status=status,
                        retry_after=retry_after,
                        ids=len(chunk),
                        refused_in_a_row=refused_in_a_row,
                    )
                    if refused_in_a_row >= S2_REFUSED_CHUNKS:
                        # The rest of the run is refused too, in all likelihood: record every
                        # unasked id as refused and stop asking.
                        unasked = arxiv_ids[i + BATCH_SIZE :]
                        self.refused_ids.update(unasked)
                        stopped = True
                        log.warning(
                            "s2_stopped_asking",
                            refused_in_a_row=refused_in_a_row,
                            unasked=len(unasked),
                        )
            finally:
                # Pace between chunks with a small jitter, even after a failure, but not
                # after the last one, and not when the run has stopped asking. Jitter need not
                # be cryptographic, hence S311.
                if has_more and not stopped:
                    factor = random.uniform(1 - S2_JITTER, 1 + S2_JITTER)  # noqa: S311
                    time.sleep(self._pause_s * factor)
            if stopped:
                break
        return out
