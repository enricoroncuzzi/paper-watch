"""Semantic Scholar: venue and citation signals for arXiv papers."""

import random
import re
import time

import httpx
import structlog

from paperwatch.domain import Signals
from paperwatch.sources.base import send_with_retries

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


def _base_id(arxiv_id: str) -> str:
    # S2 answers a versioned id with HTTP 400 for the whole chunk, so strip vN here too.
    return _VERSION.sub("", arxiv_id)


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

    def lookup(self, arxiv_ids: list[str]) -> dict[str, Signals]:
        # The health flags describe this call only.
        self.degraded = False
        self.failure = None
        self.checked_ids = set()
        out: dict[str, Signals] = {}
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
                    )
                except httpx.HTTPStatusError as exc:
                    if not _no_known_ids(exc):
                        raise
                    self.checked_ids.update(chunk)
                    continue
                # S2 answers one element per requested id, null for an unknown id. The
                # results are mapped back to the caller's ids, version suffix and all.
                for aid, paper in zip(chunk, resp.json(), strict=True):
                    if paper is None:
                        continue
                    venue = (paper.get("venue") or "").strip() or None
                    pv = paper.get("publicationVenue") or {}
                    # Some venues carry alternate_names as a bare string; only a list
                    # is trustworthy, otherwise unpacking would split it into letters.
                    alternate = pv.get("alternate_names")
                    if not isinstance(alternate, list):
                        alternate = []
                    # Keep the venue name and every alternate name, e.g. "ICLR".
                    aliases = tuple(a for a in [pv.get("name"), *alternate] if a)
                    out[aid] = Signals(
                        venue=venue,
                        venue_aliases=aliases,
                        citation_count=int(paper.get("citationCount") or 0),
                    )
                self.checked_ids.update(chunk)
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                # Optional enrichment: keep earlier chunks, record the failure, and
                # let ingestion continue. A 4xx is a bug, never a retry.
                self.degraded = True
                self.failure = f"s2 chunk at {i}: {exc}"
                log.warning("s2_lookup_failed", error=str(exc), count=len(chunk))
            finally:
                # Pace between chunks with a small jitter, even after a failure, but not
                # after the last one. Jitter need not be cryptographic, hence S311.
                if has_more:
                    factor = random.uniform(1 - S2_JITTER, 1 + S2_JITTER)  # noqa: S311
                    time.sleep(self._pause_s * factor)
        return out
