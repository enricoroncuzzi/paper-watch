"""Semantic Scholar: venue and citation signals for arXiv papers."""

import time

import httpx
import structlog

from paperwatch.domain import Signals
from paperwatch.sources.base import send_with_retries

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# The batch endpoint rejects more than 500 ids in one request body.
BATCH_SIZE = 500
log = structlog.get_logger()


class SemanticScholarSource:
    def __init__(self, client: httpx.Client, api_key: str | None, pause_s: float = 1.0) -> None:
        self._client, self._pause_s = client, pause_s
        # An API key raises the rate limit; the endpoint works without one.
        self._headers = {"x-api-key": api_key} if api_key else {}
        # Task 9 reads these to record a dead letter and mark the run partial.
        self.degraded = False
        self.failure: str | None = None

    def lookup(self, arxiv_ids: list[str]) -> dict[str, Signals]:
        # The health flags describe this call only.
        self.degraded = False
        self.failure = None
        out: dict[str, Signals] = {}
        for i in range(0, len(arxiv_ids), BATCH_SIZE):
            chunk = arxiv_ids[i : i + BATCH_SIZE]
            try:
                resp = send_with_retries(
                    self._client,
                    "POST",
                    S2_BATCH_URL,
                    headers=self._headers,
                    params={"fields": "externalIds,venue,publicationVenue,citationCount"},
                    json={"ids": [f"ARXIV:{a}" for a in chunk]},
                )
                # S2 answers one element per requested id, null for an unknown id.
                for aid, paper in zip(chunk, resp.json(), strict=True):
                    if paper is None:
                        continue
                    venue = (paper.get("venue") or "").strip() or None
                    pv = paper.get("publicationVenue") or {}
                    # Keep the venue name and every alternate name, e.g. "ICLR".
                    aliases = tuple(
                        a for a in [pv.get("name"), *(pv.get("alternate_names") or [])] if a
                    )
                    out[aid] = Signals(
                        venue=venue,
                        venue_aliases=aliases,
                        citation_count=int(paper.get("citationCount") or 0),
                    )
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                # Optional enrichment: keep earlier chunks, record the failure, and
                # let ingestion continue. A 4xx is a bug, never a retry.
                self.degraded = True
                self.failure = f"s2 chunk at {i}: {exc}"
                log.warning("s2_lookup_failed", error=str(exc), count=len(chunk))
            finally:
                # Pace every chunk, including a failed one, so retries do not hammer.
                time.sleep(self._pause_s)
        return out
