"""Semantic Scholar: venue and citation signals for arXiv papers."""

import time

import httpx
import structlog

from paperwatch.domain import Signals

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
log = structlog.get_logger()


class SemanticScholarSource:
    def __init__(self, client: httpx.Client, api_key: str | None, pause_s: float = 1.0) -> None:
        self._client, self._pause_s = client, pause_s
        # An API key raises the rate limit; the endpoint works without one.
        self._headers = {"x-api-key": api_key} if api_key else {}

    def lookup(self, arxiv_ids: list[str]) -> dict[str, Signals]:
        out: dict[str, Signals] = {}
        # The batch endpoint caps one request body, so send the ids in chunks.
        for i in range(0, len(arxiv_ids), 500):
            chunk = arxiv_ids[i : i + 500]
            try:
                resp = self._client.post(
                    S2_BATCH_URL,
                    headers=self._headers,
                    params={"fields": "externalIds,venue,publicationVenue,citationCount"},
                    json={"ids": [f"ARXIV:{a}" for a in chunk]},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # Optional enrichment: log and skip the chunk, never abort ingestion.
                log.warning("s2_lookup_failed", error=str(exc), count=len(chunk))
                continue
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
            time.sleep(self._pause_s)
        return out
