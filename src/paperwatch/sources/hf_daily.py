"""Hugging Face daily papers: community attention as a quality signal."""

from datetime import date, timedelta

import httpx

from paperwatch.sources.arxiv import arxiv_base_id
from paperwatch.sources.base import send_with_retries

HF_URL = "https://huggingface.co/api/daily_papers"
# The endpoint caps one page at 100 rows, so a full page means there is a next one.
PAGE_SIZE = 100


class HfDailySource:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        # Task 9 reads these to record a dead letter and mark the run partial.
        self.degraded = False
        self.failure: str | None = None

    def upvotes(self, since: date, until: date) -> dict[str, int]:
        # The health flags describe this call only.
        self.degraded = False
        self.failure = None
        out: dict[str, int] = {}
        day = since
        while day <= until:
            try:
                self._merge_day(out, day)
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                # A failed day must not discard earlier days; record the outage so
                # the run can dead-letter it and stay visible without blocking.
                self.degraded = True
                self.failure = f"hf {day.isoformat()}: {exc}"
            day += timedelta(days=1)
        return out

    def _merge_day(self, out: dict[str, int], day: date) -> None:
        page = 0
        while True:
            resp = send_with_retries(
                self._client,
                "GET",
                HF_URL,
                params={"date": day.isoformat(), "limit": PAGE_SIZE, "p": page},
            )
            rows = resp.json()
            for entry in rows:
                # A null paper or upvotes is a bad row to skip, not a failed page.
                paper = entry.get("paper") or {}
                base = arxiv_base_id(str(paper.get("id") or ""))
                if base:
                    # Key by base id so a versioned id joins the stored arXiv item.
                    out[base] = max(out.get(base, 0), int(paper.get("upvotes") or 0))
            if len(rows) < PAGE_SIZE:
                break
            page += 1
