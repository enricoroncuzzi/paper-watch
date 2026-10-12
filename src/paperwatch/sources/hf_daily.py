"""Hugging Face daily papers: community attention as a quality signal."""

from datetime import date, timedelta

import httpx

from paperwatch.sources.arxiv import arxiv_base_id

HF_URL = "https://huggingface.co/api/daily_papers"


class HfDailySource:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def upvotes(self, since: date, until: date) -> dict[str, int]:
        out: dict[str, int] = {}
        day = since
        # The endpoint takes one date at a time, so walk the window day by day.
        while day <= until:
            resp = self._client.get(HF_URL, params={"date": day.isoformat(), "limit": 100})
            # A failed day aborts the batch; the caller decides how to handle it.
            resp.raise_for_status()
            for entry in resp.json():
                paper = entry.get("paper", {})
                if pid := paper.get("id"):
                    # Key by base id so a versioned HF id still joins the arXiv item.
                    base = arxiv_base_id(pid)
                    out[base] = max(out.get(base, 0), int(paper.get("upvotes", 0)))
            day += timedelta(days=1)
        return out
