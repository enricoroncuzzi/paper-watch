"""Hugging Face daily papers: community attention as a quality signal."""

from datetime import date, timedelta

import httpx
import structlog

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
log = structlog.get_logger()


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
        failed_in_a_row = 0
        day = since
        while day <= until:
            try:
                self._merge_day(out, day)
                failed_in_a_row = 0
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
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
            day += timedelta(days=1)
        return out

    def _merge_day(self, out: dict[str, int], day: date) -> None:
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
            rows = resp.json()
            new_ids = 0
            for entry in rows:
                # A null paper is a bad row to skip, not a failed page.
                paper = entry.get("paper") or {}
                base = arxiv_base_id(str(paper.get("id") or ""))
                if not base:
                    continue
                try:
                    upvotes = int(paper.get("upvotes") or 0)
                except (TypeError, ValueError):
                    # A single malformed row is skipped, not the whole day.
                    continue
                if base not in seen_today:
                    seen_today.add(base)
                    new_ids += 1
                # Key by base id so a versioned id joins the stored arXiv item.
                out[base] = max(out.get(base, 0), upvotes)
            if len(rows) < PAGE_SIZE:
                break
            if new_ids == 0:
                # A full page with nothing new means the server is ignoring p.
                raise ValueError(f"hf page at p={page} added no new ids for {day.isoformat()}")
            page += 1
