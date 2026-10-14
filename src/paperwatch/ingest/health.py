"""Health verdict for one ingestion run: judges what the run did, not only that it ran."""

from collections.abc import Sequence
from dataclasses import dataclass

# Share of the per-run token cap above which a run is flagged as close to its budget.
TOKEN_WARN_RATIO = 0.8


@dataclass(frozen=True)
class Health:
    healthy: bool
    problems: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"healthy": self.healthy, "problems": list(self.problems)}


def assess_run(
    status: str,
    counts: dict[str, int],
    tokens: int,
    token_cap: int,
    stuck_cursors: list[str],
    stale_feeds: Sequence[str] = (),
) -> Health:
    problems: list[str] = []
    if status != "succeeded":
        problems.append(f"status={status}")
    # arXiv is re-read over at least the announcement lag, so an empty fetch means a broken
    # source or query, not a quiet week.
    if counts.get("fetched_arxiv", 0) == 0:
        problems.append("arxiv fetched 0 items")
    if counts.get("failed", 0):
        problems.append(f"{counts['failed']} items failed")
    if counts.get("given_up", 0):
        problems.append(f"{counts['given_up']} items given up after repeated failures")
    if counts.get("malformed", 0):
        problems.append(f"{counts['malformed']} malformed source entries")
    if counts.get("signal_failures", 0):
        problems.append(f"{counts['signal_failures']} signal sources degraded")
    if token_cap and tokens >= TOKEN_WARN_RATIO * token_cap:
        problems.append(f"tokens {tokens} >= {int(TOKEN_WARN_RATIO * 100)}% of cap {token_cap}")
    if stuck_cursors:
        problems.append("cursor not advanced: " + ", ".join(sorted(stuck_cursors)))
    if stale_feeds:
        # A warning for a human: the feed answers but has not published in 90 days.
        problems.append("feed stale or empty: " + ", ".join(sorted(stale_feeds)))
    return Health(not problems, tuple(problems))
