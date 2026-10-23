"""Health verdict for one ingestion run: judges what the run did, not only that it ran."""

from collections.abc import Sequence
from dataclasses import dataclass

# Share of the per-run token cap above which a run is flagged as close to its budget.
TOKEN_WARN_RATIO = 0.8


@dataclass(frozen=True)
class Health:
    healthy: bool
    problems: tuple[str, ...]
    # Findings worth seeing that do not make the run bad, such as one missed lab-feed day.
    warnings: tuple[str, ...] = ()
    # The follow-up loop's own state, so the morning check can tell a run that is waiting from one
    # that hung: `{"pass": n, "of": N, "next_at": ISO, "pending": {"s2": n}}` while it waits,
    # `{"passes": n, "stopped": why}` once it ended. None when the run ran no follow-up pass.
    followup: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "healthy": self.healthy,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
        }
        # A run without follow-ups writes the same record as before, so old readers and old rows
        # stay compatible in both directions.
        if self.followup is not None:
            out["followup"] = self.followup
        return out


def assess_run(
    status: str,
    counts: dict[str, int],
    tokens: int,
    token_cap: int,
    stuck_cursors: list[str],
    stale_feeds: Sequence[str] = (),
    *,
    warnings: Sequence[str] = (),
    arxiv_expected: bool = True,
    arxiv_stopped_early: bool = False,
    followup: dict[str, object] | None = None,
    extra_problems: Sequence[str] = (),
) -> Health:
    problems: list[str] = []
    if status != "succeeded":
        problems.append(f"status={status}")
    # arXiv announces Sunday-Thursday evenings, so a weekend-only window that fetched nothing is
    # a quiet day, not a broken source; a weekday window with no items still is.
    if arxiv_expected and counts.get("fetched_arxiv", 0) == 0:
        problems.append("arxiv fetched 0 items")
    if arxiv_stopped_early:
        # Some days completed before the failure, so the cursor did partly advance.
        problems.append("arxiv stopped early")
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
    # A finding the pipeline knows and no count covers, such as Semantic Scholar still refusing
    # after every follow-up pass on the second day of an outage.
    problems.extend(extra_problems)
    return Health(not problems, tuple(problems), tuple(warnings), followup)
