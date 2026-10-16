from paperwatch.ingest.health import assess_run

GOOD = {"fetched_arxiv": 40, "failed": 0, "malformed": 0, "given_up": 0}


def test_clean_run_is_healthy() -> None:
    h = assess_run("succeeded", GOOD, tokens=1_000, token_cap=200_000, stuck_cursors=[])
    assert h.healthy and h.problems == ()


def test_arxiv_expected_false_suppresses_only_that_problem() -> None:
    # A weekend-only window that fetched nothing is not a broken source.
    h = assess_run(
        "succeeded",
        {"fetched_arxiv": 0, "failed": 0, "malformed": 0, "given_up": 0},
        tokens=1_000,
        token_cap=200_000,
        stuck_cursors=[],
        arxiv_expected=False,
    )
    assert h.healthy and h.problems == ()


def test_each_problem_is_named() -> None:
    h = assess_run(
        "partial",
        {"fetched_arxiv": 0, "failed": 2, "malformed": 1, "signal_failures": 1, "given_up": 1},
        tokens=190_000,
        token_cap=200_000,
        stuck_cursors=["feed:https://x/feed"],
        stale_feeds=["https://y/feed"],
    )
    assert not h.healthy
    assert h.problems == (
        "status=partial",
        "arxiv fetched 0 items",
        "2 items failed",
        "1 items given up after repeated failures",
        "1 malformed source entries",
        "1 signal sources degraded",
        "tokens 190000 >= 80% of cap 200000",
        "cursor not advanced: feed:https://x/feed",
        "feed stale or empty: https://y/feed",
    )
