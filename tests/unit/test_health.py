from paperwatch.ingest.health import Health, assess_run

GOOD = {"fetched_arxiv": 40, "failed": 0, "malformed": 0, "given_up": 0}
WARNING = "feed missed once, retried next run: https://lab.example/feed.xml"


def test_clean_run_is_healthy() -> None:
    h = assess_run("succeeded", GOOD, tokens=1_000, token_cap=200_000, stuck_cursors=[])
    assert h.healthy and h.problems == () and h.warnings == ()


def test_a_warning_never_makes_a_run_unhealthy() -> None:
    # One missed day of a lab feed is worth seeing and nothing more: the next run retries the
    # same window, so the day it missed is not lost.
    h = assess_run(
        "succeeded", GOOD, tokens=1_000, token_cap=200_000, stuck_cursors=[], warnings=[WARNING]
    )
    assert h.healthy
    assert h.problems == () and h.warnings == (WARNING,)


def test_a_warning_is_serialised_and_does_not_hide_a_problem() -> None:
    h = assess_run(
        "partial",
        GOOD,
        tokens=1_000,
        token_cap=200_000,
        stuck_cursors=[],
        warnings=[WARNING],
        arxiv_stopped_early=True,
    )
    assert not h.healthy and h.problems == ("status=partial", "arxiv stopped early")
    assert h.as_dict() == {
        "healthy": False,
        "problems": ["status=partial", "arxiv stopped early"],
        "warnings": [WARNING],
    }
    # A Health built without warnings serialises an empty list.
    assert Health(True, ()).as_dict() == {"healthy": True, "problems": [], "warnings": []}


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


def test_arxiv_expected_false_still_reports_other_problems() -> None:
    # Suppressing the empty-fetch problem must not hide a real failure.
    h = assess_run(
        "partial",
        {"fetched_arxiv": 0, "failed": 1, "malformed": 0, "given_up": 0},
        tokens=1_000,
        token_cap=200_000,
        stuck_cursors=[],
        arxiv_expected=False,
    )
    assert not h.healthy
    assert h.problems == ("status=partial", "1 items failed")


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


def test_a_run_without_follow_ups_keeps_the_old_record() -> None:
    # A manual run, or any run before follow-up passes existed, writes the same record as before:
    # no `followup` key at all, so old readers never see an empty object.
    h = assess_run("succeeded", GOOD, tokens=0, token_cap=0, stuck_cursors=[])
    assert h.followup is None
    assert "followup" not in h.as_dict()


def test_a_provisional_follow_up_record_is_serialised() -> None:
    # The morning check reads this to tell a waiting run from a hung one.
    waiting = {"pass": 2, "of": 40, "next_at": "2026-10-12T06:45:00+00:00", "pending": {"s2": 7}}
    h = assess_run("partial", GOOD, tokens=0, token_cap=0, stuck_cursors=[], followup=waiting)
    assert h.as_dict()["followup"] == waiting
    assert Health(False, ("status=partial",), followup=waiting).followup == waiting


def test_extra_problems_make_the_run_unhealthy() -> None:
    # A finding no count covers, such as S2 still refusing on the second day of an outage.
    message = "s2 still refused after 29 follow-up passes: 152 ids"
    h = assess_run(
        "succeeded",
        GOOD,
        tokens=0,
        token_cap=0,
        stuck_cursors=[],
        extra_problems=[message],
    )
    assert not h.healthy and h.problems == (message,)
