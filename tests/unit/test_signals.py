"""unit tests for the quality-signal sources: HF upvotes and S2 venue/citations."""

import json
import time
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from paperwatch.sources.hf_daily import HF_URL, MAX_PAGES_PER_DAY, HfDailySource
from paperwatch.sources.semantic_scholar import S2_BATCH_URL, SemanticScholarSource


def _hf_row(pid: str, upvotes: int = 1) -> dict[str, object]:
    # One daily_papers row; the client reads only paper.id and paper.upvotes.
    return {"paper": {"id": pid, "upvotes": upvotes}}


@respx.mock
def test_hf_upvotes_by_base_id() -> None:
    # The fixture is a real daily-papers page trimmed to 3 entries.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/hf_daily.json").read_text())
        )
    )
    up = HfDailySource(httpx.Client()).upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert up["2510.00001"] == 42


@respx.mock
def test_hf_versioned_id_maps_to_base_id() -> None:
    # A version suffix must collapse so the row joins the stored arXiv item.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=[_hf_row("2510.00003v2", 5), _hf_row("2510.00003", 3)]
        )
    )
    up = HfDailySource(httpx.Client()).upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert up == {"2510.00003": 5}


@respx.mock
def test_hf_pages_until_a_short_page() -> None:
    # A full first page must be followed by p=1; a short second page ends the day.
    def responder(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["p"])
        if page == 0:
            return httpx.Response(200, json=[_hf_row(f"2510.{i:05d}") for i in range(100)])
        return httpx.Response(200, json=[_hf_row("2510.00050", 7)])

    route = respx.get(HF_URL).mock(side_effect=responder)
    up = HfDailySource(httpx.Client()).upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert route.call_count == 2
    assert up["2510.00050"] == 7


@respx.mock
def test_hf_queries_each_date() -> None:
    # One request per calendar day, paged from 0 with the API's 100-row cap.
    route = respx.get(HF_URL).mock(return_value=httpx.Response(200, json=[]))
    HfDailySource(httpx.Client()).upvotes(date(2026, 10, 8), date(2026, 10, 9))
    assert [c.request.url.params["date"] for c in route.calls] == [
        "2026-10-08",
        "2026-10-09",
    ]
    assert all(c.request.url.params["limit"] == "100" for c in route.calls)
    assert all(c.request.url.params["p"] == "0" for c in route.calls)


@respx.mock
def test_hf_day_failure_keeps_earlier_days_and_marks_degraded(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2026-10-08":
            return httpx.Response(200, json=[_hf_row("2510.00001", 42)])
        return httpx.Response(503)

    respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    up = src.upvotes(date(2026, 10, 8), date(2026, 10, 9))
    assert up == {"2510.00001": 42}
    assert src.degraded is True
    assert src.failure is not None and "2026-10-09" in src.failure


@respx.mock
def test_hf_clean_call_leaves_degraded_false() -> None:
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=[_hf_row("2510.00001", 42)]))
    src = HfDailySource(httpx.Client())
    src.upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert src.degraded is False
    assert src.failure is None


@respx.mock
def test_hf_skips_a_row_with_non_numeric_upvotes() -> None:
    # One bad row is skipped; the rest of the page and the day still count.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                _hf_row("2510.00001", 42),
                {"paper": {"id": "2510.00002", "upvotes": "lots"}},
            ],
        )
    )
    src = HfDailySource(httpx.Client())
    up = src.upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert up == {"2510.00001": 42}
    assert src.degraded is False


@respx.mock
def test_hf_stops_when_a_full_page_adds_nothing() -> None:
    # A server that ignores p returns the same full page forever; the day is
    # degraded instead of looping.
    rows = [_hf_row(f"2510.{i:05d}") for i in range(100)]
    route = respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    src = HfDailySource(httpx.Client())
    up = src.upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert route.call_count == 2
    assert len(up) == 100
    assert src.degraded is True


@respx.mock
def test_hf_page_cap_bounds_a_day() -> None:
    # Distinct full pages never end on their own; the named cap bounds the day.
    def responder(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["p"])
        return httpx.Response(200, json=[_hf_row(f"2510.{page:02d}{i:04d}") for i in range(100)])

    route = respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    src.upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert route.call_count == MAX_PAGES_PER_DAY
    assert src.degraded is True


def _s2_row(arxiv_id: str, *, citations: int = 0, venue: str = "arXiv.org") -> dict[str, object]:
    # The smallest real answer row: every non-null row names the paper's arXiv id, which is what
    # a row is joined back by.
    return {
        "externalIds": {"ArXiv": arxiv_id},
        "venue": venue,
        "citationCount": citations,
    }


@respx.mock
def test_s2_lookup_handles_nulls_and_empty_venue() -> None:
    # The fixture is hand-authored to the documented batch shape: one venue, one
    # arXiv.org preprint, one null.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/s2_batch.json").read_text())
        )
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001", "2510.00002", "2510.09999"])
    assert out["2510.00001"].venue == "International Conference on Learning Representations"
    assert "ICLR" in out["2510.00001"].venue_aliases
    assert out["2510.00001"].citation_count == 7
    assert out["2510.00002"].venue == "arXiv.org"
    assert "2510.09999" not in out
    # A null row is an id S2 checked and has no record for, so it is not due again tomorrow.
    assert src.checked_ids == {"2510.00001", "2510.00002", "2510.09999"}
    assert src.degraded is False
    assert src.failure is None


@respx.mock
def test_s2_keeps_the_rows_it_answered_when_an_id_is_left_out() -> None:
    # The fixture is a real answer recorded 2026-10-22 over the server's 1321 arXiv ids: the
    # 500-id chunk came back with 499 rows because S2 has no record at all for 2603.17771, so
    # that id is missing rather than null and every later row sits one position early. Pairing
    # by position would give the paper after the gap the next paper's venue.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/s2_batch_short.json").read_text())
        )
    )
    ids = [
        "2603.10564",
        "2603.14360",
        "2603.16557",
        "2603.17771",
        "2603.18104",
        "2603.18712",
        "2603.23101",
        "2603.23844",
    ]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(ids)
    answered = set(ids) - {"2603.17771"}
    assert set(out) == answered
    # The paper S2 does not know stays unanswered, and its neighbours keep their own rows:
    # 2603.18104 is an arXiv preprint, while the ICDE venue one row later in the answer belongs
    # to 2603.18712, and the last id still gets its own empty-venue row.
    assert out["2603.18104"].venue == "arXiv.org" and out["2603.18104"].citation_count == 8
    assert out["2603.18712"].venue == "IEEE International Conference on Data Engineering"
    assert out["2603.23844"].venue is None
    assert out["2603.16557"].citation_count == 9
    # Only the id S2 left out is due again; a short answer of this shape is not a degradation.
    assert src.checked_ids == answered
    assert src.degraded is False
    assert src.failure is None


@respx.mock
def test_s2_ignores_non_list_alternate_names() -> None:
    # A malformed alternate_names string must not be split into characters.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "externalIds": {"ArXiv": "2510.00001"},
                    "venue": "International Conference on Learning Representations",
                    "publicationVenue": {
                        "name": "International Conference on Learning Representations",
                        "alternate_names": "ICLR",
                    },
                    "citationCount": 3,
                }
            ],
        )
    )
    out = SemanticScholarSource(httpx.Client(), None, pause_s=0).lookup(["2510.00001"])
    aliases = out["2510.00001"].venue_aliases
    assert aliases == ("International Conference on Learning Representations",)
    assert "I" not in aliases


@respx.mock
def test_s2_sends_key_and_splits_501_ids() -> None:
    # The API key rides along and the 500-id batch cap splits 501 ids into 500 + 1.
    def responder(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["ids"])
        return httpx.Response(200, json=[None] * count)

    route = respx.post(S2_BATCH_URL).mock(side_effect=responder)
    ids = [f"2510.{i:05d}" for i in range(501)]
    SemanticScholarSource(httpx.Client(), "secret", pause_s=0).lookup(ids)
    assert route.call_count == 2
    sizes = [len(json.loads(c.request.content)["ids"]) for c in route.calls]
    assert sizes == [500, 1]
    assert all(c.request.headers["x-api-key"] == "secret" for c in route.calls)


@respx.mock
def test_s2_degraded_on_503(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(503))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert src.degraded is True
    assert src.failure is not None


@respx.mock
def test_s2_degraded_on_non_json(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(200, text="not-json"))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert src.degraded is True
    assert src.checked_ids == set()


@respx.mock
def test_s2_unattributable_row_degrades_but_keeps_the_paired_rows(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "externalIds": {"ArXiv": "2510.00001", "CorpusId": 1},
                    "venue": "arXiv.org",
                    "citationCount": 0,
                },
                {"venue": "arXiv.org", "citationCount": 0},
            ],
        )
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001", "2510.00002"])
    # A row that names no id we asked about cannot be trusted; it does not cost us the row that
    # did, and the id it might have answered stays due.
    assert out["2510.00001"].venue == "arXiv.org"
    assert "2510.00002" not in out
    assert src.degraded is True
    assert src.checked_ids == {"2510.00001"}


@respx.mock
def test_s2_degraded_on_a_longer_array(monkeypatch) -> None:
    # More rows than ids: zip(strict=True) refuses, so no id is marked checked.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"venue": "arXiv.org", "citationCount": 0},
                {"venue": "arXiv.org", "citationCount": 0},
            ],
        )
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    assert src.degraded is True
    assert src.checked_ids == set()


@respx.mock
def test_s2_degraded_on_a_non_dict_element(monkeypatch) -> None:
    # A scalar where a paper object belongs cannot be read; the chunk fails as a whole.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(200, json=[42]))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    assert src.degraded is True
    assert src.checked_ids == set()


@respx.mock
def test_s2_degraded_on_a_json_null_body(monkeypatch) -> None:
    # A top-level JSON null is not a list of rows, so the chunk fails instead of being trusted.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200, content=b"null", headers={"content-type": "application/json"}
        )
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    assert src.degraded is True
    assert src.checked_ids == set()


@respx.mock
def test_s2_retries_429_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=[_s2_row("2510.00001")]),
        ]
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001"])
    assert route.call_count == 2
    assert out["2510.00001"].venue == "arXiv.org"
    assert src.degraded is False


@respx.mock
def test_s2_honors_retry_after(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    route = respx.post(S2_BATCH_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=[_s2_row("2510.00001")]),
        ]
    )
    SemanticScholarSource(httpx.Client(), None, pause_s=0).lookup(["2510.00001"])
    assert route.call_count == 2
    assert max(sleeps) == 7  # The server's Retry-After wins over the backoff.


@respx.mock
def test_s2_does_not_retry_a_4xx(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(400))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert route.call_count == 1  # A deterministic 4xx is a bug, not a blip.
    assert src.degraded is True


@respx.mock
def test_s2_paces_between_chunks_after_a_failure(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    def responder(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["ids"])
        if count == 500:
            return httpx.Response(503)
        return httpx.Response(200, json=[None] * count)

    respx.post(S2_BATCH_URL).mock(side_effect=responder)
    ids = [f"2510.{i:05d}" for i in range(501)]
    SemanticScholarSource(httpx.Client(), None, pause_s=1.5).lookup(ids)
    # One pause between chunks, even though the first failed, and none after the last.
    assert len([s for s in sleeps if 1.2 <= s <= 1.8]) == 1


@respx.mock
def test_s2_does_not_pause_after_the_last_chunk(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(200, json=[None]))
    SemanticScholarSource(httpx.Client(), None, pause_s=1.5).lookup(["2510.00001"])
    assert sleeps == []


@respx.mock
def test_s2_pause_has_bounded_jitter(monkeypatch) -> None:
    # Ten chunks give nine inter-chunk pauses, each within +/-20% of pause_s.
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(200, json=[None] * 500))
    ids = [f"2510.{i:05d}" for i in range(500 * 10)]
    SemanticScholarSource(httpx.Client(), None, pause_s=10.0).lookup(ids)
    assert len(sleeps) == 9
    assert all(8.0 <= s <= 12.0 for s in sleeps)
    assert len(set(sleeps)) > 1  # The jitter actually varies.


@respx.mock
def test_s2_five_attempts_succeed(monkeypatch) -> None:
    # A fourth refusal needs a fifth attempt; the keyless pool made this common.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=[_s2_row("2510.00001")]),
        ]
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001"])
    assert route.call_count == 5
    assert out["2510.00001"].venue == "arXiv.org"
    assert src.degraded is False


@respx.mock
def test_s2_five_refusals_degrade_and_skip_checked_ids(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(429))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001"])
    assert route.call_count == 5
    assert out == {}
    assert src.degraded is True
    assert src.checked_ids == set()


@respx.mock
@pytest.mark.parametrize(
    ("caller_id", "sent_id"),
    [
        ("2510.00001v1", "ARXIV:2510.00001"),
        ("2510.00001v12", "ARXIV:2510.00001"),
        ("hep-th/9901001v2", "ARXIV:hep-th/9901001"),
        ("2510.00001", "ARXIV:2510.00001"),  # a bare id is sent unchanged
    ],
)
def test_s2_strips_version_suffix_and_maps_back(caller_id: str, sent_id: str) -> None:
    # S2 returns HTTP 400 for a versioned id, so the client strips vN itself.
    seen: list[list[str]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["ids"])
        base = seen[-1][0].removeprefix("ARXIV:")
        return httpx.Response(200, json=[_s2_row(base, citations=2)])

    respx.post(S2_BATCH_URL).mock(side_effect=responder)
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup([caller_id])
    assert seen == [[sent_id]]
    assert out[caller_id].citation_count == 2
    assert src.checked_ids == {caller_id}


@respx.mock
def test_s2_checked_ids_include_null_answers() -> None:
    # A null answer means S2 checked the id and found nothing, so it still counts.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200,
            json=[None, _s2_row("2510.00002", citations=1)],
        )
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001", "2510.00002"])
    assert "2510.00002" in out
    assert src.checked_ids == {"2510.00001", "2510.00002"}


@respx.mock
def test_s2_checked_ids_reset_each_call() -> None:
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(200, json=[None]))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    src.lookup(["2510.00002"])
    assert src.checked_ids == {"2510.00002"}


@respx.mock
def test_s2_two_versions_of_one_id_share_the_one_row() -> None:
    # Both caller ids strip to one ARXIV: id, so S2 answers one row for the paper; both ids map
    # back to it instead of to a position, and the row is never attributed to another paper.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(200, json=[_s2_row("2106.09685", citations=5)])
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2106.09685v1", "2106.09685v2"])
    assert out["2106.09685v1"].citation_count == 5
    assert out["2106.09685v2"].citation_count == 5
    assert src.checked_ids == {"2106.09685v1", "2106.09685v2"}


@respx.mock
def test_s2_checked_ids_keep_answered_chunks_only(monkeypatch) -> None:
    # 501 ids split into 500 + 1; the answered chunk is checked, the refused one is not.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["ids"])
        if count == 500:
            return httpx.Response(200, json=[None] * count)
        return httpx.Response(503)

    respx.post(S2_BATCH_URL).mock(side_effect=responder)
    ids = [f"2510.{i:05d}" for i in range(501)]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(ids)
    assert src.degraded is True
    assert src.checked_ids == set(ids[:500])
    assert ids[500] not in src.checked_ids


@respx.mock
def test_s2_all_unknown_chunk_counts_as_checked_with_no_signals(monkeypatch) -> None:
    # S2 answers a chunk made only of ids it has not indexed yet, routine for papers
    # announced the same day, with this exact 400. Such a chunk is checked and empty,
    # not an outage, so the next chunk still runs and the source stays healthy.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content)["ids"]
        if len(ids) == 500:
            return httpx.Response(
                200,
                json=[_s2_row(i.removeprefix("ARXIV:"), citations=1) for i in ids],
            )
        return httpx.Response(400, json={"error": "No valid paper ids given"})

    respx.post(S2_BATCH_URL).mock(side_effect=responder)
    ids = [f"2510.{i:05d}" for i in range(514)]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(ids)
    assert set(out) == set(ids[:500])
    assert src.degraded is False
    assert src.failure is None
    assert src.checked_ids == set(ids)


@respx.mock
def test_s2_other_400_still_degrades(monkeypatch) -> None:
    # Only the all-unknown answer is benign; any other 400 is a request bug.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(400, json={"error": "Unrecognized field"})
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert src.degraded is True
    assert src.checked_ids == set()
