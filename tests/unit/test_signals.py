"""unit tests for the quality-signal sources: HF upvotes and S2 venue/citations."""

import json
import time
from datetime import date
from pathlib import Path

import httpx
import respx

from paperwatch.sources.hf_daily import HF_URL, HfDailySource
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
    assert src.degraded is False
    assert src.failure is None


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


@respx.mock
def test_s2_degraded_on_short_array(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(200, json=[{"venue": "arXiv.org", "citationCount": 0}])
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001", "2510.00002"])
    # The pair already yielded is kept; the missing id is not invented, and the
    # run is marked degraded instead of raising.
    assert out["2510.00001"].venue == "arXiv.org"
    assert "2510.00002" not in out
    assert src.degraded is True


@respx.mock
def test_s2_retries_429_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=[{"venue": "arXiv.org", "citationCount": 0}]),
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
            httpx.Response(200, json=[{"venue": "arXiv.org", "citationCount": 0}]),
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
def test_s2_paces_after_a_failed_chunk(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(503))
    SemanticScholarSource(httpx.Client(), None, pause_s=1.5).lookup(["2510.00001"])
    # The finally-block pause runs even when the chunk failed.
    assert 1.5 in sleeps
