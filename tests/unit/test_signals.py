"""unit tests for the quality-signal sources: HF upvotes and S2 venue/citations."""

import json
from datetime import date
from pathlib import Path

import httpx
import respx

from paperwatch.sources.hf_daily import HF_URL, HfDailySource
from paperwatch.sources.semantic_scholar import S2_BATCH_URL, SemanticScholarSource


@respx.mock
def test_hf_upvotes_by_base_id() -> None:
    # The fixture holds a real daily-papers page trimmed to 3 entries, upvotes included.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/hf_daily.json").read_text())
        )
    )
    up = HfDailySource(httpx.Client()).upvotes(date(2026, 10, 9), date(2026, 10, 9))
    assert up["2510.00001"] == 42


@respx.mock
def test_s2_lookup_handles_nulls_and_empty_venue() -> None:
    # The fixture is a real batch shape: one venue, one arXiv.org preprint, one null.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/s2_batch.json").read_text())
        )
    )
    out = SemanticScholarSource(httpx.Client(), None, pause_s=0).lookup(
        ["2510.00001", "2510.00002", "2510.09999"]
    )
    assert out["2510.00001"].venue == "International Conference on Learning Representations"
    assert "ICLR" in out["2510.00001"].venue_aliases
    assert out["2510.00001"].citation_count == 7
    assert out["2510.00002"].venue == "arXiv.org"
    assert "2510.09999" not in out


@respx.mock
def test_s2_failure_returns_empty_not_raise() -> None:
    # Enrichment is optional: a source failure degrades to no signal, it never aborts.
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(503))
    assert SemanticScholarSource(httpx.Client(), None, pause_s=0).lookup(["2510.00001"]) == {}
