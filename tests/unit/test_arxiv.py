"""unit tests for the arXiv source: version dedupe and query window."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

from paperwatch.domain import load_profiles
from paperwatch.sources.arxiv import ARXIV_URL, ArxivSource, arxiv_base_id

# The recorded Atom response; tests never touch the network.
FIXTURE = Path("tests/fixtures/arxiv_week.xml").read_text()


def test_base_id_strips_version() -> None:
    # Review Focus 3: v1 and v2 of a paper must collapse to one base id.
    assert arxiv_base_id("http://arxiv.org/abs/2510.00001v2") == "2510.00001"
    assert arxiv_base_id("2510.00001v1") == "2510.00001"
    assert arxiv_base_id("2510.00001") == "2510.00001"


@respx.mock
def test_fetch_parses_and_dedupes_versions() -> None:
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=FIXTURE))
    profiles = load_profiles(Path("config/profiles.yaml"))[:1]
    src = ArxivSource(httpx.Client(), profiles, page_size=100, pause_s=0)
    items = src.fetch(datetime(2025, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))
    ids = [i.external_id for i in items]
    # Five entries, two of them versions of one paper, so four distinct items.
    assert len(ids) == len(set(ids)) == 4
    item = next(i for i in items if i.external_id == "2510.00001")
    assert item.source == "arxiv"
    assert item.licence == "CC0-1.0 (metadata)"
    assert item.url == "https://arxiv.org/abs/2510.00001"
    assert item.text and item.title and item.authors


@respx.mock
def test_query_carries_submitted_date_window() -> None:
    route = respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=FIXTURE))
    profiles = load_profiles(Path("config/profiles.yaml"))[:1]
    ArxivSource(httpx.Client(), profiles, pause_s=0).fetch(
        datetime(2026, 10, 3, tzinfo=UTC), datetime(2026, 10, 10, tzinfo=UTC)
    )
    q = route.calls[0].request.url.params["search_query"]
    assert "submittedDate:[202610030000 TO 202610100000]" in q
