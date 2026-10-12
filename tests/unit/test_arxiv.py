"""unit tests for the arXiv source: version dedupe, paging, and query window."""

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from paperwatch.domain import load_profiles
from paperwatch.sources.arxiv import ARXIV_URL, ArxivSource, arxiv_base_id

# The recorded Atom response; tests never touch the network.
FIXTURE = Path("tests/fixtures/arxiv_week.xml").read_text()
# The first profile is enough for one query per fetch in these tests.
PROFILES = load_profiles(Path("config/profiles.yaml"))[:1]


def _atom(total: int, *ids: str) -> str:
    # Minimal Atom feed with an opensearch total, used to drive the paging branches.
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/{i}</id>"
        f"<title>title {i}</title><summary>abstract of {i}</summary>"
        f"<author><name>Author</name></author>"
        f"<published>2026-10-08T17:59:59Z</published></entry>"
        for i in ids
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"
    )


def test_base_id_strips_version() -> None:
    # Review Focus 3: v1 and v2 of a paper must collapse to one base id.
    assert arxiv_base_id("http://arxiv.org/abs/2510.00001v2") == "2510.00001"
    assert arxiv_base_id("2510.00001v1") == "2510.00001"
    assert arxiv_base_id("2510.00001") == "2510.00001"
    # Old-style ids keep their archive prefix and subdirectory after the version.
    assert arxiv_base_id("cs/0601001v2") == "cs/0601001"
    assert arxiv_base_id("http://arxiv.org/abs/cs/0601001v2") == "cs/0601001"
    assert arxiv_base_id("quant-ph/0201001v12") == "quant-ph/0201001"
    assert arxiv_base_id("http://arxiv.org/abs/math.GT/0301001v1") == "math.GT/0301001"


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
    # The fixture lists v2 first; the v2 abstract must be the survivor.
    assert "explicit planner" in item.text


@respx.mock
def test_dedupe_keeps_highest_version_when_v1_first() -> None:
    # v1 before v2 in the feed: version order, not page order, picks the survivor.
    respx.get(ARXIV_URL).mock(
        return_value=httpx.Response(200, text=_atom(2, "2510.00030v1", "2510.00030v2"))
    )
    src = ArxivSource(httpx.Client(), PROFILES, page_size=100, pause_s=0)
    items = src.fetch(datetime(2025, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))
    assert len(items) == 1
    assert items[0].text == "abstract of 2510.00030v2"


@respx.mock
def test_query_carries_submitted_date_window() -> None:
    route = respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=FIXTURE))
    profiles = load_profiles(Path("config/profiles.yaml"))[:1]
    ArxivSource(httpx.Client(), profiles, pause_s=0).fetch(
        datetime(2026, 10, 3, tzinfo=UTC), datetime(2026, 10, 10, tzinfo=UTC)
    )
    q = route.calls[0].request.url.params["search_query"]
    assert "submittedDate:[202610030000 TO 202610100000]" in q


@respx.mock
def test_query_window_converts_aware_times_to_utc() -> None:
    # 06:00 in Rome (CEST, UTC+2) is the 04:00 GMT stamp arXiv expects.
    route = respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=FIXTURE))
    rome = ZoneInfo("Europe/Rome")
    ArxivSource(httpx.Client(), PROFILES, pause_s=0).fetch(
        datetime(2026, 10, 3, 6, 0, tzinfo=rome), datetime(2026, 10, 10, 6, 0, tzinfo=rome)
    )
    q = route.calls[0].request.url.params["search_query"]
    assert "submittedDate:[202610030400 TO 202610100400]" in q


@respx.mock
def test_pagination_stops_at_total_results() -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        # Two singleton pages; the total says stop after the second.
        start = int(request.url.params["start"])
        ids = [f"2510.0001{start}v1"] if start < 2 else []
        return httpx.Response(200, text=_atom(2, *ids))

    route = respx.get(ARXIV_URL).mock(side_effect=responder)
    src = ArxivSource(httpx.Client(), PROFILES, page_size=1, pause_s=0)
    items = src.fetch(datetime(2025, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))
    assert route.call_count == 2
    assert {i.external_id for i in items} == {"2510.00010", "2510.00011"}


@respx.mock
def test_short_page_with_higher_total_raises() -> None:
    # totalResults says 50 but the page holds 1: fail so the run dead-letters it.
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=_atom(50, "2510.00020v1")))
    src = ArxivSource(httpx.Client(), PROFILES, page_size=2, pause_s=0)
    with pytest.raises(ValueError, match="short"):
        src.fetch(datetime(2025, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))


@respx.mock
def test_malformed_entry_raises_with_id() -> None:
    # No published date: parsing fails, and the error must name the entry.
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><id>http://arxiv.org/abs/2510.00099v1</id>"
        "<title>broken</title><summary>no date</summary></entry></feed>"
    )
    respx.get(ARXIV_URL).mock(return_value=httpx.Response(200, text=xml))
    src = ArxivSource(httpx.Client(), PROFILES, page_size=100, pause_s=0)
    with pytest.raises(ValueError, match=r"2510\.00099v1"):
        src.fetch(datetime(2025, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))
