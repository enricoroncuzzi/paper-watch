"""unit tests for the arXiv OAI-PMH source: request shape, paging, mapping, retries."""

import signal
import time
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import httpx
import pytest

from paperwatch.config import ARXIV_OAI_PREFIX
from paperwatch.domain import Profile, RejectedEntry, load_profiles
from paperwatch.sources.arxiv import ArxivOaiSource, _split_authors, arxiv_base_id, oai_sets

FIXTURES = Path("tests/fixtures/oai")
CS_PAGE1 = (FIXTURES / "cs_page1.xml").read_text()
CS_PAGE2 = (FIXTURES / "cs_page2.xml").read_text()
STATML_DAY = (FIXTURES / "statml_day.xml").read_text()
NO_RECORDS = (FIXTURES / "noRecordsMatch.xml").read_text()
BAD_TOKEN = (FIXTURES / "badResumptionToken.xml").read_text()
DELETED = (FIXTURES / "deleted.xml").read_text()
ALL_PROFILES = load_profiles(Path("config/profiles.yaml"))
# Only the agentic profile is cs, so a harness against it makes one set sequence.
AGENTIC = [p for p in ALL_PROFILES if p.name == "agentic-engineering"]
DAY = date(2026, 10, 6)


def _record(
    ident: str,
    *,
    raw_id: str | None = "2510.00001",
    title: str = "An agent paper",
    abstract: str = "A short study.",
    authors: str = "A, B and C",
    categories: str = "cs.AI",
    versions: tuple[tuple[str, str], ...] = (("v1", "Mon, 06 Oct 2026 10:00:00 GMT"),),
    deleted: bool = False,
) -> str:
    # One OAI record with an arXivRaw body; deleted records carry no metadata.
    status = ' status="deleted"' if deleted else ""
    header = (
        f"<header{status}><identifier>{ident}</identifier>"
        "<datestamp>2026-10-06</datestamp></header>"
    )
    if deleted:
        return f"<record>{header}</record>"
    id_el = "" if raw_id is None else f"<id>{raw_id}</id>"
    vers = "".join(f'<version version="{v}"><date>{d}</date></version>' for v, d in versions)
    raw = (
        '<arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">'
        f"{id_el}{vers}<title>{title}</title><authors>{authors}</authors>"
        f"<categories>{categories}</categories><abstract>{abstract}</abstract></arXivRaw>"
    )
    return f"<record>{header}<metadata>{raw}</metadata></record>"


def _page(*records: str, resumption: str = "") -> str:
    # A minimal but real-shaped ListRecords response.
    tok = f"<resumptionToken>{resumption}</resumptionToken>" if resumption else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        "<responseDate>2026-10-16T08:00:00Z</responseDate>"
        '<request verb="ListRecords">https://oaipmh.arxiv.org/oai</request>'
        f"<ListRecords>{''.join(records)}{tok}</ListRecords></OAI-PMH>"
    )


def _client(handler: object) -> httpx.Client:
    # MockTransport keeps every test offline; no test touches the network.
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _harvest(text: str, profiles: list[Profile] | None = None, **kwargs: object) -> list[object]:
    src = ArxivOaiSource(
        _client(lambda request: httpx.Response(200, text=text)),
        profiles or AGENTIC,
        pause_s=0,
        **kwargs,  # type: ignore[arg-type]
    )
    return list(src.harvest(DAY))


def _flat(pages: list[object]) -> list[object]:
    return [item for page in pages for item in page]  # type: ignore[union-attr]


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


def test_oai_sets_for_the_committed_profiles() -> None:
    assert oai_sets(ALL_PROFILES) == ["cs", "stat:stat:ML"]


def test_oai_sets_maps_a_non_cs_category() -> None:
    p = Profile("x", 1.0, ("eess.IV",), ("term",), "desc", 0.5)
    assert oai_sets([p]) == ["eess:eess:IV"]


@pytest.mark.parametrize("category", ["quant-ph", "hep-th", "gr-qc", "physics"])
def test_oai_sets_rejects_a_category_with_no_sub_archive(category: str) -> None:
    # No dot means no sub-archive and no set the current mapping can name.
    p = Profile("x", 1.0, (category,), ("term",), "desc", 0.5)
    with pytest.raises(ValueError, match=category):
        oai_sets([p])


@pytest.mark.parametrize("category", ["astro-ph.CO", "cond-mat.str-el", "nlin.CD", "math-ph.MP"])
def test_oai_sets_rejects_a_physics_umbrella_archive(category: str) -> None:
    # These archives live under arXiv's physics umbrella (physics:<archive>), not their own.
    p = Profile("x", 1.0, (category,), ("term",), "desc", 0.5)
    with pytest.raises(ValueError, match=category):
        oai_sets([p])


def test_oai_sets_allows_a_physics_sub_archive() -> None:
    # physics.soc-ph is a real set as physics:physics:soc-ph; only the umbrella names are wrong.
    p = Profile("x", 1.0, ("physics.soc-ph",), ("term",), "desc", 0.5)
    assert oai_sets([p]) == ["physics:physics:soc-ph"]


def test_source_rejects_a_profile_with_no_oai_set() -> None:
    # The source validates its sets at construction, so a bad profile fails at startup.
    p = Profile("x", 1.0, ("quant-ph",), ("term",), "desc", 0.5)
    with pytest.raises(ValueError, match="quant-ph"):
        ArxivOaiSource(_client(lambda request: httpx.Response(200, text=_page())), [p])


def test_request_shape_then_resumption_token() -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return httpx.Response(200, text=CS_PAGE1 if len(calls) == 1 else CS_PAGE2)

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    list(src.harvest(DAY))
    first = calls[0]
    assert first["verb"] == "ListRecords"
    assert first["metadataPrefix"] == ARXIV_OAI_PREFIX == "arXivRaw"
    assert first["set"] == "cs"
    assert first["from"] == "2026-10-06" and first["until"] == "2026-10-06"
    # No other filter rides along on the first page.
    assert set(first) == {"verb", "metadataPrefix", "set", "from", "until"}
    # The resumption request carries the token and nothing else.
    assert calls[1] == {"verb": "ListRecords", "resumptionToken": "cs-token-2"}


def test_pages_are_yielded_before_the_next_request() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("resumptionToken", "first"))
        if "resumptionToken" not in request.url.params:
            return httpx.Response(200, text=_page(_record("oai:arXiv.org:1"), resumption="tok"))
        return httpx.Response(200, text=_page(_record("oai:arXiv.org:2", raw_id="2510.00002")))

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    gen = src.harvest(DAY)
    first = next(gen)
    assert len(calls) == 1  # the second page is not requested until the first is consumed
    assert [i.external_id for i in first] == ["2510.00001"]  # type: ignore[union-attr]
    second = next(gen)
    assert len(calls) == 2
    assert [i.external_id for i in second] == ["2510.00002"]  # type: ignore[union-attr]
    with pytest.raises(StopIteration):
        next(gen)


def test_word_boundary_bot_does_not_match_robot() -> None:
    xml = _page(
        _record("oai:arXiv.org:1", title="A chat bot for teams"),
        _record("oai:arXiv.org:2", raw_id="2510.00002", title="Robotics and robot arms"),
    )
    titles = [i.title for i in _flat(_harvest(xml))]  # type: ignore[union-attr]
    assert titles == ["A chat bot for teams"]


def test_phrase_matches_across_a_line_break_and_case() -> None:
    xml = _page(_record("oai:arXiv.org:1", title="Tool\n   use and TOOL USE in agents"))
    assert len(_flat(_harvest(xml))) == 1


def test_terms_accept_plurals_but_not_substrings() -> None:
    # The Atom `all:` query it replaces stemmed terms, so a plural must still match; a suffix
    # must not turn "bot" into a match inside "robot".
    xml = _page(
        _record("oai:arXiv.org:1", title="Agents that plan"),
        _record("oai:arXiv.org:2", raw_id="2510.00002", title="A chat bot"),
        _record("oai:arXiv.org:3", raw_id="2510.00003", title="Chat bots for teams"),
        _record("oai:arXiv.org:4", raw_id="2510.00004", title="Harnesses at work"),
        _record("oai:arXiv.org:5", raw_id="2510.00005", title="A robot arm"),
        _record("oai:arXiv.org:6", raw_id="2510.00006", title="Robots and robotics"),
        _record("oai:arXiv.org:7", raw_id="2510.00007", title="A botnet takedown"),
        _record("oai:arXiv.org:8", raw_id="2510.00008", title="Harnessing noise"),
    )
    titles = [i.title for i in _flat(_harvest(xml))]  # type: ignore[union-attr]
    assert titles == [
        "Agents that plan",
        "A chat bot",
        "Chat bots for teams",
        "Harnesses at work",
    ]


def test_category_filter_drops_records_outside_every_profile() -> None:
    xml = _page(_record("oai:arXiv.org:1", title="An agent paper", categories="math.OC"))
    assert _flat(_harvest(xml)) == []


def test_record_matching_two_profiles_is_yielded_once() -> None:
    # cs.LG is in agentic and ml-architectures; stat.ML is in ml-architectures, so the
    # record is cross-listed and matches both profiles but is one item.
    xml = _page(
        _record(
            "oai:arXiv.org:1",
            title="An attention agent",
            categories="cs.LG stat.ML",
        )
    )
    items = _flat(_harvest(xml, ALL_PROFILES))
    assert [i.external_id for i in items] == ["2510.00001"]  # type: ignore[union-attr]


def test_mapping_from_the_real_record() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=CS_PAGE1 if "set=cs" in str(request.url) else CS_PAGE2)

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    items = _flat(list(src.harvest(DAY)))
    item = next(i for i in items if i.external_id == "1909.09849")
    assert item.source == "arxiv"
    assert item.version == 4  # highest of v1..v4
    assert item.published_at == datetime(2019, 9, 21, 16, 5, 31, tzinfo=UTC)  # the v1 date
    assert item.authors[0] == "Mark Rowland" and len(item.authors) == 7
    assert item.categories == ("cs.MA", "cs.AI", "cs.LG", "stat.ML")
    assert item.licence == "CC0-1.0 (metadata)"
    assert item.url == "https://arxiv.org/abs/1909.09849"
    # The v4 title is the one stored, whitespace-normalised.
    assert "\n" not in item.title and item.title.startswith("Multiagent Evaluation")


def test_authors_drop_affiliations_then_split() -> None:
    # An affiliation may itself contain a comma or "and", so a parenthesised group must go
    # before the split, not after it.
    xml = _page(
        _record("oai:arXiv.org:1", authors="Alice (MIT), Bob and Carol (DeepMind)"),
        _record(
            "oai:arXiv.org:2",
            raw_id="2510.00002",
            title="Second agent paper",
            authors="Wenhui Chu (University at Albany, State University of New York)",
        ),
        _record(
            "oai:arXiv.org:3",
            raw_id="2510.00003",
            title="Third agent paper",
            authors="Ziyan Chen (1 and 2)",
        ),
    )
    items = _flat(_harvest(xml))
    assert items[0].authors == ("Alice", "Bob", "Carol")  # type: ignore[union-attr]
    assert items[1].authors == ("Wenhui Chu",)  # type: ignore[union-attr]
    assert items[2].authors == ("Ziyan Chen",)  # type: ignore[union-attr]


def test_authors_stop_on_unbalanced_parentheses() -> None:
    # A stray ")" before a stray "(" never pairs; a split that loops would hang the daily run
    # while it holds the single-run lock, so a regression must fail here, not stall the suite.
    def timeout(signum: int, frame: object) -> None:
        raise TimeoutError("_split_authors did not return")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(5)
    try:
        assert _split_authors("Foo 1), Bar (2") == ("Foo 1)", "Bar (2")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_authors_drop_nested_affiliations() -> None:
    xml = _page(_record("oai:arXiv.org:1", authors="Name (Univ (Dept, Lab), City), Zed"))
    assert _flat(_harvest(xml))[0].authors == ("Name", "Zed")  # type: ignore[union-attr]


def test_cross_listed_record_is_yielded_once_across_sets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("set") == "stat:stat:ML":
            return httpx.Response(200, text=STATML_DAY)
        if "resumptionToken" in params:
            return httpx.Response(200, text=CS_PAGE2)
        return httpx.Response(200, text=CS_PAGE1)

    src = ArxivOaiSource(_client(handler), ALL_PROFILES, pause_s=0)
    ids = [i.external_id for i in _flat(list(src.harvest(DAY)))]  # type: ignore[union-attr]
    # 1909.09849 is in both the cs set and stat:stat:ML; 1506.04782 is filtered by terms.
    assert ids.count("1909.09849") == 1
    assert "1506.04782" not in ids


def test_real_cs_fixture_yields_only_matching_papers() -> None:
    # 2503.21975 is in cs.AI but matches no profile term, and 2301.07757 is outside every profile
    # category and also has no term; the term rule drops both here. The category rule on its own
    # is covered by test_category_filter_drops_records_outside_every_profile.
    def handler(request: httpx.Request) -> httpx.Response:
        if "resumptionToken" in request.url.params:
            return httpx.Response(200, text=CS_PAGE2)
        return httpx.Response(200, text=CS_PAGE1)

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    ids = [i.external_id for i in _flat(list(src.harvest(DAY)))]  # type: ignore[union-attr]
    assert ids == ["1909.09849", "2404.09982", "2610.04539"]


def test_a_doctype_body_is_rejected_before_parsing() -> None:
    # OAI-PMH never carries a DTD; refusing one blocks the billion-laughs shape up front.
    xml = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE OAI-PMH [<!ENTITY x "y">]>'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords/></OAI-PMH>'
    )
    with pytest.raises(ValueError, match="DOCTYPE"):
        _harvest(xml)


def test_no_records_match_yields_nothing_and_does_not_raise() -> None:
    assert _harvest(NO_RECORDS) == []


def test_bad_resumption_token_raises_naming_the_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "resumptionToken" in request.url.params:
            return httpx.Response(200, text=BAD_TOKEN)
        return httpx.Response(200, text=_page(_record("oai:arXiv.org:1"), resumption="stale"))

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    with pytest.raises(ValueError, match="badResumptionToken"):
        list(src.harvest(DAY))


def test_deleted_record_is_skipped_and_not_rejected() -> None:
    src = ArxivOaiSource(
        _client(lambda request: httpx.Response(200, text=DELETED)), AGENTIC, pause_s=0
    )
    ids = [i.external_id for i in _flat(list(src.harvest(DAY)))]  # type: ignore[union-attr]
    assert ids == ["2404.09982"]
    assert src.rejected == []


def test_record_without_id_is_rejected_and_the_rest_of_the_page_survives() -> None:
    xml = _page(
        _record("oai:arXiv.org:bad", raw_id=None, title="Agent one"),
        _record("oai:arXiv.org:good", raw_id="2510.00002", title="Agent two"),
    )
    src = ArxivOaiSource(_client(lambda request: httpx.Response(200, text=xml)), AGENTIC, pause_s=0)
    ids = [i.external_id for i in _flat(list(src.harvest(DAY)))]  # type: ignore[union-attr]
    assert ids == ["2510.00002"]
    assert len(src.rejected) == 1
    assert src.rejected[0].source == "arxiv"
    assert src.rejected[0].ref == "oai:arXiv.org:bad"
    assert src.rejected[0].error


def test_rejected_is_reset_each_harvest() -> None:
    src = ArxivOaiSource(
        _client(lambda request: httpx.Response(200, text=_page(_record("oai:arXiv.org:1")))),
        AGENTIC,
        pause_s=0,
    )
    src.rejected.append(RejectedEntry(source="arxiv", ref="old", error="stale"))
    list(src.harvest(DAY))
    assert src.rejected == []


def test_retry_after_is_honored_then_the_request_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "7"})
        return httpx.Response(200, text=_page(_record("oai:arXiv.org:1")))

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    list(src.harvest(DAY))
    assert len(calls) == 2
    assert 7.0 in sleeps  # the server's Retry-After wins over the fallback wait


def test_two_retryable_failures_raise_after_two_attempts() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429)

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0, retry_wait_s=0)
    with pytest.raises(httpx.HTTPStatusError):
        list(src.harvest(DAY))
    assert len(calls) == 2  # attempts=2: one retry, then the exception propagates


def test_missing_retry_after_uses_the_configured_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, text=_page(_record("oai:arXiv.org:1")))

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0, retry_wait_s=42)
    list(src.harvest(DAY))
    assert max(sleeps) == 42.0


def test_requests_and_retries_are_paced_by_a_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"t": 100.0}
    starts: list[float] = []

    def fake_monotonic() -> float:
        return clock["t"]

    def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        starts.append(clock["t"])
        if len(calls) == 1:
            return httpx.Response(503)  # a retry must also keep the pause
        if "resumptionToken" in request.url.params:
            return httpx.Response(200, text=_page(_record("oai:arXiv.org:2", raw_id="2510.00002")))
        return httpx.Response(200, text=_page(_record("oai:arXiv.org:1"), resumption="tok"))

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=3.0, retry_wait_s=1.0)
    list(src.harvest(DAY))
    assert len(starts) == 3
    # Every consecutive request starts at least pause_s after the previous one, retry included.
    assert all(b - a >= 3.0 for a, b in pairwise(starts))


def test_a_repeated_resumption_token_degrades_arxiv_naming_set_and_day() -> None:
    # A pager that hands the same token back would refetch one page forever: the harvest stops with
    # a problem that names the set and the day, so the morning check says which harvest broke.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # A distinct record per page, so each page carries something new.
        calls.append(request.url.params.get("resumptionToken", "first"))
        return httpx.Response(
            200,
            text=_page(
                _record(f"oai:arXiv.org:{len(calls)}", raw_id=f"2510.{len(calls):05d}"),
                resumption="loop",
            ),
        )

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    pages: list[list[object]] = []
    with pytest.raises(
        ValueError, match=r"resumption token 'loop' repeated for set cs on 2026-10-06"
    ):
        for page in src.harvest(DAY):
            pages.append(page)
    # The page that carried the repeated token was already delivered, and no third request went out.
    assert len(pages) == 2
    assert calls == ["first", "loop"]


def test_a_page_cap_degrades_arxiv_naming_set_and_day(monkeypatch) -> None:
    # The token keeps changing, so only the cap stops the pager; the server is asked page_cap times
    # and never once more.
    monkeypatch.setattr("paperwatch.sources.arxiv.MAX_PAGES_PER_DAY", 3)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # A distinct record per page, so each page carries something new.
        calls.append(request.url.params.get("resumptionToken", "first"))
        return httpx.Response(
            200,
            text=_page(
                _record(f"oai:arXiv.org:{len(calls)}", raw_id=f"2510.{len(calls):05d}"),
                resumption=f"tok{len(calls)}",
            ),
        )

    src = ArxivOaiSource(_client(handler), AGENTIC, pause_s=0)
    pages: list[list[object]] = []
    with pytest.raises(ValueError, match=r"page cap \(3\) reached for set cs on 2026-10-06"):
        for page in src.harvest(DAY):
            pages.append(page)
    assert len(pages) == 3
    assert len(calls) == 3
