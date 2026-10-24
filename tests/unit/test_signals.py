"""unit tests for the quality-signal sources: HF upvotes and S2 venue/citations."""

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from structlog.testing import capture_logs

from paperwatch.config import HF_ORG_MAX_CHARS, INT4_MAX
from paperwatch.sources.hf_daily import (
    HF_URL,
    MAX_PAGES_PER_DAY,
    PROJECT_PAGE_MAX_CHARS,
    HfDailySource,
    HfPaper,
)
from paperwatch.sources.semantic_scholar import (
    S2_BATCH_URL,
    S2_REFUSED_CHUNKS,
    SemanticScholarSource,
)


def _hf_row(pid: str, upvotes: int = 1) -> dict[str, object]:
    # One daily_papers row; the client reads paper.id, its upvotes and the optional extras.
    return {"paper": {"id": pid, "upvotes": upvotes}}


def _upvotes(out: dict[str, HfPaper]) -> dict[str, int]:
    # The existing tests assert the upvote map; the client now also returns the extras.
    return {key: paper.upvotes for key, paper in out.items()}


@respx.mock
def test_hf_upvotes_by_base_id() -> None:
    # The fixture is a real daily-papers page trimmed to 3 entries.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/hf_daily.json").read_text())
        )
    )
    up = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert up["2510.00001"].upvotes == 42


@respx.mock
def test_hf_papers_keep_the_extras() -> None:
    # The daily listing carries the repo, stars, project page, comments and organisation; the
    # client keeps them so the page and the ranking can use them without another request.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=json.loads(Path("tests/fixtures/hf_daily.json").read_text())
        )
    )
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"] == HfPaper(
        upvotes=42,
        comments=3,
        github_repo="https://github.com/MirroS-Lab/AgentGarten",
        github_stars=156,
        project_page="https://mirros-lab.github.io/agent-garten/",
        organization="MirroS",
        listed_at=datetime(2026, 10, 9, tzinfo=UTC),
    )


@respx.mock
def test_hf_extras_drop_non_github_repo_links() -> None:
    # A hostile or malformed link from Hugging Face is stored as no link at all, so nothing
    # later can render it; only an https GitHub repository URL survives.
    long_page = "https://example.com/" + "a" * 2048
    rows = [
        {
            "paper": {
                "id": "2510.00001",
                "upvotes": 1,
                "githubRepo": "javascript:alert(1)",
                "projectPage": "javascript:alert(1)",
            }
        },
        {
            "paper": {
                "id": "2510.00002",
                "upvotes": 1,
                "githubRepo": "http://github.com/a/b",
                "projectPage": "http://example.com/a",
            }
        },
        {
            "paper": {
                "id": "2510.00003",
                "upvotes": 1,
                "githubRepo": "https://evil.example/a/b",
                "projectPage": long_page,
            }
        },
        {
            "paper": {
                "id": "2510.00004",
                "upvotes": 1,
                "githubRepo": "https://github.com/MirroS-Lab/AgentGarten/",
                "projectPage": "https://example.com/page",
            }
        },
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    for dropped in ("2510.00001", "2510.00002", "2510.00003"):
        assert out[dropped].github_repo is None
        assert out[dropped].project_page is None
    assert out["2510.00004"].github_repo == "https://github.com/MirroS-Lab/AgentGarten/"
    assert out["2510.00004"].project_page == "https://example.com/page"


@respx.mock
def test_hf_out_of_range_numbers_become_none() -> None:
    # The columns are int4: a number they cannot hold is dropped like any other malformed extra,
    # so one hostile row cannot fail the whole listing write for every paper.
    rows = [
        {
            "paper": {
                "id": "2510.00001",
                "upvotes": 42,
                "githubStars": INT4_MAX + 1,
                "numComments": -1,
            }
        },
        # An upvote count the column cannot hold skips the row, as a malformed one does.
        {"paper": {"id": "2510.00002", "upvotes": INT4_MAX + 1}},
        {"paper": {"id": "2510.00003", "upvotes": 7, "githubStars": INT4_MAX}},
        # _as_int must also refuse a value int() cannot take at all.
        {"paper": {"id": "2510.00004", "upvotes": 4, "githubStars": []}},
        # A negative count is malformed, and keeping it would store 0 through merge_signals.
        {"paper": {"id": "2510.00005", "upvotes": -1}},
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].upvotes == 42
    assert out["2510.00001"].github_stars is None
    assert out["2510.00001"].comments is None
    assert "2510.00002" not in out
    assert out["2510.00003"].github_stars == INT4_MAX
    assert out["2510.00004"].github_stars is None
    assert out["2510.00004"].upvotes == 4
    assert "2510.00005" not in out


@respx.mock
def test_hf_an_infinite_number_is_dropped() -> None:
    # JSON's "1e999" parses to infinity; int(inf) raises OverflowError, which is a bad extra, not
    # a failed day.
    content = b'[{"paper": {"id": "2510.00001", "upvotes": 3, "githubStars": 1e999}}]'
    respx.get(HF_URL).mock(return_value=httpx.Response(200, content=content))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].upvotes == 3
    assert out["2510.00001"].github_stars is None
    assert src.degraded is False


@respx.mock
def test_hf_clips_an_over_long_organisation() -> None:
    # The column holds HF_ORG_MAX_CHARS; clipping here is what keeps a 300-character name from
    # failing the write (the store clips too, for a caller that does not come through the client).
    long_org = "O" * (HF_ORG_MAX_CHARS + 44)
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"paper": {"id": "2510.00001", "upvotes": 1, "organization": {"name": long_org}}}
            ],
        )
    )
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].organization == long_org[:HF_ORG_MAX_CHARS]


@respx.mock
def test_hf_the_newest_days_extras_win_and_the_counts_take_the_max() -> None:
    # Days are walked newest first, so the first non-null extra is the newest one: an older page
    # with a different value must not replace it, while the counts take the maximum across the
    # window and the listing time is the earliest.
    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2026-10-09":
            return httpx.Response(
                200,
                json=[
                    {
                        "paper": {
                            "id": "2510.00001",
                            "upvotes": 10,
                            "githubStars": 12,
                            "githubRepo": "https://github.com/new/repo",
                            "projectPage": "https://example.com/new",
                            "organization": {"fullname": "NewOrg"},
                            "submittedOnDailyAt": "2026-10-09T00:00:00.000Z",
                        },
                        "numComments": 5,
                    }
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "paper": {
                        "id": "2510.00001",
                        "upvotes": 5,
                        "githubStars": 10,
                        "githubRepo": "https://github.com/old/repo",
                        "projectPage": "https://example.com/old",
                        "organization": {"fullname": "OldOrg"},
                        "submittedOnDailyAt": "2026-10-08T00:00:00.000Z",
                    },
                    "numComments": 2,
                }
            ],
        )

    respx.get(HF_URL).mock(side_effect=responder)
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 8), date(2026, 10, 9))
    assert out == {
        "2510.00001": HfPaper(
            upvotes=10,
            comments=5,
            github_repo="https://github.com/new/repo",
            github_stars=12,
            project_page="https://example.com/new",
            organization="NewOrg",
            listed_at=datetime(2026, 10, 8, tzinfo=UTC),
        )
    }


@respx.mock
def test_hf_repo_link_rejects_a_look_alike_host_and_an_extra_path() -> None:
    # The host dot is escaped and the whole value must match: githubXcom is another host, and a
    # path below owner/repo is not a repository URL.
    rows = [
        {"paper": {"id": "2510.00001", "upvotes": 1, "githubRepo": "https://githubXcom/a/b"}},
        {"paper": {"id": "2510.00002", "upvotes": 1, "githubRepo": "https://github.com/a/b/c"}},
        {"paper": {"id": "2510.00003", "upvotes": 1, "githubRepo": "https://github.com/a/b\n"}},
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert all(paper.github_repo is None for paper in out.values())


@respx.mock
def test_hf_a_hostile_listing_time_is_none() -> None:
    # An offset can push the instant outside years 1..9999, which the driver cannot read back; the
    # value is dropped and the neighbouring rows keep their extras.
    rows = [
        {
            "paper": {
                "id": "2510.00001",
                "upvotes": 1,
                "submittedOnDailyAt": "0001-01-01T00:00:00+05:00",
            }
        },
        {
            "paper": {
                "id": "2510.00002",
                "upvotes": 2,
                "submittedOnDailyAt": "9999-12-31T23:59:59-05:00",
            }
        },
        {
            "paper": {
                "id": "2510.00003",
                "upvotes": 3,
                "submittedOnDailyAt": "0001-01-01T00:00:00Z",
            }
        },
        {
            "paper": {
                "id": "2510.00004",
                "upvotes": 4,
                "submittedOnDailyAt": "9999-12-31T23:59:59Z",
            }
        },
        {
            "paper": {
                "id": "2510.00005",
                "upvotes": 5,
                "submittedOnDailyAt": "1970-01-01T00:00:00Z",
            }
        },
        {
            "paper": {
                "id": "2510.00006",
                "upvotes": 6,
                "submittedOnDailyAt": "2026-10-09T05:00:00+05:00",
            }
        },
        # The span is inclusive at both ends, and one second outside either end is dropped.
        {
            "paper": {
                "id": "2510.00007",
                "upvotes": 7,
                "submittedOnDailyAt": "2019-12-31T23:59:59Z",
            }
        },
        {
            "paper": {
                "id": "2510.00008",
                "upvotes": 8,
                "submittedOnDailyAt": "2020-01-01T00:00:00Z",
            }
        },
        {
            "paper": {
                "id": "2510.00009",
                "upvotes": 9,
                "submittedOnDailyAt": "2100-01-01T00:00:00Z",
            }
        },
        {
            "paper": {
                "id": "2510.00010",
                "upvotes": 10,
                "submittedOnDailyAt": "2100-01-01T00:00:01Z",
            }
        },
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    for dropped in (
        "2510.00001",
        "2510.00002",
        "2510.00003",
        "2510.00004",
        "2510.00005",
        "2510.00007",
        "2510.00010",
    ):
        assert out[dropped].listed_at is None
    # Both ends of the span are kept, so the bounds are inclusive and exactly 2020 to 2100.
    assert out["2510.00008"].listed_at == datetime(2020, 1, 1, tzinfo=UTC)
    assert out["2510.00009"].listed_at == datetime(2100, 1, 1, tzinfo=UTC)
    # The normal row keeps its instant, normalised to UTC whatever offset it arrived with.
    assert out["2510.00006"].listed_at == datetime(2026, 10, 9, tzinfo=UTC)
    assert out["2510.00006"].listed_at.utcoffset() == timedelta(0)
    assert src.degraded is False


@respx.mock
def test_hf_a_non_list_answer_is_a_failed_day() -> None:
    # A 200 with an error envelope or a wrapped list is API drift: it must stay a visible failed
    # day, not look like a day that legitimately listed nothing.
    for body in (
        b'{"error": "rate limited"}',
        b'{"papers": [{"paper": {"id": "2510.00001", "upvotes": 5}}]}',
        b'"oops"',
    ):

        def responder(request: httpx.Request, body: bytes = body) -> httpx.Response:
            if request.url.params["date"] == "2026-10-09":
                return httpx.Response(200, content=body)
            return httpx.Response(200, json=[{"paper": {"id": "2510.00002", "upvotes": 7}}])

        respx.get(HF_URL).mock(side_effect=responder)
        src = HfDailySource(httpx.Client())
        out = src.papers(date(2026, 10, 8), date(2026, 10, 9))
        assert out == {"2510.00002": HfPaper(upvotes=7)}
        assert src.degraded is True
        assert src.failure is not None
        respx.reset()


@respx.mock
def test_hf_a_deeply_nested_value_degrades_the_day() -> None:
    # A page can nest JSON deeper than Python can parse: that day is degraded, the earlier day is
    # kept and papers() returns instead of failing the whole run.
    nested = b'[{"paper": {"id": "2510.00001", "upvotes": 1, "x": ' + b"[" * 100_000
    nested += b"0" + b"]" * 100_000 + b"}}]"

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2026-10-09":
            return httpx.Response(200, content=nested)
        return httpx.Response(200, json=[{"paper": {"id": "2510.00002", "upvotes": 7}}])

    respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 8), date(2026, 10, 9))
    assert out == {"2510.00002": HfPaper(upvotes=7)}
    assert src.degraded is True


@respx.mock
def test_hf_a_huge_json_integer_is_a_bad_extra() -> None:
    # json.loads refuses an integer of more than 4,300 digits; one such value is dropped as an
    # out-of-range count and the rest of the page still counts.
    huge = "9" * 5_000
    body = (
        '[{"paper": {"id": "2510.00001", "upvotes": 1, "githubStars": '
        + huge
        + '}, "x": '
        + huge
        + '}, {"paper": {"id": "2510.00002", "upvotes": 2, "githubStars": 7}}]'
    ).encode()
    respx.get(HF_URL).mock(return_value=httpx.Response(200, content=body))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].github_stars is None
    assert out["2510.00002"].github_stars == 7
    assert src.degraded is False


@pytest.mark.parametrize(
    ("arxiv_id", "kept"),
    [
        ("2510.12345", True),
        ("2510.1234", True),
        ("2510.12345v3", True),
        ("hep-th/9901001", True),
        ("2510.123", False),
        ("2510.123456", False),
        ("-/1234567", False),
        ("hep-th/990100", False),
        ("hep-th/99010011", False),
        # Fullwidth digits are not an arXiv id: the class stays ASCII.
        ("\uff12\uff15\uff11\uff10.\uff11\uff12\uff13\uff14\uff15", False),
    ],
)
@respx.mock
def test_hf_arxiv_id_shapes(arxiv_id: str, kept: bool) -> None:
    # The exact id shape is pinned: both real styles are accepted and everything else is skipped.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(200, json=[{"paper": {"id": arxiv_id, "upvotes": 1}}])
    )
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert (arxiv_id.removesuffix("v3") in out) is kept


@respx.mock
def test_hf_hostile_and_huge_ids_skip_only_their_row() -> None:
    # Only a real arXiv id becomes a key: a stored external_id is matched by it, and a hostile or
    # enormous one would cost the whole listing write for every paper.
    good = "2510.00005"
    rows = [
        {"paper": {"id": "2510.1\x00", "upvotes": 1}},
        {"paper": {"id": "2510.1\ud800", "upvotes": 1}},
        {"paper": {"id": "2510." + "9" * 200_000, "upvotes": 1}},
        {"paper": {"id": good, "upvotes": 7, "githubStars": 3}},
    ]
    body = json.dumps(rows).encode()
    respx.get(HF_URL).mock(return_value=httpx.Response(200, content=body))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert list(out) == [good]
    assert out[good].github_stars == 3
    assert src.degraded is False


@respx.mock
def test_hf_a_row_that_is_not_an_object_skips_only_itself() -> None:
    # A row (or a paper) of the wrong shape is a bad row; the neighbouring good rows still count.
    rows = [
        {"paper": {"id": "2510.00001", "upvotes": 1, "githubStars": 1}},
        None,
        "not an object",
        42,
        {"paper": ["not", "an", "object"]},
        {"paper": "not an object"},
        {"paper": {"id": "2510.00002", "upvotes": 2, "githubStars": 2}},
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert set(out) == {"2510.00001", "2510.00002"}
    assert out["2510.00002"].github_stars == 2
    assert src.degraded is False


@respx.mock
def test_hf_repo_link_is_bounded_to_the_github_widths() -> None:
    # GitHub allows a 39-character owner and a 100-character repo, and the bound is 200
    # characters, written here so the test pins the number rather than restating the constant.
    owner = "a" * 39
    repo = "b" * (200 - len("https://github.com/") - len(owner) - 1)
    at_limit = f"https://github.com/{owner}/{repo}"
    over_limit = at_limit + "c"
    assert len(at_limit) == 200
    rows = [
        {"paper": {"id": "2510.00001", "upvotes": 1, "githubRepo": at_limit}},
        {"paper": {"id": "2510.00002", "upvotes": 1, "githubRepo": over_limit}},
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].github_repo == at_limit
    assert out["2510.00002"].github_repo is None


@respx.mock
def test_hf_rejects_control_and_surrogate_characters() -> None:
    # Postgres text cannot hold NUL and a lone surrogate cannot be encoded at all; both are bad
    # extras, so the row survives with the value dropped instead of failing the listing write.
    rows = [
        {"paper": {"id": "2510.00001", "upvotes": 1, "organization": {"name": "a\x00b"}}},
        {"paper": {"id": "2510.00002", "upvotes": 1, "organization": {"name": "a\ud800b"}}},
        {"paper": {"id": "2510.00003", "upvotes": 1, "projectPage": "https://a.com/\ud800"}},
        {"paper": {"id": "2510.00004", "upvotes": 1, "projectPage": "https://a.com/\x00"}},
    ]
    # Raw content, escaped by json.dumps: the client's own JSON encoder refuses a lone surrogate.
    body = json.dumps(rows).encode()
    respx.get(HF_URL).mock(return_value=httpx.Response(200, content=body))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].organization is None
    assert out["2510.00002"].organization is None
    assert out["2510.00003"].project_page is None
    assert out["2510.00004"].project_page is None
    assert all(paper.upvotes == 1 for paper in out.values())
    assert src.degraded is False


@respx.mock
def test_hf_project_page_length_boundary_and_hostile_shapes() -> None:
    # Both ends of the length bound are pinned, and a page must have a real https host: no
    # credentials, no whitespace and no control characters.
    base = "https://example.com/"
    at_limit = base + "a" * (PROJECT_PAGE_MAX_CHARS - len(base))
    over_limit = at_limit + "a"
    pages = {
        "2510.00001": at_limit,
        "2510.00002": over_limit,
        "2510.00003": "https://",
        "2510.00004": "https://a b.example/c",
        "2510.00005": "https://a\nb.example/c",
        "2510.00006": "https://user@example.com/c",
        "2510.00007": "https://example.com/\x00",
        # A malformed host must not raise out of urlsplit and cost the whole day.
        "2510.00008": "https://[::1/p",
        "2510.00009": "https://[x]/p",
        "2510.00010": "https://a.com\uff03@b.com/p",
        # The scheme stays strictly the lowercase https the plan names.
        "2510.00011": "HTTPS://example.com",
        "2510.00012": "//]",
        "2510.00015": "httpS://example.com",
        # DEL is a control character too, and a password-only URL still carries credentials.
        "2510.00013": "https://example.com/\x7f",
        "2510.00014": "https://:pw@example.com/c",
    }
    rows = [
        {"paper": {"id": pid, "upvotes": 1, "projectPage": page}} for pid, page in pages.items()
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    src = HfDailySource(httpx.Client())
    out = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].project_page == at_limit
    for dropped in pages:
        if dropped != "2510.00001":
            assert out[dropped].project_page is None
    # Every row survives a bad page, and the malformed host is a bad extra, not a failed day.
    assert src.degraded is False


@respx.mock
def test_hf_organisation_prefers_the_paper_and_falls_back_to_the_entry() -> None:
    # paper fullname, paper name, then the entry's; the paper's own organisation is the one the
    # listing attaches, so a different entry organisation never wins.
    rows = [
        {
            "paper": {"id": "2510.00001", "upvotes": 1},
            "organization": {"name": "EntrySlug", "fullname": "Entry Org"},
        },
        {
            "paper": {"id": "2510.00002", "upvotes": 1, "organization": {"name": "PaperSlug"}},
            "organization": {"name": "EntrySlug"},
        },
        {
            "paper": {
                "id": "2510.00003",
                "upvotes": 1,
                "organization": {"name": "PaperSlug", "fullname": "Paper Full"},
            }
        },
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].organization == "Entry Org"
    assert out["2510.00002"].organization == "PaperSlug"
    assert out["2510.00003"].organization == "Paper Full"


@respx.mock
def test_hf_listing_time_is_utc_and_a_malformed_one_is_none() -> None:
    # A naive stamp is read as UTC (the listing only sends UTC) and a malformed one is a bad extra
    # that keeps the row.
    rows = [
        {
            "paper": {
                "id": "2510.00001",
                "upvotes": 1,
                "submittedOnDailyAt": "2026-10-09T00:00:00",
            }
        },
        {
            "paper": {
                "id": "2510.00002",
                "upvotes": 1,
                "submittedOnDailyAt": "not-a-date",
            }
        },
    ]
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].listed_at == datetime(2026, 10, 9, tzinfo=UTC)
    assert out["2510.00002"].listed_at is None
    assert out["2510.00002"].upvotes == 1


@respx.mock
def test_hf_an_absent_star_count_stays_none() -> None:
    # A missing count is "not observed", never zero: zero would read as a real star count.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(200, json=[{"paper": {"id": "2510.00001", "upvotes": 1}}])
    )
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].github_stars is None
    assert out["2510.00001"].comments is None


@respx.mock
def test_hf_a_bad_extra_field_keeps_the_row() -> None:
    # A malformed extra is dropped, not the paper: the upvotes still count.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=[{"paper": {"id": "2510.00001", "upvotes": 42, "githubStars": "many"}}]
        )
    )
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert out["2510.00001"].upvotes == 42
    assert out["2510.00001"].github_stars is None


@respx.mock
def test_hf_extras_merge_across_days() -> None:
    # The window walks newest first: counts take the maximum, the first non-null extras win and
    # the listing time is the earliest seen, so one record describes the whole window.
    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2026-10-09":
            return httpx.Response(
                200,
                json=[
                    {
                        "paper": {
                            "id": "2510.00001",
                            "upvotes": 5,
                            "githubStars": 12,
                            "githubRepo": "https://github.com/a/b",
                            "submittedOnDailyAt": "2026-10-09T00:00:00.000Z",
                        }
                    }
                ],
            )
        return httpx.Response(
            200,
            json=[
                {
                    "paper": {
                        "id": "2510.00001",
                        "upvotes": 3,
                        "githubStars": 10,
                        "submittedOnDailyAt": "2026-10-08T00:00:00.000Z",
                    }
                }
            ],
        )

    respx.get(HF_URL).mock(side_effect=responder)
    out = HfDailySource(httpx.Client()).papers(date(2026, 10, 8), date(2026, 10, 9))
    assert out == {
        "2510.00001": HfPaper(
            upvotes=5,
            github_stars=12,
            github_repo="https://github.com/a/b",
            listed_at=datetime(2026, 10, 8, tzinfo=UTC),
        )
    }


@respx.mock
def test_hf_versioned_id_maps_to_base_id() -> None:
    # A version suffix must collapse so the row joins the stored arXiv item.
    respx.get(HF_URL).mock(
        return_value=httpx.Response(
            200, json=[_hf_row("2510.00003v2", 5), _hf_row("2510.00003", 3)]
        )
    )
    up = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert _upvotes(up) == {"2510.00003": 5}


@respx.mock
def test_hf_pages_until_a_short_page() -> None:
    # A full first page must be followed by p=1; a short second page ends the day.
    def responder(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["p"])
        if page == 0:
            return httpx.Response(200, json=[_hf_row(f"2510.{i:05d}") for i in range(100)])
        return httpx.Response(200, json=[_hf_row("2510.00050", 7)])

    route = respx.get(HF_URL).mock(side_effect=responder)
    up = HfDailySource(httpx.Client()).papers(date(2026, 10, 9), date(2026, 10, 9))
    assert route.call_count == 2
    assert up["2510.00050"].upvotes == 7


@respx.mock
def test_hf_queries_each_date() -> None:
    # One request per calendar day, newest first (the freshest listings carry the upvotes a run
    # needs) and paged from 0 with the API's 100-row cap.
    route = respx.get(HF_URL).mock(return_value=httpx.Response(200, json=[]))
    HfDailySource(httpx.Client()).papers(date(2026, 10, 8), date(2026, 10, 9))
    assert [c.request.url.params["date"] for c in route.calls] == [
        "2026-10-09",
        "2026-10-08",
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
    up = src.papers(date(2026, 10, 8), date(2026, 10, 9))
    assert _upvotes(up) == {"2510.00001": 42}
    assert src.degraded is True
    assert src.failure is not None and "2026-10-09" in src.failure


@respx.mock
def test_hf_clean_call_leaves_degraded_false() -> None:
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=[_hf_row("2510.00001", 42)]))
    src = HfDailySource(httpx.Client())
    src.papers(date(2026, 10, 9), date(2026, 10, 9))
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
    up = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert _upvotes(up) == {"2510.00001": 42}
    assert src.degraded is False


@respx.mock
def test_hf_stops_when_a_full_page_adds_nothing() -> None:
    # A server that ignores p returns the same full page forever; the day is
    # degraded instead of looping.
    rows = [_hf_row(f"2510.{i:05d}") for i in range(100)]
    route = respx.get(HF_URL).mock(return_value=httpx.Response(200, json=rows))
    src = HfDailySource(httpx.Client())
    up = src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert route.call_count == 2
    assert len(up) == 100
    assert src.degraded is True


@respx.mock
def test_hf_page_cap_bounds_a_day() -> None:
    # Distinct full pages never end on their own; the named cap bounds the day.
    def responder(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["p"])
        return httpx.Response(200, json=[_hf_row(f"2510.{page:02d}{i:03d}") for i in range(100)])

    route = respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    src.papers(date(2026, 10, 9), date(2026, 10, 9))
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
    # More rows than ids, and none naming a requested id: nothing pairs and no id is checked.
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
def test_s2_null_in_a_shifted_answer_stays_due() -> None:
    # Three rows for four ids: the answer has shifted, so a null cannot be attributed to an id by
    # position. Only the ids whose rows came back are checked, and a one-row gap is not degraded.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200,
            json=[_s2_row("2510.00001"), None, _s2_row("2510.00004")],
        )
    )
    ids = ["2510.00001", "2510.00002", "2510.00003", "2510.00004"]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(ids)
    assert set(out) == {"2510.00001", "2510.00004"}
    assert src.checked_ids == {"2510.00001", "2510.00004"}
    assert src.degraded is False
    assert src.failure is None


@respx.mock
def test_s2_row_for_an_id_not_asked_about_is_unpaired() -> None:
    # A row for a paper we did not ask for is a broken answer, not something to attribute to a
    # requested id; the id we did ask about keeps its own row.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(
            200, json=[_s2_row("2510.00001"), _s2_row("2510.00099", citations=4)]
        )
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001"])
    assert set(out) == {"2510.00001"}
    assert src.checked_ids == {"2510.00001"}
    assert src.degraded is True
    # The message names what came back, so a recurring anomaly can be recognised from the log.
    assert "2510.00099" in str(src.failure)


@respx.mock
def test_s2_row_naming_a_version_still_pairs() -> None:
    # The request strips vN, but S2 may echo the versioned id back; it is the same paper.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(200, json=[_s2_row("2510.00001v2", citations=3)])
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(["2510.00001"])
    assert out["2510.00001"].citation_count == 3
    assert src.checked_ids == {"2510.00001"}
    assert src.degraded is False


@respx.mock
def test_s2_degrades_when_most_of_a_chunk_is_left_out() -> None:
    # Two rows for ten ids is not S2 not knowing one paper: it is an answer this client cannot
    # understand, so it is reported. The two rows that did pair are kept.
    ids = [f"2510.{i:05d}" for i in range(10)]
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(200, json=[_s2_row(ids[0], citations=1), _s2_row(ids[1])])
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    out = src.lookup(ids)
    assert set(out) == set(ids[:2])
    assert src.checked_ids == set(ids[:2])
    assert src.degraded is True
    assert "2 rows for 10 ids" in str(src.failure)


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
    # A bug is not a refusal: nothing should ask for this id again as if the pool were busy.
    assert src.refused_ids == set()


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
    assert src.refused_ids == set()


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


@respx.mock
def test_s2_refused_ids_are_the_chunk_that_ran_out_of_attempts(monkeypatch) -> None:
    # S2 said "come back later" on every attempt: those ids are still due, and named as refused.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(429))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001", "2510.00002"]) == {}
    assert route.call_count == 5
    assert src.refused_ids == {"2510.00001", "2510.00002"}
    assert src.checked_ids == set()


@respx.mock
def test_s2_transport_error_is_a_refusal(monkeypatch) -> None:
    # A transport blip is the other retryable failure: the chunk is refused, not buggy.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(side_effect=httpx.ConnectError("boom"))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert src.refused_ids == {"2510.00001"}
    assert src.degraded is True


@respx.mock
def test_s2_omitted_id_is_unchecked_but_not_refused() -> None:
    # An id S2 leaves out is a paper it does not know, not a refusal: a retry pass must not spend
    # its budget there as if the pool were congested.
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(200, json=[_s2_row("2510.00001"), None])
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001", "2510.00002", "2510.00003"])
    assert src.checked_ids == {"2510.00001"}
    assert src.refused_ids == set()


@respx.mock
def test_s2_refused_ids_reset_each_call(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    responses = [httpx.Response(503), *[httpx.Response(429)] * 5, httpx.Response(200, json=[None])]
    respx.post(S2_BATCH_URL).mock(side_effect=responses)
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    assert src.refused_ids == {"2510.00001"}
    src.lookup(["2510.00002"])
    assert src.refused_ids == set()


@respx.mock
def test_s2_stops_asking_after_two_refused_chunks(monkeypatch) -> None:
    # A keyless pool that refuses two whole chunks in a row is refusing the run, not the chunk:
    # the third is never asked, and its ids count as refused so they stay due for the next pass.
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    route = respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(429))
    ids = [f"2510.{i:05d}" for i in range(500 * 3)]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=1.5)
    assert src.lookup(ids) == {}
    assert route.call_count == 5 * S2_REFUSED_CHUNKS  # the third chunk is never sent
    assert src.refused_ids == set(ids)
    # One inter-chunk pause, and none after the second refusal: the run has stopped asking.
    assert len([s for s in sleeps if 1.2 <= s <= 1.8]) == 1


@respx.mock
def test_s2_refusal_streak_resets_on_an_answered_chunk(monkeypatch) -> None:
    # A chunk S2 answers ends the streak, so the refusals on either side of it do not add up; only
    # two in a row do. Without the reset the breaker would fire on the third chunk.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    asked = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal asked
        asked += 1
        count = len(json.loads(request.content)["ids"])
        if asked == 1:  # chunk 1 refused
            return httpx.Response(503)
        if asked == 6:  # chunk 2 answered
            return httpx.Response(200, json=[None] * count)
        return httpx.Response(503)  # chunks 3 and 4 refused, then the breaker stops the run

    route = respx.post(S2_BATCH_URL).mock(side_effect=responder)
    ids = [f"2510.{i:05d}" for i in range(500 * 5)]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(ids)
    assert route.call_count == 5 + 1 + 5 + 5
    assert src.checked_ids == set(ids[500:1000])
    assert src.refused_ids == set(ids[:500]) | set(ids[1000:])


@respx.mock
def test_s2_logs_every_refusal_with_its_wait(monkeypatch) -> None:
    # The morning check reads these lines to tell a congested hour from a blocked key.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=[_s2_row("2510.00001")]),
        ]
    )
    with capture_logs() as logs:
        SemanticScholarSource(httpx.Client(), None, pause_s=0).lookup(["2510.00001"])
    [entry] = [entry for entry in logs if entry["event"] == "s2_attempt_refused"]
    assert entry["status"] == 429 and entry["retry_after"] == "7"
    assert entry["wait_s"] == 7.0  # the server's Retry-After, not the backoff
    assert entry["at"].endswith("+00:00")  # its own UTC stamp, not only the log line's
    assert entry["error"] == "HTTPStatusError"


@respx.mock
def test_s2_logs_a_refused_chunk_and_the_stop(monkeypatch) -> None:
    # One line per refused chunk and one for the stop, so a congested run reads at a glance.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(503))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    ids = [f"2510.{i:05d}" for i in range(500 * 3)]
    with capture_logs() as logs:
        src.lookup(ids)
    refused = [entry for entry in logs if entry["event"] == "s2_chunk_refused"]
    assert [entry["refused_in_a_row"] for entry in refused] == [1, 2]
    # The status and Retry-After are structured fields, not only inside the error string.
    assert {entry["status"] for entry in refused} == {503}
    assert {entry["retry_after"] for entry in refused} == {None}
    [stop] = [entry for entry in logs if entry["event"] == "s2_stopped_asking"]
    assert stop["unasked"] == 500
    assert stop["refused_in_a_row"] == S2_REFUSED_CHUNKS


@respx.mock
@pytest.mark.parametrize(
    "bug",
    [
        pytest.param(lambda: httpx.Response(400), id="deterministic-4xx"),
        pytest.param(
            lambda: httpx.Response(200, json={"error": "Unrecognized field"}), id="non-list-body"
        ),
        pytest.param(
            lambda: httpx.Response(400, json={"error": "No valid paper ids given"}),
            id="unknown-400",
        ),
    ],
)
def test_s2_a_failure_that_is_not_a_refusal_ends_the_streak(monkeypatch, bug) -> None:
    # Only a refusal carries the streak: a bug between two refusals must not stop the run, so the
    # fourth chunk is still asked and only the two refused chunks are named.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    asked = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal asked
        asked += 1
        count = len(json.loads(request.content)["ids"])
        if asked == 6:  # the second chunk fails once, with the bug under test
            return bug()
        if asked == 12:  # the fourth chunk is answered
            return httpx.Response(200, json=[None] * count)
        return httpx.Response(429)  # chunks 1 and 3 are refused, 5 attempts each

    route = respx.post(S2_BATCH_URL).mock(side_effect=responder)
    ids = [f"2510.{i:05d}" for i in range(500 * 4)]
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(ids)
    assert route.call_count == 12
    assert src.refused_ids == set(ids[:500]) | set(ids[1000:1500])
    assert set(ids[1500:]) <= src.checked_ids


@respx.mock
def test_s2_anomaly_names_a_problem_that_is_not_a_refusal() -> None:
    # A shape this client cannot read is not a throttle: it is reported apart, so a run's later
    # clean retry of the refused ids cannot hide it (review S4).
    ids = [f"2510.{i:05d}" for i in range(10)]
    respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(200, json=[_s2_row(ids[0]), _s2_row(ids[1])])
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(ids)
    assert src.degraded is True and src.refused_ids == set()
    assert src.anomaly is not None and "2 rows for 10 ids" in src.anomaly


@respx.mock
def test_s2_anomaly_is_set_for_a_deterministic_4xx(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(400, json={"error": "nope"}))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    assert src.degraded is True and src.refused_ids == set()
    assert src.anomaly is not None and "400" in src.anomaly


@respx.mock
def test_s2_anomaly_is_not_set_by_a_refusal(monkeypatch) -> None:
    # The pool refusing us is a refusal, not an anomaly: a later pass may clear it.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(429))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(["2510.00001"])
    assert src.refused_ids == {"2510.00001"}
    assert src.anomaly is None


@respx.mock
def test_s2_anomaly_describes_one_call_only() -> None:
    ids = [f"2510.{i:05d}" for i in range(10)]
    respx.post(S2_BATCH_URL).mock(
        side_effect=[
            httpx.Response(200, json=[_s2_row(ids[0])]),
            httpx.Response(200, json=[None] * 10),
        ]
    )
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    src.lookup(ids)
    assert src.anomaly is not None
    src.lookup(ids)
    assert src.degraded is False and src.anomaly is None


@respx.mock
def test_s2_5xx_after_every_attempt_is_refused_not_an_anomaly(monkeypatch) -> None:
    # A gateway error is the pool being unavailable, not a request bug: the ids stay due for a
    # follow-up pass, and the chunk is never reported as an answer this client cannot read.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(502))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001", "2510.00002"]) == {}
    assert route.call_count == 5
    assert src.refused_ids == {"2510.00001", "2510.00002"}
    assert src.checked_ids == set()
    assert src.anomaly is None
    assert src.degraded is True


@respx.mock
def test_s2_does_not_follow_a_redirect(monkeypatch) -> None:
    # The batch endpoint is named in code; a 3xx means it moved, which a follow-up pass cannot fix,
    # so it is an anomaly and the ids stay unasked rather than refused.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.post(S2_BATCH_URL).mock(
        return_value=httpx.Response(302, headers={"location": "https://elsewhere.test/batch"})
    )
    respx.post("https://elsewhere.test/batch").mock(return_value=httpx.Response(200, json=[None]))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert route.call_count == 1
    assert src.refused_ids == set()
    assert src.anomaly is not None


@respx.mock
def test_hf_stops_asking_after_two_failed_days_in_a_row(monkeypatch) -> None:
    # A gateway that answers 504 after a minute turns 31 days x 3 attempts into hours of waiting,
    # and the days it would have asked are re-read by every run anyway.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    route = respx.get(HF_URL).mock(return_value=httpx.Response(504))
    src = HfDailySource(httpx.Client())
    assert src.papers(date(2026, 10, 8), date(2026, 10, 31)) == {}
    assert route.call_count == 2 * 3  # two days, three attempts each, then it stopped
    assert src.degraded is True
    # The window is walked newest first, so the two days it gave up on are the newest ones; the
    # failure names the second of them, so the morning check says which day broke the run.
    assert src.failure is not None and "2026-10-30" in src.failure
    assert "2 days in a row" in src.failure


@respx.mock
def test_hf_failed_day_streak_ignores_a_deterministic_failure(monkeypatch) -> None:
    # A day that fails for its own reason (here a 400: a bug, never a retry) is not evidence that
    # the endpoint is down, so it resets the streak instead of counting towards it.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2026-10-30":
            return httpx.Response(400)
        return httpx.Response(502)

    route = respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    src.papers(date(2026, 10, 8), date(2026, 10, 31))
    # Newest first: 10-31 failed, 10-30 reset the streak, 10-29 and 10-28 tripped the breaker.
    asked = [call.request.url.params["date"] for call in route.calls]
    assert sorted(set(asked)) == ["2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31"]
    assert src.failure is not None and "2026-10-28" in src.failure


@respx.mock
def test_hf_answers_after_a_failed_day_keep_the_source_healthy(monkeypatch) -> None:
    # One failed day between answers is a slow day, not an outage: the breaker must not trip.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "2026-10-09":
            return httpx.Response(503)
        return httpx.Response(200, json=[_hf_row("2510.00001", 7)])

    respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    up = src.papers(date(2026, 10, 8), date(2026, 10, 12))
    assert _upvotes(up) == {"2510.00001": 7}
    assert src.degraded is True  # the failed day is still reported
    assert src.failure is not None and "2026-10-09" in src.failure
    assert "stopped" not in src.failure


@respx.mock
def test_hf_cap_reaches_the_source(monkeypatch) -> None:
    # The per-source cap is wired to the client, so an over-long page fails the day instead of
    # holding the body in memory.
    monkeypatch.setattr("paperwatch.sources.hf_daily.MAX_HF_BODY_BYTES", 16)
    respx.get(HF_URL).mock(return_value=httpx.Response(200, json=[_hf_row("2510.00001", 1)]))
    src = HfDailySource(httpx.Client())
    src.papers(date(2026, 10, 9), date(2026, 10, 9))
    assert src.degraded is True
    assert src.failure is not None and "more than 16 bytes" in src.failure


@respx.mock
def test_s2_cap_reaches_the_source(monkeypatch) -> None:
    # An over-long batch answer is an answer this client cannot read, so the chunk is an anomaly
    # (not a refusal) and the ids stay unasked.
    monkeypatch.setattr("paperwatch.sources.semantic_scholar.MAX_S2_BODY_BYTES", 16)
    respx.post(S2_BATCH_URL).mock(return_value=httpx.Response(200, json=[_s2_row("2510.00001")]))
    src = SemanticScholarSource(httpx.Client(), None, pause_s=0)
    assert src.lookup(["2510.00001"]) == {}
    assert src.degraded is True
    assert src.anomaly is not None and "more than 16 bytes" in src.anomaly
    assert src.refused_ids == set()


@respx.mock
def test_hf_bad_old_days_do_not_stop_the_newest_ones(monkeypatch) -> None:
    # The listings a run needs are the newest ones, so the window is walked newest first: two bad
    # days at the far end of it must not cost the fresh upvotes, which is what oldest-first did.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] in ("2026-09-23", "2026-09-24"):
            return httpx.Response(500)
        return httpx.Response(200, json=[_hf_row("2510.00001", 9)])

    respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    up = src.papers(date(2026, 9, 23), date(2026, 10, 23))
    assert _upvotes(up) == {"2510.00001": 9}
    assert src.degraded is True  # the two bad days are still reported
    assert src.failure is not None and "2026-09-23" in src.failure


@respx.mock
def test_hf_breaker_counts_consecutive_days_only(monkeypatch) -> None:
    # fail, ok, fail is not an outage: the streak resets on every answered day, so the breaker
    # never trips and the whole window is asked.
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] in ("2026-10-31", "2026-10-29"):
            return httpx.Response(503)
        return httpx.Response(200, json=[_hf_row("2510.00001", 5)])

    route = respx.get(HF_URL).mock(side_effect=responder)
    src = HfDailySource(httpx.Client())
    up = src.papers(date(2026, 10, 1), date(2026, 10, 31))
    assert _upvotes(up) == {"2510.00001": 5}
    asked = {call.request.url.params["date"] for call in route.calls}
    assert len(asked) == 31  # every day, none skipped
    assert src.failure is not None and "2026-10-29" in src.failure
    assert "stopped" not in src.failure
