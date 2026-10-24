"""The corpus routes: the browsable table and one item with its history, signals and drafts.

Every row is seeded through the repository, so these tests run offline and a failure here means
the route, the cursor, the filters or the store changed rather than the ingestion.
"""

import base64
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from paperwatch.api.app import AgentDeps, create_app
from paperwatch.api.auth import HmacVerifier, mint_hmac_token
from paperwatch.api.limits import RequestLimiter
from paperwatch.api.search_mode import ModeDecision
from paperwatch.config import (
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    Settings,
)
from paperwatch.domain import AdmissionBasis, Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.store.models import Item, ItemSummary, SignalSnapshot
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 4, 9, tzinfo=UTC)
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret


class ExplodingEmbedder:
    """An embedder that fails the test if a corpus read ever asks it for a vector."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the corpus routes never embed")


class FixedMode:
    def current(self) -> ModeDecision:
        return ModeDecision("hybrid", "ok")


class NoChatCalls:
    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        raise AssertionError("the corpus routes never spend a model call")


class AuthedClient(TestClient):
    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", None) or {})
        if "authorization" not in {name.lower() for name in headers}:
            headers["Authorization"] = f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}"
        return super().request(method, url, headers=headers, **kwargs)


def client(db_session: Session, *, profiles: frozenset[str] = frozenset()) -> AuthedClient:
    deps = AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=ExplodingEmbedder(),
        models=[],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=NoChatCalls(),  # type: ignore[arg-type]
        profile_names=profiles,
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=FixedMode(),  # type: ignore[arg-type]
    )
    return AuthedClient(create_app(deps), base_url="http://localhost")


def seed(
    db_session: Session,
    external_id: str,
    *,
    title: str,
    body: str,
    published_at: datetime = NOW,
    tier: Tier = Tier.VETTED_PREPRINT,
    source: str = "arxiv",
    profile: str = "agentic-engineering",
    basis: AdmissionBasis = AdmissionBasis.SEED,
    hf_upvotes: int | None = None,
    citations: int = 0,
    venue: str | None = None,
    commit: bool = True,
) -> UUID:
    repo = Repository(db_session)
    item = SourceItem(
        source,
        external_id,
        title,
        ("Ada Lovelace",),
        body,
        f"https://example.org/{external_id}",
        published_at,
    )
    item_id = repo.upsert_item(
        item,
        tier,
        profile,
        0.9,
        Signals(venue=venue, citation_count=citations, hf_upvotes=hf_upvotes),
        FakeEmbedder().embed([f"{title}\n{body}"])[0],
        basis=basis,
    )
    # Committed, because the routes open and close their own sessions: a flush alone would be
    # rolled back by the first of them.
    if commit:
        db_session.commit()
    return item_id


def page_of(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def test_items_page_newest_first_with_a_stable_cursor(db_session: Session) -> None:
    api = client(db_session)
    oldest = seed(
        db_session,
        "oldest",
        title="Oldest",
        body="oldest body",
        published_at=NOW - timedelta(days=5),
    )
    # Two items share a published date, which is the normal case: a day's arXiv window lands on
    # one timestamp, so the cursor has to break the tie on a key that never moves.
    tie_a = seed(db_session, "tie-a", title="Tie A", body="tie body a", published_at=NOW)
    tie_b = seed(db_session, "tie-b", title="Tie B", body="tie body b", published_at=NOW)

    first = page_of(api.get("/items?limit=1"))
    assert first["total"] == 3
    assert len(first["items"]) == 1
    assert first["next_cursor"]

    seen = [first["items"][0]["id"]]
    cursor = first["next_cursor"]
    while cursor:
        body = page_of(api.get(f"/items?limit=1&cursor={cursor}"))
        assert body["total"] == 3
        seen.extend(item["id"] for item in body["items"])
        cursor = body["next_cursor"]
    # Every item exactly once: a tie never repeats a row nor hides one.
    assert len(seen) == 3
    assert set(seen) == {str(oldest), str(tie_a), str(tie_b)}
    # Newest first, and the shared date ordered by id descending, so the walk is reproducible.
    assert seen[2] == str(oldest)
    assert seen[:2] == sorted([str(tie_a), str(tie_b)], reverse=True)
    assert first["items"][0]["title"] == ("Tie B" if seen[0] == str(tie_b) else "Tie A")

    # The row carries the facts the table prints, and the API's own label for the basis.
    row = first["items"][0]
    assert set(row) == {
        "id",
        "title",
        "source",
        "tier",
        "basis",
        "basis_label",
        "profile",
        "venue",
        "citation_count",
        "hf_upvotes",
        "github_stars",
        "published_at",
    }
    assert row["tier"] == "vetted_preprint"
    assert (
        row["basis_label"]
        == "preprint from a recognised lab's authors, not yet independently vetted"
    )


def test_filters_combine(db_session: Session) -> None:
    api = client(db_session, profiles=frozenset({"agentic-engineering", "ai-forensics"}))
    seed(
        db_session,
        "want",
        title="Watermarks in generated images",
        body="A forensics paper about re-encoded watermarks.",
        profile="ai-forensics",
        tier=Tier.PEER_REVIEWED,
        basis=AdmissionBasis.VENUE,
        venue="ICLR 2027",
        citations=31,
        published_at=NOW,
    )
    seed(
        db_session,
        "wrong-source",
        title="Watermarks in generated images, the blog cut",
        body="Our team writes about watermarks.",
        profile="ai-forensics",
        tier=Tier.LAB_BLOG,
        source="feed",
        published_at=NOW,
    )
    seed(
        db_session,
        "wrong-profile",
        title="Watermarks for agents",
        body="Agent harness work.",
        profile="agentic-engineering",
        tier=Tier.PEER_REVIEWED,
        basis=AdmissionBasis.VENUE,
        venue="ICLR 2027",
        published_at=NOW,
    )
    seed(
        db_session,
        "too-old",
        title="Watermarks, an older paper",
        body="Forensics, last year.",
        profile="ai-forensics",
        tier=Tier.PEER_REVIEWED,
        basis=AdmissionBasis.VENUE,
        venue="ICLR 2026",
        published_at=NOW - timedelta(days=90),
    )

    combined = page_of(
        api.get(
            "/items",
            params={
                "tier": "peer_reviewed",
                "source": "arxiv",
                "profile": "ai-forensics",
                "since": (NOW - timedelta(days=30)).date().isoformat(),
            },
        )
    )
    assert [item["title"] for item in combined["items"]] == ["Watermarks in generated images"]
    assert combined["total"] == 1
    assert combined["next_cursor"] is None

    # Each filter narrows on its own, too.
    assert page_of(api.get("/items?source=feed"))["total"] == 1
    assert page_of(api.get("/items?tier=lab_blog"))["total"] == 1
    assert page_of(api.get("/items?profile=agentic-engineering"))["total"] == 1
    assert page_of(api.get("/items"))["total"] == 4


def test_search_uses_the_text_branch_and_needs_no_model(db_session: Session) -> None:
    # The client's embedder raises, so a search that embedded anything would fail this test.
    api = client(db_session)
    seed(
        db_session,
        "replay",
        title="Replay buffers for coding agents",
        body="A replay buffer replaces the running summary and cuts tokens per task.",
    )
    seed(
        db_session,
        "watermark",
        title="Detecting re-encoded watermarks",
        body="A forensics approach to image watermarks.",
    )

    found = page_of(api.get("/items?q=replay+buffer"))
    assert [item["title"] for item in found["items"]] == ["Replay buffers for coding agents"]
    assert found["total"] == 1
    assert found["next_cursor"] is None

    # Any shared stem matches, and the search is the same in every mode because it never embeds.
    assert page_of(api.get("/items?q=watermarks"))["total"] == 1
    assert page_of(api.get("/items?q=rotary+embeddings"))["items"] == []

    # A search pages by offset inside its cursor, ranked, with the match count as its total.
    first = page_of(api.get("/items?q=replay+buffer&limit=1"))
    assert len(first["items"]) == 1
    assert first["next_cursor"] is None


def test_an_unknown_profile_is_422(db_session: Session) -> None:
    api = client(db_session, profiles=frozenset({"agentic-engineering"}))
    seed(db_session, "one", title="One", body="One body.")
    refused = api.get("/items?profile=neuromorphic")
    assert refused.status_code == 422
    assert "neuromorphic" in refused.text


def test_a_cursor_that_is_not_ours_is_422(db_session: Session) -> None:
    api = client(db_session)
    assert api.get("/items?cursor=not-a-cursor").status_code == 422
    # A base64 blob of the wrong shape is not a cursor either.
    wrong = base64.urlsafe_b64encode(json.dumps({"k": "newest"}).encode()).decode()
    assert api.get(f"/items?cursor={wrong}").status_code == 422


def test_item_detail_has_history_signals_summary_and_drafts(db_session: Session) -> None:
    api = client(db_session)
    item_id = seed(
        db_session,
        "detail",
        title="Replay Buffers Beat Summaries for Long-Running Coding Agents",
        body="A replay buffer replaces the running summary and cuts tokens per task by 41%.",
        hf_upvotes=12,
    )
    repo = Repository(db_session)
    # The tier moves once, so the page has a timeline to draw rather than a single point.
    repo.admit_item(
        item_id,
        Tier.PEER_REVIEWED,
        AdmissionBasis.VENUE,
        None,
        Signals(venue="ICLR 2027", citation_count=31),
        NOW,
    )
    row = db_session.get_one(Item, item_id)
    row.github_repo = "https://github.com/example/replay-buffers"
    row.project_page = "https://example.org/project"
    row.hf_org = "example-lab"
    # The venue arrives with the signals the pipeline merges, not with the admission call.
    row.venue = "ICLR 2027"
    row.citation_count = 31
    db_session.add_all(
        [
            SignalSnapshot(
                item_id=item_id,
                observed_at=NOW - timedelta(days=2),
                hf_upvotes=8,
                hf_comments=1,
                github_stars=410,
                citation_count=0,
            ),
            SignalSnapshot(
                item_id=item_id,
                observed_at=NOW - timedelta(days=1),
                hf_upvotes=12,
                hf_comments=3,
                github_stars=512,
                citation_count=31,
            ),
        ]
    )
    db_session.add(
        ItemSummary(
            item_id=item_id,
            text_hash="hash",
            model="test-model",
            prompt_version="s1",
            status="checked",
            fields={
                "what": {
                    "text": "A replay buffer for long-running coding agents.",
                    "quote": "A replay buffer replaces the running summary",
                },
                "result": {
                    "text": "Tokens per task fell by 41%.",
                    "quote": "cuts tokens per task by 41%",
                },
                "why": {"text": "Cheaper long runs.", "interpretation": True},
                "catch": {"text": "One benchmark.", "interpretation": True},
            },
            failed_checks=[],
            created_at=NOW,
        )
    )
    repo.add_draft(
        item_id=item_id,
        inputs={"title": "x"},
        pov="",
        language="en",
        model="test-model",
        prompt_version="d1",
        hook="A hook",
        body="A body",
        hashtags=("agents",),
        source_link="https://example.org/detail",
        check={"flags": [], "counts": {}},
        regenerated_from=None,
        at=NOW,
    )
    db_session.commit()

    detail = page_of(api.get(f"/items/{item_id}"))
    assert detail["id"] == str(item_id)
    assert detail["title"].startswith("Replay Buffers")
    assert detail["authors"] == ["Ada Lovelace"]
    assert detail["url"] == "https://example.org/detail"
    assert detail["text"].startswith("A replay buffer")
    assert detail["tier"] == "peer_reviewed"
    assert detail["basis"] == "venue"
    assert detail["basis_label"] == "peer-reviewed via venue: ICLR 2027"
    assert detail["github_repo"] == "https://github.com/example/replay-buffers"
    assert detail["project_page"] == "https://example.org/project"
    assert detail["org"] == "example-lab"
    # Oldest first and complete: the first admission is a move too, so the timeline starts with it.
    assert detail["tier_history"] == [
        {
            "from_tier": None,
            "to_tier": "vetted_preprint",
            "basis": "seed",
            "changed_at": NOW.isoformat().replace("+00:00", "Z"),
        },
        {
            "from_tier": "vetted_preprint",
            "to_tier": "peer_reviewed",
            "basis": "venue",
            "changed_at": NOW.isoformat().replace("+00:00", "Z"),
        },
    ]
    # Oldest first, so the table reads as a series and the page prints no sorting logic.
    assert [signal["observed_at"] for signal in detail["signals"]] == [
        (NOW - timedelta(days=2)).isoformat().replace("+00:00", "Z"),
        (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    ]
    assert detail["signals"][1]["github_stars"] == 512
    assert detail["summary"]["status"] == "checked"
    assert detail["summary"]["result"]["quote"] == "cuts tokens per task by 41%"
    assert len(detail["drafts"]) == 1
    assert set(detail["drafts"][0]) == {"id", "created_at"}


def test_an_item_without_a_summary_or_drafts_says_null(db_session: Session) -> None:
    api = client(db_session)
    item_id = seed(db_session, "bare", title="Bare item", body="Nothing else stored.")
    detail = page_of(api.get(f"/items/{item_id}"))
    assert detail["summary"] is None
    assert detail["drafts"] == []
    assert detail["signals"] == []
    # One row: the admission itself, which is where the item's tier started.
    assert [change["from_tier"] for change in detail["tier_history"]] == [None]


def test_a_candidate_item_is_404(db_session: Session) -> None:
    api = client(db_session)
    repo = Repository(db_session)
    item = SourceItem(
        "arxiv",
        "candidate",
        "Not admitted yet",
        ("Ada Lovelace",),
        "A candidate body.",
        "https://example.org/candidate",
        NOW,
    )
    item_id = repo.record_classified(item, "candidate", "agentic-engineering", 0.2)
    db_session.commit()
    assert api.get(f"/items/{item_id}").status_code == 404
    # A candidate is not in the browsable table either.
    assert page_of(api.get("/items"))["total"] == 0


def b64(values: dict[str, object]) -> str:
    """A cursor-shaped blob, for the ones the API did not issue."""
    raw = json.dumps(values, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def test_a_search_pages_by_offset_and_holds_the_cap_for_any_limit(db_session: Session) -> None:
    api = client(db_session)
    for i in range(260):
        seed(
            db_session,
            f"cap-{i:03d}",
            title=f"Replay buffer study {i:03d}",
            body="A replay buffer for coding agents.",
            published_at=NOW - timedelta(minutes=i),
            commit=False,
        )
    db_session.commit()

    # The cap is 200 whatever page size the caller picks: a limit that does not divide it must not
    # let the table walk past it, and the total says what the walk will really return.
    for limit in (30, 50):
        seen: list[str] = []
        query = f"/items?q=replay+buffer&limit={limit}"
        while True:
            page = page_of(api.get(query))
            assert page["total"] == 200
            seen.extend(item["id"] for item in page["items"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
            query = f"/items?q=replay+buffer&limit={limit}&cursor={cursor}"
        assert len(seen) == 200
        assert len(set(seen)) == 200


def test_a_cursor_the_api_did_not_issue_is_422(db_session: Session) -> None:
    api = client(db_session)
    seed(db_session, "one", title="Replay buffer", body="A replay buffer.")
    for forged in (
        {"k": "search", "o": 201},
        {"k": "search", "o": 2**63},
        {"k": "search", "o": 0},
        {"k": "search", "o": "many"},
        {"k": "newest", "p": "not-a-date", "i": "not-a-uuid"},
    ):
        refused = api.get("/items", params={"q": "replay", "cursor": b64(forged)})
        # A cursor is the API's own token: anything else is a client bug, not a 500.
        assert refused.status_code == 422, forged


def test_a_control_character_in_the_query_is_422(db_session: Session) -> None:
    api = client(db_session)
    seed(db_session, "one", title="Replay buffer", body="A replay buffer.")
    # psycopg refuses a NUL in a text parameter, so the route must refuse it before the query.
    assert api.get("/items", params={"q": "\x00x"}).status_code == 422


def test_a_last_page_that_exactly_fills_the_limit_has_no_cursor(db_session: Session) -> None:
    api = client(db_session)
    for i in range(20):
        seed(
            db_session,
            f"exact-{i:02d}",
            title=f"Exact page {i:02d}",
            body="An abstract.",
            published_at=NOW - timedelta(hours=i),
            commit=False,
        )
    db_session.commit()

    # A cursor that leads to an empty page is a "Load more" that loads nothing.
    page = page_of(api.get("/items?limit=20"))
    assert len(page["items"]) == 20
    assert page["next_cursor"] is None

    # The same at a two-page boundary, where the second page is exactly full.
    second = page_of(api.get("/items?limit=10"))
    assert second["next_cursor"] is not None
    last = page_of(api.get(f"/items?limit=10&cursor={second['next_cursor']}"))
    assert len(last["items"]) == 10
    assert last["next_cursor"] is None


def test_a_cursor_from_the_other_branch_is_422(db_session: Session) -> None:
    api = client(db_session)
    for i in range(3):
        seed(
            db_session,
            f"both-{i}",
            title=f"Replay buffer {i}",
            body="A replay buffer.",
            published_at=NOW - timedelta(hours=i),
            commit=False,
        )
    db_session.commit()

    newest = page_of(api.get("/items?limit=1"))["next_cursor"]
    search = page_of(api.get("/items?q=replay&limit=1"))["next_cursor"]
    assert newest and search
    # A keyset position is not an offset and an offset is not a keyset position.
    assert api.get(f"/items?q=replay&cursor={newest}").status_code == 422
    assert api.get(f"/items?cursor={search}").status_code == 422


def test_a_blank_query_is_no_search(db_session: Session) -> None:
    api = client(db_session)
    seed(
        db_session,
        "older",
        title="Older item",
        body="An abstract.",
        published_at=NOW - timedelta(days=1),
    )
    seed(db_session, "newer", title="Newer item", body="An abstract.", published_at=NOW)

    plain = page_of(api.get("/items"))
    for blank in ("", "   "):
        assert page_of(api.get("/items", params={"q": blank})) == plain
    assert [item["title"] for item in plain["items"]] == ["Newer item", "Older item"]


def test_the_default_limit_is_25_and_51_is_refused(db_session: Session) -> None:
    api = client(db_session)
    for i in range(30):
        seed(
            db_session,
            f"limit-{i:02d}",
            title=f"Limit {i:02d}",
            body="An abstract.",
            published_at=NOW - timedelta(minutes=i),
            commit=False,
        )
    db_session.commit()

    default = page_of(api.get("/items"))
    assert len(default["items"]) == 25
    assert default["next_cursor"] is not None
    assert api.get("/items?limit=51").status_code == 422
    widest = page_of(api.get("/items?limit=50"))
    assert len(widest["items"]) == 30
    assert widest["next_cursor"] is None


def test_the_cursor_is_taken_from_the_last_row_of_the_page(db_session: Session) -> None:
    api = client(db_session)
    ids = [
        seed(
            db_session,
            f"row-{i}",
            title=f"Row {i}",
            body="An abstract.",
            published_at=NOW - timedelta(hours=i),
            commit=False,
        )
        for i in range(4)
    ]
    db_session.commit()

    first = page_of(api.get("/items?limit=2"))
    assert [item["id"] for item in first["items"]] == [str(ids[0]), str(ids[1])]
    second = page_of(api.get(f"/items?limit=2&cursor={first['next_cursor']}"))
    assert [item["id"] for item in second["items"]] == [str(ids[2]), str(ids[3])]
    assert second["next_cursor"] is None


def test_since_includes_the_utc_midnight(db_session: Session) -> None:
    api = client(db_session)
    seed(
        db_session,
        "midnight",
        title="Just after midnight",
        body="An abstract.",
        published_at=datetime(2026, 11, 1, tzinfo=UTC),
        commit=False,
    )
    seed(
        db_session,
        "before",
        title="A second before",
        body="An abstract.",
        published_at=datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC),
        commit=False,
    )
    db_session.commit()

    found = page_of(api.get("/items?since=2026-11-01"))
    assert [item["title"] for item in found["items"]] == ["Just after midnight"]
    assert found["total"] == 1


def test_the_item_page_keeps_the_newest_thirty_readings(db_session: Session) -> None:
    api = client(db_session)
    item_id = seed(db_session, "series", title="A series", body="An abstract.")
    for i in range(35):
        db_session.add(
            SignalSnapshot(
                item_id=item_id,
                observed_at=NOW + timedelta(days=i),
                hf_upvotes=i,
                hf_comments=None,
                github_stars=100 + i,
                citation_count=0,
            )
        )
    db_session.commit()

    detail = page_of(api.get(f"/items/{item_id}"))
    assert len(detail["signals"]) == 30
    # The newest thirty, still oldest first, so the table reads as a series.
    assert [point["hf_upvotes"] for point in detail["signals"]] == list(range(5, 35))


def test_tier_history_is_oldest_first_and_drafts_belong_to_their_item(db_session: Session) -> None:
    api = client(db_session)
    item_id = seed(db_session, "moves", title="Moving item", body="An abstract.")
    other_id = seed(db_session, "other", title="Other item", body="Another abstract.")
    repo = Repository(db_session)
    repo.admit_item(
        item_id,
        Tier.PEER_REVIEWED,
        AdmissionBasis.VENUE,
        None,
        Signals(venue="ICLR 2027"),
        NOW + timedelta(days=2),
    )
    repo.admit_item(
        item_id,
        Tier.LAB_BLOG,
        AdmissionBasis.FEED,
        None,
        Signals(),
        NOW + timedelta(days=5),
    )
    older = repo.add_draft(
        item_id=item_id,
        inputs={"title": "x"},
        pov="",
        language="en",
        model="test-model",
        prompt_version="d1",
        hook="An old hook",
        body="A body",
        hashtags=(),
        source_link="https://example.org/moves",
        check={"flags": [], "counts": {}},
        regenerated_from=None,
        at=NOW,
    )
    newer = repo.add_draft(
        item_id=item_id,
        inputs={"title": "x"},
        pov="",
        language="en",
        model="test-model",
        prompt_version="d1",
        hook="A new hook",
        body="A body",
        hashtags=(),
        source_link="https://example.org/moves",
        check={"flags": [], "counts": {}},
        regenerated_from=None,
        at=NOW + timedelta(hours=1),
    )
    elsewhere = repo.add_draft(
        item_id=other_id,
        inputs={"title": "y"},
        pov="",
        language="en",
        model="test-model",
        prompt_version="d1",
        hook="Another hook",
        body="A body",
        hashtags=(),
        source_link="https://example.org/other",
        check={"flags": [], "counts": {}},
        regenerated_from=None,
        at=NOW,
    )
    db_session.commit()

    detail = page_of(api.get(f"/items/{item_id}"))
    assert [change["changed_at"] for change in detail["tier_history"]] == [
        NOW.isoformat().replace("+00:00", "Z"),
        (NOW + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
        (NOW + timedelta(days=5)).isoformat().replace("+00:00", "Z"),
    ]
    # The page prints the newest draft's date, and never another item's draft.
    assert [draft["id"] for draft in detail["drafts"]] == [str(newer.id), str(older.id)]
    assert str(elsewhere.id) not in {draft["id"] for draft in detail["drafts"]}


def test_an_item_that_is_not_pick_grade_is_early(db_session: Session) -> None:
    api = client(db_session)
    # A vetted preprint from a lab's authors is Rising, so a draft of it must say it is early work.
    preprint = seed(
        db_session,
        "preprint",
        title="A lab's preprint",
        body="An abstract.",
        tier=Tier.VETTED_PREPRINT,
        basis=AdmissionBasis.LAB_AUTHOR,
        commit=False,
    )
    paper = seed(
        db_session,
        "paper",
        title="A peer-reviewed paper",
        body="An abstract.",
        tier=Tier.PEER_REVIEWED,
        basis=AdmissionBasis.VENUE,
        venue="ICLR 2027",
        commit=False,
    )
    db_session.commit()

    assert page_of(api.get(f"/items/{preprint}"))["early"] is True
    assert page_of(api.get(f"/items/{paper}"))["early"] is False
