"""The runs and today routes the page reads.

These tests cover the wire shape the page depends on, not the ingestion: every run and dead letter
is seeded through the repository, so a failure here means the read path changed rather than the
pipeline.
"""

import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperwatch.api.app import API_VERSION, AgentDeps, create_app
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
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.store.models import DeadLetter
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 16, 6, tzinfo=UTC)
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret
# The fixed counts key set the page renders as its six cells; the route fills absent keys with 0.
COUNT_KEYS = [
    "fetched",
    "fetched_arxiv",
    "stored",
    "new",
    "irrelevant",
    "failed",
    "admitted_late",
    "followup_passes",
    "s2_refused",
]


class FixedMode:
    def __init__(self, decision: ModeDecision) -> None:
        self.decision = decision

    def current(self) -> ModeDecision:
        return self.decision


class NoChatCalls:
    def record_chat_call(self, **kwargs: Any) -> None:
        raise AssertionError("a run read makes no chat call")

    def chat_calls_since(self, at: datetime) -> int:
        raise AssertionError("a run read makes no chat call")


class AuthedClient(TestClient):
    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", None) or {})
        if "authorization" not in {name.lower() for name in headers}:
            headers["Authorization"] = f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}"
        return super().request(method, url, headers=headers, **kwargs)


def app_deps(db_session: Session, mode: ModeDecision | None = None) -> AgentDeps:
    return AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=[],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=NoChatCalls(),
        profile_names=frozenset(),
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=FixedMode(mode or ModeDecision("hybrid", "ok")),  # type: ignore[arg-type]
    )


def client(db_session: Session, mode: ModeDecision | None = None) -> AuthedClient:
    # localhost, not TestClient's default testserver, so the host allow-list accepts the request.
    return AuthedClient(create_app(app_deps(db_session, mode)), base_url="http://localhost")


def finished_run(
    db_session: Session,
    *,
    started_at: datetime = NOW,
    counts: dict[str, int] | None = None,
    tokens: int = 0,
    health: dict[str, object] | None = None,
    status: str = "succeeded",
) -> uuid.UUID:
    repo = Repository(db_session)
    run_id = repo.start_run(started_at - timedelta(hours=6), started_at, at=started_at)
    repo.finish_run(
        run_id,
        status,
        counts if counts is not None else {"fetched": 1, "stored": 1},
        tokens,
        health,
        at=started_at + timedelta(minutes=3),
    )
    db_session.flush()
    return run_id


def test_runs_lists_newest_first_with_health(db_session: Session) -> None:
    older = finished_run(
        db_session,
        started_at=NOW - timedelta(days=1),
        counts={"fetched": 40, "fetched_arxiv": 30, "stored": 9, "new": 4},
        tokens=1200,
        health={"healthy": True, "problems": [], "warnings": []},
    )
    newer = finished_run(
        db_session,
        counts={"fetched": 12, "stored": 8},
        tokens=3456,
        health={"healthy": False, "problems": ["arxiv fetched 0 items"], "warnings": ["x"]},
    )

    body = client(db_session).get("/runs").json()

    assert [run["id"] for run in body["runs"]] == [str(newer), str(older)]
    newest, oldest = body["runs"]
    assert newest["healthy"] is False
    assert newest["problems"] == ["arxiv fetched 0 items"]
    assert newest["warnings"] == ["x"]
    assert newest["tokens"] == 3456
    assert newest["status"] == "succeeded"
    assert newest["finished_at"] is not None
    # A key the run never counted is 0, so the page renders a number instead of a blank.
    assert newest["counts"] == {key: 0 for key in COUNT_KEYS} | {"fetched": 12, "stored": 8}
    assert oldest["healthy"] is True
    assert oldest["counts"]["fetched_arxiv"] == 30


def test_a_waiting_follow_up_is_reported(db_session: Session) -> None:
    next_at = NOW + timedelta(hours=2)
    followup = {
        "pass": 3,
        "of": 40,
        "next_at": next_at.isoformat(),
        "pending": {"s2": 12},
    }
    finished_run(
        db_session,
        counts={"followup_passes": 3},
        health={"healthy": True, "problems": [], "warnings": [], "followup": followup},
    )

    run = client(db_session).get("/runs").json()["runs"][0]

    assert run["followup"] is not None
    assert run["followup"]["pass"] == 3
    assert run["followup"]["of"] == 40
    assert run["followup"]["pending_s2"] == 12
    assert datetime.fromisoformat(run["followup"]["next_at"]) == next_at
    assert run["counts"]["followup_passes"] == 3


def test_a_run_without_follow_up_reports_none(db_session: Session) -> None:
    finished_run(db_session, health={"healthy": True, "problems": [], "warnings": []})

    assert client(db_session).get("/runs").json()["runs"][0]["followup"] is None


def test_run_detail_lists_its_dead_letters(db_session: Session) -> None:
    repo = Repository(db_session)
    run_id = finished_run(db_session, health={"healthy": True, "problems": [], "warnings": []})
    long_error = "boom " * 200
    long_ref = "ref-" + "x" * 600
    repo.add_dead_letter(run_id, "feed", "https://example.org/feed", long_error)
    repo.add_dead_letter(run_id, "feed", long_ref, "short error")
    repo.add_dead_letter(run_id, "arxiv", "2601.00001", "timeout")
    # `add_dead_letter` returns nothing, so the resolved letter is found by its ref.
    resolved = db_session.scalar(select(DeadLetter.id).where(DeadLetter.ref == "2601.00001"))
    assert resolved is not None
    repo.resolve_dead_letter(resolved, at=NOW + timedelta(minutes=1))
    db_session.flush()

    detail = client(db_session).get(f"/runs/{run_id}").json()

    assert detail["id"] == str(run_id)
    assert detail["window_start"] is not None
    assert detail["window_end"] is not None
    letters = {letter["ref"]: letter for letter in detail["dead_letters"]}
    assert len(letters) == 3
    assert letters["https://example.org/feed"]["source"] == "feed"
    assert letters["2601.00001"]["source"] == "arxiv"
    # A provider message can be a whole page of HTML; the page shows a line, so the route clips.
    assert len(letters["https://example.org/feed"]["error"]) == 500
    assert letters["https://example.org/feed"]["attempts"] == 1
    # `ref` is clipped like `error`: a stored id can be a whole URL.
    clipped = ("ref-" + "x" * 600)[:500]
    assert letters[clipped]["ref"] == clipped
    assert letters["2601.00001"]["resolved_at"] is not None
    assert letters["2601.00001"]["error"] == "timeout"


def test_an_unknown_run_is_404(db_session: Session) -> None:
    assert client(db_session).get(f"/runs/{uuid.uuid4()}").status_code == 404


def test_today_is_null_before_the_first_run(db_session: Session) -> None:
    body = client(db_session).get("/today").json()

    assert body["run"] is None
    assert body["search"] == {"mode": "hybrid", "reason": "ok", "since": None}


def test_today_reports_the_last_run_and_the_search_mode(db_session: Session) -> None:
    older = finished_run(db_session, started_at=NOW - timedelta(days=1))
    newer = finished_run(
        db_session,
        health={"healthy": False, "problems": ["3 items failed"], "warnings": []},
    )
    decision = ModeDecision("text_only", "ingestion", since=NOW - timedelta(minutes=5))

    body = client(db_session, decision).get("/today").json()

    assert body["run"]["id"] == str(newer)
    assert body["run"]["id"] != str(older)
    assert body["run"]["problems"] == ["3 items failed"]
    assert body["search"]["mode"] == "text_only"
    assert body["search"]["reason"] == "ingestion"
    assert datetime.fromisoformat(body["search"]["since"]) == NOW - timedelta(minutes=5)


def test_a_failed_run_closed_by_the_next_reads_healthy_null(db_session: Session) -> None:
    # `fail_open_runs` marks a crashed run failed without a verdict, so the page must render
    # "unknown", not "unhealthy".
    finished_run(db_session, status="failed", health=None)

    body = client(db_session).get("/today").json()

    assert body["run"]["status"] == "failed"
    assert body["run"]["healthy"] is None
    assert body["run"]["problems"] == []
    assert body["run"]["warnings"] == []


def test_a_malformed_follow_up_reads_as_none(db_session: Session) -> None:
    # A bad health record must not 500 the whole list: the strip just shows no waiting pass.
    finished_run(
        db_session,
        health={
            "healthy": True,
            "problems": [],
            "warnings": [],
            "followup": {"pass": "three", "of": 40},
        },
    )

    assert client(db_session).get("/runs").json()["runs"][0]["followup"] is None


def test_runs_and_today_need_a_credential(db_session: Session) -> None:
    # The gate covers every route; these two must not be an exception the page can lean on.
    plain = TestClient(create_app(app_deps(db_session)), base_url="http://localhost")

    assert plain.get("/runs").status_code == 401
    assert plain.get("/today").status_code == 401
    assert plain.get(f"/runs/{uuid.uuid4()}").status_code == 401


def test_limit_is_bounded(db_session: Session) -> None:
    finished_run(db_session)

    assert client(db_session).get("/runs", params={"limit": 50}).status_code == 200
    assert client(db_session).get("/runs", params={"limit": 1}).status_code == 200
    assert client(db_session).get("/runs", params={"limit": 0}).status_code == 422
    assert client(db_session).get("/runs", params={"limit": 51}).status_code == 422


def test_the_contract_names_version_0_6_and_the_page_routes(db_session: Session) -> None:
    document = client(db_session).app.openapi()

    assert API_VERSION == "0.6"
    # The suggestions, drafts and threads routes ship in the same version, so the page's client is
    # generated once and its thread types exist beside the card and draft types.
    assert set(document["paths"]) >= {
        "/runs",
        "/runs/{run_id}",
        "/today",
        "/suggestions",
        "/suggestions/{item_id}/events",
        "/drafts",
        "/drafts/{draft_id}",
        "/drafts/{draft_id}/regenerate",
        "/threads",
        "/threads/{thread_id}",
    }
    assert "RunSummary" in document["components"]["schemas"]
    assert "DraftData" in document["components"]["schemas"]
    assert "ThreadDetailData" in document["components"]["schemas"]
