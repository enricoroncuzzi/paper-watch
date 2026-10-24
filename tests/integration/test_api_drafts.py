"""The drafts routes: generate, check, save, regenerate, list, and never log a point of view.

Every draft is written by a scripted chat model, so these tests are offline and a failure here
means the route, the prompt wiring, the check or the store changed. The model chain in these tests
is what it is in production: one OpenRouter model, with no fallback.
"""

import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.orm import Session
from structlog.testing import capture_logs

import paperwatch.llm.chat as chat_module
from paperwatch.api.app import AgentDeps, create_app
from paperwatch.api.auth import HmacVerifier, mint_hmac_token
from paperwatch.api.limits import RequestLimiter
from paperwatch.api.routes.drafts import DraftService
from paperwatch.api.routes.suggestions import SuggestionService
from paperwatch.api.search_mode import ModeDecision
from paperwatch.config import (
    CONCURRENT_ANSWERS,
    LOOPBACK_HOSTS,
    PROFILES_PATH,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    SUGGESTIONS_PATH,
    Settings,
)
from paperwatch.domain import AdmissionBasis, Signals, SourceItem, Tier, load_profiles
from paperwatch.drafts.generate import DraftSource, generate_draft
from paperwatch.llm.chat import CallGuard, ChatModel, ScriptedChatModel, make_chat_models
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.llm.ledger import Ledger
from paperwatch.store.models import Draft, Item, ItemSummary, SuggestionEvent
from paperwatch.store.repository import Repository
from paperwatch.suggestions.config import load_suggestions

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 3, 9, tzinfo=UTC)
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret
TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents"
URL = "https://arxiv.org/abs/2601.01234"
REPO = "https://github.com/example/replay-buffers"
TEXT = (
    "We present Replay Buffers, a memory layer for long-running coding agents. Replaying exact "
    "tool results and dropping the ones that no longer matter cut tokens per task by 41% on 40 "
    "long-horizon coding tasks, with no loss in task success. The same policy cut retries by 31% "
    "against a summary-based baseline. Every task is Python, and the benchmark is small. Code and "
    f"logs are at {REPO}."
)
POV = (
    "My own harness replays the last twelve tool results. I would test this on our 300-step traces."
)
REPLY: dict[str, Any] = {
    "hook": (
        "Most agent harnesses forget by summarising. The evidence says that is the expensive way."
    ),
    "body": (
        "Replaying tool results, and dropping the ones that stop mattering, cut tokens per task "
        "by 41% on 40 coding tasks and retries by 31%. Every task is Python and the benchmark is "
        "small, so this is a good reason to test the idea, not to adopt it. The code is open."
    ),
    "hashtags": ["agents", "memory", "llm"],
    "source_link": URL,
}


class Clock:
    """A clock a test moves forward, so two drafts have different saved times."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingCalls:
    """The `ChatCallStore` stand-in: counts every reservation the guard makes."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.reserved = 0
        self.failed = 0
        self.filled = 0

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        if self.refuse:
            return None, cap
        self.reserved += 1
        return self.reserved, self.reserved

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        self.filled += 1

    def fail_chat_call(self, call_id: int) -> None:
        self.failed += 1


class NoQueue:
    """Stands in for the summary queue: records what the route asked for, starts no thread."""

    model_id = "test-model"

    def __init__(self) -> None:
        self.submitted: list[UUID] = []

    def submit(self, item_ids: Any) -> None:
        self.submitted.extend(item_ids)


class FixedMode:
    def current(self) -> ModeDecision:
        return ModeDecision("hybrid", "ok")


class AuthedClient(TestClient):
    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", None) or {})
        if "authorization" not in {name.lower() for name in headers}:
            headers["Authorization"] = f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}"
        return super().request(method, url, headers=headers, **kwargs)


def seed_item(
    db_session: Session,
    external_id: str = "replay",
    *,
    title: str = TITLE,
    text: str = TEXT,
    url: str = URL,
    tier: Tier = Tier.PEER_REVIEWED,
    basis: AdmissionBasis | None = None,
    github_repo: str | None = REPO,
    published_at: datetime = NOW - timedelta(days=1),
) -> UUID:
    repo = Repository(db_session)
    item = SourceItem("arxiv", external_id, title, ("Ada Lovelace",), text, url, published_at)
    item_id = repo.upsert_item(
        item,
        tier,
        "agentic-engineering",
        0.93,
        Signals(),
        FakeEmbedder().embed([f"{title}\n{text}"])[0],
    )
    row = db_session.get_one(Item, item_id)
    row.admission_basis = (basis or AdmissionBasis.VENUE).value
    row.venue = "ICLR" if tier is Tier.PEER_REVIEWED else None
    row.github_repo = github_repo
    # Committed, because the routes open and close their own sessions.
    db_session.commit()
    return item_id


def suggestions_service(db_session: Session, queue: Any) -> SuggestionService:
    return SuggestionService(
        lambda: db_session,
        {profile.name: profile for profile in load_profiles(PROFILES_PATH)},
        load_suggestions(SUGGESTIONS_PATH),
        queue,
        Clock(),
    )


def draft_service(
    db_session: Session,
    models: list[ChatModel],
    *,
    calls: RecordingCalls | None = None,
    limiter: RequestLimiter | None = None,
    suggestions: SuggestionService | None = None,
    clock: Clock | None = None,
) -> DraftService:
    return DraftService(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        models_factory=lambda: models,
        settings=Settings(),
        chat_calls=calls or RecordingCalls(),  # type: ignore[arg-type]
        limiter=limiter
        or RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=Clock(),
        ),
        suggestions=suggestions or suggestions_service(db_session, NoQueue()),
        config=load_suggestions(SUGGESTIONS_PATH),
        clock=clock or Clock(),
    )


def _step(step: object) -> object:
    # A string is a model that answered prose; a dict is the JSON reply it was asked for; an
    # exception is a provider that refused.
    if isinstance(step, str | dict):
        # A provider names the model it actually used on every reply.
        return AIMessage(
            step if isinstance(step, str) else json.dumps(step),
            response_metadata={"model_name": "vendor/model:free"},
        )
    return step


def scripted(reply: object = REPLY) -> ChatModel:
    steps = reply if isinstance(reply, list) else [reply]
    return ChatModel("openrouter", ScriptedChatModel(script=[_step(step) for step in steps]))  # type: ignore[list-item]


def client(db_session: Session, service: DraftService) -> AuthedClient:
    # The same suggestions service the drafts service invalidates, or the page's cached ranking
    # would keep showing an item that was already drafted.
    deps = AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=[],
        settings=Settings(),
        clock=Clock(),
        chat_calls=RecordingCalls(),  # type: ignore[arg-type]
        profile_names=frozenset(),
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=Clock(),
        ),
        search_mode=FixedMode(),  # type: ignore[arg-type]
        suggestions=service.suggestions,
        drafts=service,
    )
    return AuthedClient(create_app(deps), base_url="http://localhost")


def store_summary(db_session: Session, item_id: UUID) -> None:
    # A stored summary is part of a draft's inputs; the response's check count does not read it,
    # but the prompt does, and the page shows the same 30-second context in the dialog.
    db_session.add(
        ItemSummary(
            item_id=item_id,
            text_hash="hash",
            model="test-model",
            prompt_version="s1",
            status="checked",
            fields={
                "what": {"text": "A memory layer for coding agents.", "quote": None},
                "result": {
                    "text": "Tokens fall by 41% and retries by 31%.",
                    "quote": "cut tokens per task by 41% on 40 long-horizon coding tasks",
                },
                "why": {"text": "Memory is a retrieval problem.", "interpretation": True},
                "catch": {"text": "Every task is Python.", "interpretation": True},
            },
            failed_checks=[],
            created_at=NOW,
        )
    )
    db_session.commit()


def drafts(db_session: Session) -> list[Draft]:
    db_session.expire_all()
    return list(db_session.scalars(select(Draft).order_by(Draft.created_at)).all())


def test_a_draft_is_generated_saved_and_checked(db_session: Session) -> None:
    item_id = seed_item(db_session)
    store_summary(db_session, item_id)
    calls = RecordingCalls()
    model = scripted()
    api = client(db_session, draft_service(db_session, [model], calls=calls))

    response = api.post("/drafts", json={"item_id": str(item_id), "pov": POV})

    assert response.status_code == 201
    body = response.json()
    assert body["item_id"] == str(item_id)
    assert body["item_title"] == TITLE
    assert body["item_url"] == URL
    assert body["language"] == "en"
    assert body["hook"] == REPLY["hook"]
    assert body["body"] == REPLY["body"]
    assert body["hashtags"] == REPLY["hashtags"]
    assert body["source_link"] == URL
    assert body["flags"] == []
    assert body["counts"]["number"] == [3, 3]
    assert body["chars"] == len(
        f"{REPLY['hook']}\n\n{REPLY['body']}\n\n#agents #memory #llm\n\n{URL}"
    )
    # The model was asked once, with the stored source and the captain's own words in blocks.
    prompt = model.model.seen[0][-1].content
    assert "<source>" in prompt and "<point_of_view>" in prompt
    assert "Replay Buffers" in prompt and "replays the last twelve" in prompt
    assert calls.reserved == 1 and calls.filled == 1

    [saved] = drafts(db_session)
    assert saved.pov == POV
    assert saved.model == "vendor/model:free"
    assert saved.prompt_version == "d1"
    assert saved.inputs["early"] is False
    assert saved.inputs["summary"]["result"]["quote"].startswith("cut tokens per task")
    assert saved.check["counts"]["number"] == [3, 3]


def test_the_pov_is_stored_but_not_returned_by_the_list(db_session: Session) -> None:
    item_id = seed_item(db_session)
    api = client(db_session, draft_service(db_session, [scripted()]))
    created = api.post("/drafts", json={"item_id": str(item_id), "pov": POV}).json()

    listed = api.get("/drafts?limit=50")

    assert listed.status_code == 200
    [row] = listed.json()["drafts"]
    assert row["id"] == created["id"]
    assert "pov" not in row
    detail = api.get(f"/drafts/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["pov"] == POV


def test_the_item_leaves_the_suggestions_after_a_draft(db_session: Session) -> None:
    item_id = seed_item(db_session)
    queue = NoQueue()
    suggestions = suggestions_service(db_session, queue)
    api = client(db_session, draft_service(db_session, [scripted()], suggestions=suggestions))
    assert [card["item_id"] for card in api.get("/suggestions").json()["picks"]] == [str(item_id)]

    assert api.post("/drafts", json={"item_id": str(item_id)}).status_code == 201

    assert api.get("/suggestions").json()["picks"] == []
    drafted = db_session.scalars(
        select(SuggestionEvent).where(
            SuggestionEvent.item_id == item_id, SuggestionEvent.event == "drafted"
        )
    ).all()
    assert len(drafted) == 1


def test_regenerate_keeps_the_old_draft_and_links_the_new_one(db_session: Session) -> None:
    item_id = seed_item(db_session)
    clock = Clock()
    model = scripted([REPLY, {**REPLY, "hook": "Again."}])
    api = client(db_session, draft_service(db_session, [model], clock=clock))
    first = api.post("/drafts", json={"item_id": str(item_id), "pov": POV}).json()
    clock.now = NOW + timedelta(seconds=30)

    # The gate accepts a POST only as JSON; the page's own proxy sends the same empty body.
    second = api.post(f"/drafts/{first['id']}/regenerate", json={})

    assert second.status_code == 201
    regenerated = second.json()
    assert regenerated["id"] != first["id"]
    assert regenerated["regenerated_from"] == first["id"]
    assert regenerated["hook"] == "Again."
    listed = api.get("/drafts").json()["drafts"]
    assert [row["id"] for row in listed] == [regenerated["id"], first["id"]]
    # The same inputs and the same point of view, stored with the new draft.
    [saved] = [row for row in drafts(db_session) if str(row.id) == regenerated["id"]]
    assert saved.pov == POV


def test_a_rising_item_is_marked_early(db_session: Session) -> None:
    item_id = seed_item(db_session, tier=Tier.VETTED_PREPRINT, basis=AdmissionBasis.LAB_AUTHOR)
    reply = {
        **REPLY,
        "hook": "An early memory layer for coding agents.",
        "body": (
            "The authors report that replaying tool results cut tokens per task by 41% on 40 "
            "tasks. The work is early and not yet independently reviewed."
        ),
    }
    model = scripted(reply)
    api = client(db_session, draft_service(db_session, [model]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 201
    assert response.json()["flags"] == []
    # The model was told the item is early, so the notice is asked for rather than hoped for.
    assert "early" in model.model.seen[0][-1].content
    [saved] = drafts(db_session)
    assert saved.inputs["early"] is True


def test_the_pov_never_reaches_the_logs(
    db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    # The point of view is the captain's own text: it is stored with the draft and sent only to the
    # provider that refuses training. Not a log line, not a refusal body, not an exception message.
    phrase = "quartz-zebra-4417-my-own-harness"
    first = seed_item(db_session, "replay")
    second = seed_item(db_session, "refusal", published_at=NOW - timedelta(days=2))
    third = seed_item(db_session, "unusable", published_at=NOW - timedelta(days=3))
    model = scripted(
        [
            REPLY,
            RuntimeError(f"provider echoed: {phrase}"),
            "this is prose, not the JSON I asked for",
        ]
    )
    api = client(db_session, draft_service(db_session, [model]))

    # uvicorn's own logger level can be left at ERROR by another test's real server, so this test
    # sets it for its own duration; the capture then sees the line it would have logged.
    with capture_logs() as logs:
        caplog.set_level(logging.DEBUG)
        caplog.set_level(logging.INFO, logger="uvicorn.access")
        ok = api.post("/drafts", json={"item_id": str(first), "pov": phrase})
        refused = api.post("/drafts", json={"item_id": str(second), "pov": phrase})
        unusable = api.post("/drafts", json={"item_id": str(third), "pov": phrase})
        # The access logger is the server's own: prove this capture would have seen its line, so
        # an empty result means no leak rather than a logger the test never watched.
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "127.0.0.1", "POST", "/drafts", "1.1", 201
        )

    assert ok.status_code == 201
    assert refused.status_code == 503 and unusable.status_code == 503
    assert "POST /drafts" in caplog.text
    captured = json.dumps(logs, default=str) + caplog.text
    assert phrase not in captured
    # The draft text, the stored text and the title are as private as the point of view.
    assert REPLY["hook"] not in captured
    assert REPLY["body"][:40] not in captured
    assert "Replaying exact tool results" not in captured
    assert TITLE not in captured
    for response in (ok, refused, unusable):
        assert phrase not in response.text
    # The refusal keeps the provider's error type for the operator and nothing else.
    assert refused.json()["message"] == (
        "No model that refuses training on inputs answered, so the draft was not generated."
    )


def test_daily_cap_is_503_with_retry_at(db_session: Session) -> None:
    item_id = seed_item(db_session)
    api = client(
        db_session, draft_service(db_session, [scripted()], calls=RecordingCalls(refuse=True))
    )

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "daily_cap"
    assert body["retry_at"].startswith("2026-11-04T00:00:00")
    assert drafts(db_session) == []


def test_a_provider_refusal_fails_without_a_fallback(db_session: Session) -> None:
    item_id = seed_item(db_session)
    calls = RecordingCalls()
    primary = scripted(RuntimeError("provider refused"))
    fallback = ScriptedChatModel(script=[AIMessage(json.dumps(REPLY))])
    api = client(
        db_session,
        draft_service(
            db_session,
            [
                ChatModel("openrouter", primary),
                ChatModel("groq", fallback),
            ],
            calls=calls,
        ),
    )

    response = api.post("/drafts", json={"item_id": str(item_id), "pov": POV})

    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"
    # The point of view may only go to a provider that refuses training, so there is no second try.
    assert fallback.seen == []
    # An attempt that reached the provider was spent, even though it produced no draft.
    assert calls.failed == 1 and calls.filled == 0
    assert drafts(db_session) == []


def test_draft_requests_count_against_the_limiter(db_session: Session) -> None:
    item_id = seed_item(db_session)
    limiter = RequestLimiter(
        per_minute=1,
        per_day=REQUESTS_PER_DAY,
        concurrent=CONCURRENT_ANSWERS,
        clock=Clock(),
    )
    model = scripted([REPLY, REPLY])
    api = client(db_session, draft_service(db_session, [model], limiter=limiter))

    assert api.post("/drafts", json={"item_id": str(item_id)}).status_code == 201
    created = api.get("/drafts").json()["drafts"][0]["id"]
    regenerated = api.post(f"/drafts/{created}/regenerate", json={})
    assert regenerated.status_code == 429
    refused = api.post("/drafts", json={"item_id": str(item_id)})

    assert refused.status_code == 429
    assert refused.json()["detail"] == "rate_limited"
    assert refused.headers["retry-after"] == str(refused.json()["retry_after_s"])
    assert len(model.model.seen) == 1


def test_an_unusable_reply_saves_nothing(db_session: Session) -> None:
    item_id = seed_item(db_session)
    api = client(db_session, draft_service(db_session, [scripted("not json at all")]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 503
    assert response.json()["code"] == "unusable"
    assert drafts(db_session) == []


def test_a_non_admitted_item_is_404(db_session: Session) -> None:
    item_id = seed_item(db_session)
    row = db_session.get_one(Item, item_id)
    row.status = "candidate"
    row.tier = None
    row.admission_basis = None
    db_session.commit()
    model = scripted()
    api = client(db_session, draft_service(db_session, [model]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 404
    assert model.model.seen == []


def test_short_stored_text_is_422(db_session: Session) -> None:
    item_id = seed_item(db_session, text="Too short to draft from.")
    model = scripted()
    api = client(db_session, draft_service(db_session, [model]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 422
    assert model.model.seen == []


def test_the_pov_length_is_capped_at_two_thousand_characters(db_session: Session) -> None:
    item_id = seed_item(db_session)
    model = scripted()
    api = client(db_session, draft_service(db_session, [model]))

    response = api.post("/drafts", json={"item_id": str(item_id), "pov": "x" * 2001})

    assert response.status_code == 422
    assert model.model.seen == []


def test_a_flagged_draft_is_saved_as_written(db_session: Session) -> None:
    item_id = seed_item(db_session)
    invented = {
        **REPLY,
        "body": REPLY["body"].replace("41%", "45%"),
    }
    api = client(db_session, draft_service(db_session, [scripted(invented)]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 201
    body = response.json()
    [flag] = body["flags"]
    assert (flag["n"], flag["kind"]) == (1, "number")
    assert "45%" in flag["text"]
    # The checker never edits the text: the draft is saved as the model wrote it.
    assert "45%" in body["body"]
    [saved] = drafts(db_session)
    assert saved.body == invented["body"]
    assert saved.check["flags"][0]["kind"] == "number"


def test_a_404_or_a_422_spends_no_limiter_slot(db_session: Session) -> None:
    good = seed_item(db_session)
    short = seed_item(db_session, "short", text="Too short to draft from.")
    limiter = RequestLimiter(
        per_minute=1,
        per_day=REQUESTS_PER_DAY,
        concurrent=CONCURRENT_ANSWERS,
        clock=Clock(),
    )
    model = scripted()
    api = client(db_session, draft_service(db_session, [model], limiter=limiter))

    assert api.post("/drafts", json={"item_id": str(uuid4())}).status_code == 404
    assert api.post("/drafts", json={"item_id": str(short)}).status_code == 422
    # Neither refusal reached the model, so neither spent the one slot this client has.
    assert api.post("/drafts", json={"item_id": str(good)}).status_code == 201
    assert len(model.model.seen) == 1


def test_regenerate_keeps_the_early_flag_and_the_stored_summary(db_session: Session) -> None:
    item_id = seed_item(db_session, tier=Tier.VETTED_PREPRINT, basis=AdmissionBasis.LAB_AUTHOR)
    store_summary(db_session, item_id)
    reply = {
        **REPLY,
        "hook": "An early memory layer for coding agents.",
        "body": (
            "The authors report that replaying tool results cut tokens per task by 41% on 40 "
            "tasks. The work is early and not yet independently reviewed."
        ),
    }
    api = client(db_session, draft_service(db_session, [scripted([reply, reply])]))
    first = api.post("/drafts", json={"item_id": str(item_id)}).json()

    second = api.post(f"/drafts/{first['id']}/regenerate", json={})

    assert second.status_code == 201
    [saved] = [row for row in drafts(db_session) if str(row.id) == second.json()["id"]]
    # A regenerate re-runs the stored inputs, so the early notice and the summary survive it.
    assert saved.inputs["early"] is True
    summary = saved.inputs["summary"]
    assert isinstance(summary, dict)
    assert summary["result"]["quote"].startswith("cut tokens per task")


def test_a_failed_drafted_event_still_saves_the_draft(db_session: Session) -> None:
    item_id = seed_item(db_session)
    suggestions = suggestions_service(db_session, NoQueue())

    def broken(item_id: UUID, event: str) -> None:
        raise RuntimeError("the event store is down")

    suggestions.record = broken  # type: ignore[method-assign]
    api = client(db_session, draft_service(db_session, [scripted()], suggestions=suggestions))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    # The model was paid for: a failed event must not turn a saved draft into an error the reader
    # would retry into a duplicate.
    assert response.status_code == 201
    assert len(drafts(db_session)) == 1


def test_an_excerpt_summary_is_not_sent_as_context(db_session: Session) -> None:
    item_id = seed_item(db_session)
    db_session.add(
        ItemSummary(
            item_id=item_id,
            text_hash="hash",
            model="test-model",
            prompt_version="s1",
            status="excerpt",
            fields={"excerpt": "The abstract's own first sentences."},
            failed_checks=["shape"],
            created_at=NOW,
        )
    )
    db_session.commit()
    api = client(db_session, draft_service(db_session, [scripted()]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 201
    [saved] = drafts(db_session)
    # An excerpt is the stored text again, not a checked summary: it must not become context.
    assert saved.inputs["summary"] is None


def test_the_point_of_view_limit_and_language_are_enforced(db_session: Session) -> None:
    item_id = seed_item(db_session)
    api = client(db_session, draft_service(db_session, [scripted()]))

    assert api.post("/drafts", json={"item_id": str(item_id), "pov": "x" * 2000}).status_code == 201
    assert api.post("/drafts", json={"item_id": str(item_id), "pov": "x" * 2001}).status_code == 422
    assert (
        api.post("/drafts", json={"item_id": str(item_id), "language": "en\nx"}).status_code == 422
    )


def test_the_draft_list_limit_is_bounded(db_session: Session) -> None:
    seed_item(db_session)
    api = client(db_session, draft_service(db_session, [scripted()]))
    assert api.get("/drafts", params={"limit": 1}).status_code == 200
    assert api.get("/drafts", params={"limit": 0}).status_code == 422
    assert api.get("/drafts", params={"limit": 51}).status_code == 422


def test_the_hashtags_are_normalised(db_session: Session) -> None:
    item_id = seed_item(db_session)
    reply = {**REPLY, "hashtags": ["#agents", "ai safety"]}
    api = client(db_session, draft_service(db_session, [scripted(reply)]))

    response = api.post("/drafts", json={"item_id": str(item_id)})

    assert response.status_code == 201
    assert response.json()["hashtags"] == ["agents", "aisafety"]


class _DraftProvider(BaseHTTPRequestHandler):
    """A local OpenAI-compatible endpoint, so the real ChatOpenAI path is exercised offline."""

    bodies: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        type(self).bodies.append(self.rfile.read(length))
        payload = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "vendor/model:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(REPLY)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return  # the server's own access line is noise in a test


def test_the_real_draft_chain_denies_data_collection_and_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The configured ChatOpenAI object is tested in test_chat.py; this drives the real client, so
    # the wire body and the openai/httpx loggers are covered too.
    phrase = "quartz-zebra-4417-my-own-harness"
    _DraftProvider.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DraftProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        chat_module,
        "OPENROUTER_BASE_URL",
        f"http://127.0.0.1:{server.server_address[1]}/api/v1",
    )
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        openrouter_api_key="or-key",
        groq_api_key="g",
        groq_chat_model="some-model",
    )
    models = make_chat_models(settings, purpose="draft")
    source = DraftSource(
        item_id=uuid4(),
        title=TITLE,
        url=URL,
        source="arxiv",
        tier="peer_reviewed",
        basis_label="peer-reviewed via venue: ICLR",
        venue="ICLR",
        org=None,
        text=TEXT,
        github_repo=REPO,
        project_page=None,
        summary=None,
        early=False,
    )
    guard = CallGuard(RecordingCalls(), Ledger(100_000), 800, Clock())  # type: ignore[arg-type]
    try:
        with capture_logs() as logs, caplog.at_level(logging.DEBUG):
            text, answered = generate_draft(models, guard, source, pov=phrase, language="en")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert text.hook == REPLY["hook"]
    assert answered == "vendor/model:free"
    # One model, OpenRouter only: a Groq key and model were configured and never built.
    assert [model.provider for model in models] == ["openrouter"]
    assert len(_DraftProvider.bodies) == 1
    body = json.loads(_DraftProvider.bodies[0])
    assert body["provider"] == {"data_collection": "deny"}
    # The point of view may go to this provider (that is the feature), but never to a log line.
    assert phrase in json.dumps(body)
    captured = json.dumps(logs, default=str) + caplog.text
    assert phrase not in captured
