"""The suggestions route: the ranked list, its summaries and the interactions it records.

Every item is seeded through the repository and every summary is written by a scripted chat model,
so these tests run offline and a failure here means the route, the ranking wiring or the queue
changed rather than the ingestion or a provider.
"""

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from paperwatch.agent.graph import AnswerEvent
from paperwatch.agent.tools import GET_ITEM_MAX_CHARS
from paperwatch.api.app import AgentDeps, create_app, run_question
from paperwatch.api.auth import HmacVerifier, mint_hmac_token
from paperwatch.api.limits import RequestLimiter
from paperwatch.api.routes.suggestions import SuggestionService
from paperwatch.api.search_mode import ModeDecision
from paperwatch.api.summary_queue import SummaryQueue
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
from paperwatch.llm.chat import CallGuard, ChatModel, ModelUnavailable, ScriptedChatModel
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.llm.ledger import Ledger
from paperwatch.retrieval.hybrid import Filters
from paperwatch.store.models import Item, ItemSummary, SignalSnapshot, SuggestionEvent
from paperwatch.store.repository import Repository
from paperwatch.suggestions.config import load_suggestions
from paperwatch.suggestions.summary import SummaryWriter, text_hash

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 2, 9, tzinfo=UTC)
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret
TEXT = (
    "We introduce AgentBench-Lite, a benchmark of 41 tasks for tool-using language agents. "
    "Agents built on Llama-3.1-8B solve 41% of the tasks, up from 28% for the previous best "
    "baseline. The gain comes from a two-stage planner that rewrites each tool call before it "
    "runs. We do not measure cost, latency or the planner's own token budget, and every task, "
    "the harness and the two-stage planner ship as open code for other labs to run and extend."
)
TITLE = "AgentBench-Lite: a small benchmark for tool-using agents"
# A second, clearly different item: two items with one body are near-duplicates by design and the
# ranking merges them into one card.
SURVEY_TITLE = "A survey of memory designs for long-horizon agents"
SURVEY = (
    "We survey 120 papers on agent memory and organise them by what is stored, when it is read "
    "and how it is forgotten. We find that long-term memory helps only on tasks longer than 50 "
    "steps, that summarisation-based memory loses less than retrieval-based memory across "
    "sessions, and that most reported gains shrink when the retrieval budget is held constant. "
    "We release the annotated table, the harness that reproduces every number, and the raw logs "
    "of the 40 runs behind our main claim."
)


def _raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "what": {
            "text": "The authors release AgentBench-Lite, a benchmark of 41 tasks for "
            "tool-using language agents.",
            "quote": "a benchmark of 41 tasks for tool-using language agents",
        },
        "result": {
            "text": "Llama-3.1-8B agents solve 41% of the tasks against 28% for the previous "
            "best baseline.",
            "quote": "solve 41% of the tasks, up from 28% for the previous best baseline",
        },
        "why": {"text": "A smaller agent benchmark makes harness comparisons cheaper to run."},
        "catch": {"text": "Not measured in the abstract."},
    }
    raw.update(overrides)
    return raw


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


class NoChatCalls:
    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        return 1, 1

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        return None

    def fail_chat_call(self, call_id: int) -> None:
        return None


class AuthedClient(TestClient):
    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", None) or {})
        if "authorization" not in {name.lower() for name in headers}:
            headers["Authorization"] = f"Bearer {mint_hmac_token(TEST_SECRET, 'test', time.time())}"
        return super().request(method, url, headers=headers, **kwargs)


def seed_item(
    db_session: Session,
    external_id: str = "swe",
    *,
    title: str = TITLE,
    text: str = TEXT,
    tier: Tier = Tier.PEER_REVIEWED,
    basis: AdmissionBasis | None = None,
    published_at: datetime = NOW - timedelta(days=1),
    hf_upvotes: int | None = None,
) -> UUID:
    repo = Repository(db_session)
    item = SourceItem(
        "arxiv",
        external_id,
        title,
        ("Ada Lovelace",),
        text,
        f"https://example.org/{external_id}",
        published_at,
    )
    item_id = repo.upsert_item(
        item,
        tier,
        "agentic-engineering",
        0.93,
        Signals(hf_upvotes=hf_upvotes),
        FakeEmbedder().embed([f"{title}\n{text}"])[0],
    )
    row = db_session.get_one(Item, item_id)
    # The admission basis is what the ranking reads, so a test sets it directly on the stored row.
    row.admission_basis = (basis or AdmissionBasis.VENUE).value
    row.venue = "ICLR" if tier is Tier.PEER_REVIEWED else None
    row.hf_upvotes = hf_upvotes
    # Committed, because the route opens and closes its own sessions: a flush alone would be
    # rolled back by the first of them.
    db_session.commit()
    return item_id


def service(
    db_session: Session, queue: Any, *, models: list[ChatModel] | None = None
) -> SuggestionService:
    session_factory = lambda: db_session  # noqa: E731 - a stand-in session factory
    return SuggestionService(
        session_factory,
        {profile.name: profile for profile in load_profiles(PROFILES_PATH)},
        load_suggestions(SUGGESTIONS_PATH),
        queue,
        lambda: NOW,
    )


def client(
    db_session: Session, queue: Any, *, models: list[ChatModel] | None = None
) -> AuthedClient:
    deps = AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=models or [],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=NoChatCalls(),  # type: ignore[arg-type]
        profile_names=frozenset(),
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=FixedMode(),  # type: ignore[arg-type]
        suggestions=service(db_session, queue),
    )
    return AuthedClient(create_app(deps), base_url="http://localhost")


def test_the_list_returns_at_once_with_pending_summaries(db_session: Session) -> None:
    queue = NoQueue()
    first = seed_item(db_session, "swe")
    seed_item(
        db_session,
        "mem",
        title=SURVEY_TITLE,
        text=SURVEY,
        tier=Tier.VETTED_PREPRINT,
        basis=AdmissionBasis.HF_UPVOTES,
    )

    response = client(db_session, queue).get("/suggestions")

    assert response.status_code == 200
    body = response.json()
    assert body["rising_note"] == "Rising, not yet independently vetted."
    assert len(body["picks"]) == 2
    lead = body["picks"][0]
    assert lead["summary"]["status"] == "pending"
    # The page shows something true while the summary is written: the abstract's own sentences.
    assert lead["summary"]["excerpt"].startswith("We introduce AgentBench-Lite")
    assert lead["basis_label"] == "peer-reviewed via venue: ICLR"
    assert lead["also"] == []
    # Every visible card is queued for a summary, and the route did not wait for one.
    assert set(queue.submitted) == {first, UUID(body["picks"][1]["item_id"])}


def test_a_finished_summary_appears_on_the_next_request(db_session: Session) -> None:
    queue = NoQueue()
    item_id = seed_item(db_session, "swe")
    api = client(db_session, queue)
    assert api.get("/suggestions").json()["picks"][0]["summary"]["status"] == "pending"

    # What the queue does in the background: one checked summary, stored against the source text.
    summary = checked_summary()
    assert summary.status == "checked"
    Repository(db_session).upsert_summary(
        item_id, summary, text_hash=text_hash(TITLE, TEXT), model="test-model", at=NOW
    )
    db_session.flush()

    body = api.get("/suggestions").json()

    card = body["picks"][0]
    assert card["summary"]["status"] == "checked"
    assert card["summary"]["what"]["quote"].startswith("a benchmark of 41 tasks")
    assert card["summary"]["why"]["interpretation"] is True
    assert card["summary"]["catch"]["quote"] is None


def test_a_dismissed_item_leaves_the_list_immediately(db_session: Session) -> None:
    queue = NoQueue()
    dismissed = seed_item(db_session, "swe")
    kept = seed_item(
        db_session, "mem", title=SURVEY_TITLE, text=SURVEY, published_at=NOW - timedelta(days=3)
    )
    api = client(db_session, queue)
    before = {card["item_id"] for card in api.get("/suggestions").json()["picks"]}
    assert before == {str(dismissed), str(kept)}

    posted = api.post(f"/suggestions/{dismissed}/events", json={"event": "dismissed"})

    assert posted.status_code == 204
    after = {card["item_id"] for card in api.get("/suggestions").json()["picks"]}
    assert after == {str(kept)}


def test_shown_is_recorded_at_most_once_an_hour(db_session: Session) -> None:
    queue = NoQueue()
    item_id = seed_item(db_session, "swe")
    api = client(db_session, queue)

    api.get("/suggestions")
    api.get("/suggestions")

    shown = db_session.scalars(
        select(SuggestionEvent).where(
            SuggestionEvent.item_id == item_id, SuggestionEvent.event == "shown"
        )
    ).all()
    assert len(shown) == 1
    # The event records what the reader saw: the rank, the score and its components.
    assert shown[0].rank == 1
    assert shown[0].score is not None
    assert set(shown[0].components or {}) >= {"momentum", "recency", "credibility"}
    assert shown[0].list_version is not None


def test_rising_cards_carry_the_note_and_their_basis(db_session: Session) -> None:
    queue = NoQueue()
    seed_item(db_session, "swe")
    rising_id = seed_item(
        db_session,
        "viral",
        title=SURVEY_TITLE,
        text=SURVEY,
        tier=Tier.VETTED_PREPRINT,
        basis=AdmissionBasis.LAB_AUTHOR,
        hf_upvotes=400,
        published_at=NOW - timedelta(hours=6),
    )

    body = client(db_session, queue).get("/suggestions").json()

    assert body["rising_note"] == "Rising, not yet independently vetted."
    card = next(entry for entry in body["rising"] if entry["item_id"] == str(rising_id))
    assert card["basis"] == "lab_author"
    assert card["basis_label"] == (
        "preprint from a recognised lab's authors, not yet independently vetted"
    )


class Refusal:
    """A switch the cap tests flip without rebuilding the queue."""

    def __init__(self, refuse: bool) -> None:
        self.refuse = refuse


class Store:
    def __init__(self, refusal: Refusal) -> None:
        self.refusal = refusal

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int | None, int]:
        return (None, cap) if self.refusal.refuse else (1, 1)

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        return None

    def fail_chat_call(self, call_id: int) -> None:
        return None


def queue_for(
    db_session: Session,
    script: list[Any],
    *,
    refuse: bool | Refusal = False,
    clock: Any = None,
) -> SummaryQueue:
    refusal = refuse if isinstance(refuse, Refusal) else Refusal(refuse)
    writer = SummaryWriter(
        [ChatModel("openrouter", ScriptedChatModel(script=script))],
        lambda: CallGuard(Store(refusal), Ledger(10_000), 10, lambda: NOW),  # type: ignore[arg-type]
    )
    return SummaryQueue(
        lambda: writer,
        lambda: db_session,
        clock or (lambda: NOW),
        lambda: "test-model",
    )


def test_a_changed_abstract_regenerates_its_summary(db_session: Session) -> None:
    queue = NoQueue()
    item_id = seed_item(db_session, "swe")
    Repository(db_session).upsert_summary(
        item_id,
        checked_summary(),
        text_hash="an-old-abstract",
        model="test-model",
        at=NOW - timedelta(days=2),
    )
    db_session.flush()

    summary_queue = queue_for(db_session, [AIMessage(content=json.dumps(_raw()))])
    summary_queue.submit([item_id])
    summary_queue.join(timeout=10)

    row = db_session.get_one(ItemSummary, item_id)
    assert row.text_hash == text_hash(TITLE, TEXT)
    assert row.status == "checked"
    # The list the reader sees is the regenerated one.
    body = client(db_session, queue).get("/suggestions").json()
    assert body["picks"][0]["summary"]["status"] == "checked"


def test_the_daily_cap_leaves_summaries_pending_without_errors(db_session: Session) -> None:
    queue = NoQueue()
    item_id = seed_item(db_session, "swe")

    summary_queue = queue_for(db_session, [], refuse=True)
    summary_queue.submit([item_id])
    summary_queue.join(timeout=10)

    assert db_session.get(ItemSummary, item_id) is None
    response = client(db_session, queue).get("/suggestions")
    assert response.status_code == 200
    assert response.json()["picks"][0]["summary"]["status"] == "pending"


def test_asking_about_an_item_registers_it_as_s1(db_session: Session) -> None:
    item_id = seed_item(db_session, "swe")
    db_session.commit()
    model = ScriptedChatModel(script=[AIMessage(content="It is a 41-task benchmark [S1].")])
    deps = AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=[ChatModel("openrouter", model)],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=NoChatCalls(),  # type: ignore[arg-type]
        profile_names=frozenset(),
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=FixedMode(),  # type: ignore[arg-type]
        suggestions=service(db_session, NoQueue()),
    )

    events = list(run_question(deps, "What does this paper measure?", Filters(), item_id=item_id))

    # The model was asked about S1 before any tool ran, and it could cite it.
    asked = "\n".join(str(message.content) for message in model.seen[0])
    assert "The reader is asking about S1." in asked
    # The item itself travels in the escaped block, so the model can answer from it and not from
    # an alias it has never seen.
    assert "<source>" in asked
    assert TITLE in asked
    assert "41 tasks" in asked
    answer = next(event for event in events if isinstance(event, AnswerEvent))
    assert [citation.alias for citation in answer.answer.citations] == ["S1"]
    assert answer.answer.citations[0].item_id == item_id


def test_the_ask_route_forwards_item_id_to_the_answer(db_session: Session) -> None:
    item_id = seed_item(db_session, "swe")
    model = ScriptedChatModel(script=[AIMessage(content="It is a 41-task benchmark [S1].")])
    api = client(db_session, NoQueue(), models=[ChatModel("openrouter", model)])

    response = api.post("/ask", json={"question": "What does it measure?", "item_id": str(item_id)})

    assert response.status_code == 200
    # Over HTTP too, the model sees the item before any tool runs, and the answer cites it as S1.
    asked = "\n".join(str(message.content) for message in model.seen[0])
    assert "The reader is asking about S1." in asked
    assert TITLE in asked
    frames = [block for block in response.text.split("\n\n") if block.startswith("event: answer")]
    [answer] = [json.loads(frame.split("data: ", 1)[1]) for frame in frames]
    assert answer["supported"] is True
    assert [(c["id"], c["item_id"]) for c in answer["citations"]] == [("S1", str(item_id))]


def test_asking_about_a_non_admitted_item_is_422(db_session: Session) -> None:
    api = client(db_session, NoQueue())

    response = api.post("/ask", json={"question": "What does it measure?", "item_id": str(uuid4())})

    assert response.status_code == 422
    # The same refusal shape the ask route's other 422s use, so the page can point at the field.
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "item_id"]


def checked_summary() -> Any:
    # One summary as the queue writes it: a scripted answer that passes every check.
    writer = SummaryWriter(
        [ChatModel("openrouter", ScriptedChatModel(script=[AIMessage(json.dumps(_raw()))]))],
        lambda: CallGuard(NoChatCalls(), Ledger(10_000), 10, lambda: NOW),  # type: ignore[arg-type]
    )
    return writer.write(title=TITLE, text=TEXT)


def test_the_card_label_measures_momentum_from_the_observing_run(db_session: Session) -> None:
    # The snapshot is older than the run: snapshots are written only on change, so the label must
    # count from the run that last read the listing, not from when the count last moved.
    queue = NoQueue()
    item_id = seed_item(
        db_session,
        "swe",
        tier=Tier.VETTED_PREPRINT,
        basis=AdmissionBasis.HF_UPVOTES,
        hf_upvotes=160,
    )
    repo = Repository(db_session)
    run_at = NOW - timedelta(days=3)
    run_id = repo.start_run(run_at, run_at + timedelta(minutes=3), at=run_at)
    repo.finish_run(run_id, "succeeded", {"hf_listing_stored": 1}, 0, at=run_at)
    db_session.add_all(
        [
            SignalSnapshot(
                item_id=item_id,
                observed_at=NOW - timedelta(days=10),
                hf_upvotes=100,
                citation_count=0,
            ),
            SignalSnapshot(
                item_id=item_id,
                observed_at=NOW - timedelta(days=1),
                hf_upvotes=160,
                citation_count=0,
            ),
        ]
    )
    db_session.commit()

    body = client(db_session, queue).get("/suggestions").json()

    card = next(entry for entry in body["picks"] if entry["item_id"] == str(item_id))
    assert card["momentum_label"] == "+60 upvotes in 3 days"


def test_the_item_that_hit_the_cap_is_summarised_the_next_day(db_session: Session) -> None:
    item_id = seed_item(db_session, "swe")
    clock = SimpleClock(NOW)
    refusal = Refusal(True)
    summary_queue = queue_for(
        db_session, [AIMessage(content=json.dumps(_raw()))], refuse=refusal, clock=clock
    )

    summary_queue.submit([item_id])
    summary_queue.join(timeout=10)
    assert db_session.get(ItemSummary, item_id) is None

    # The cap resets at UTC midnight; the same queue must pick the item up again.
    refusal.refuse = False
    clock.now = NOW + timedelta(days=1)
    summary_queue.submit([item_id])
    summary_queue.join(timeout=10)

    row = db_session.get(ItemSummary, item_id)
    assert row is not None
    assert row.status == "checked"


def test_one_provider_outage_pauses_the_whole_queue(db_session: Session) -> None:
    # An outage is not the item's fault: every visible card would fail the same way, so the queue
    # pauses once instead of spending one doomed call per card.
    first = seed_item(db_session, "swe")
    second = seed_item(db_session, "survey", title=SURVEY_TITLE, text=SURVEY)

    class Down:
        def __init__(self) -> None:
            self.calls = 0

        def write(self, *, title: str, text: str) -> object:
            self.calls += 1
            raise ModelUnavailable("no provider answered")

    writer = Down()
    summary_queue = SummaryQueue(
        lambda: writer,  # type: ignore[arg-type]
        lambda: db_session,
        lambda: NOW,
        lambda: "test-model",
    )

    summary_queue.submit([first, second])
    summary_queue.join(timeout=10)

    assert writer.calls == 1


def test_a_summary_over_its_budget_is_stored_as_the_excerpt(db_session: Session) -> None:
    # The budget is per summary, so an item that overruns it once overruns it on every attempt:
    # storing the excerpt keeps each page refresh from spending another call on it.
    item_id = seed_item(db_session, "swe")
    model = ScriptedChatModel(
        script=[
            AIMessage(
                content=json.dumps(_raw()),
                usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
            )
        ]
    )
    writer = SummaryWriter(
        [ChatModel("openrouter", model)],
        lambda: CallGuard(Store(Refusal(False)), Ledger(0), 10, lambda: NOW),  # type: ignore[arg-type]
    )
    summary_queue = SummaryQueue(
        lambda: writer, lambda: db_session, lambda: NOW, lambda: "test-model"
    )

    summary_queue.submit([item_id])
    summary_queue.join(timeout=10)
    summary_queue.submit([item_id])
    summary_queue.join(timeout=10)

    row = db_session.get_one(ItemSummary, item_id)
    assert row.status == "excerpt"
    assert row.text_hash == text_hash(TITLE, TEXT)
    assert len(model.seen) == 1


class RecordingQueue:
    """Stands in for the queue while a test reads what the route decided to submit."""

    model_id = "test-model"

    def __init__(self) -> None:
        self.submitted: list[UUID] = []

    def submit(self, item_ids: Any) -> None:
        self.submitted.extend(item_ids)


class SimpleClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_only_a_missing_or_stale_summary_is_queued_and_served_as_pending(
    db_session: Session,
) -> None:
    queue = RecordingQueue()
    fresh_id = seed_item(db_session, "swe")
    stale_id = seed_item(db_session, "survey", title=SURVEY_TITLE, text=SURVEY)
    repo = Repository(db_session)
    repo.upsert_summary(
        fresh_id,
        checked_summary(),
        text_hash=text_hash(TITLE, TEXT),
        model="test-model",
        at=NOW,
    )
    repo.upsert_summary(
        stale_id,
        checked_summary(),
        text_hash="an-old-abstract",
        model="test-model",
        at=NOW,
    )
    db_session.commit()

    body = client(db_session, queue).get("/suggestions").json()

    cards = {card["item_id"]: card for card in (*body["picks"], *body["rising"])}
    assert cards[str(fresh_id)]["summary"]["status"] == "checked"
    # The stale summary describes text that has been replaced: the card says one is on its way.
    assert cards[str(stale_id)]["summary"]["status"] == "pending"
    assert queue.submitted == [stale_id]


def test_asking_about_a_long_lab_post_clips_the_source_block(db_session: Session) -> None:
    # A lab post stores the whole article. The block must be the tools' own: clipped to the
    # get_item bound and carrying the alias and reliability attributes, or one answer spends a
    # quarter of the per-answer budget on round one.
    long_text = ("We report a long lab write-up with 41 tasks. " * 500)[:20000]
    item_id = seed_item(
        db_session,
        "long-post",
        title="A long lab write-up",
        text=long_text,
        tier=Tier.LAB_BLOG,
        basis=AdmissionBasis.FEED,
    )
    db_session.commit()
    model = ScriptedChatModel(script=[AIMessage(content="It describes a long write-up [S1].")])
    deps = AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        embedder=FakeEmbedder(),
        models=[ChatModel("openrouter", model)],
        settings=Settings(),
        clock=lambda: NOW,
        chat_calls=NoChatCalls(),  # type: ignore[arg-type]
        profile_names=frozenset(),
        verifier=HmacVerifier(TEST_SECRET, time.time),
        allowed_hosts=LOOPBACK_HOSTS,
        limiter=RequestLimiter(
            per_minute=REQUESTS_PER_MINUTE,
            per_day=REQUESTS_PER_DAY,
            concurrent=CONCURRENT_ANSWERS,
            clock=lambda: NOW,
        ),
        search_mode=FixedMode(),  # type: ignore[arg-type]
        suggestions=service(db_session, NoQueue()),
    )

    list(run_question(deps, "What does this describe?", Filters(), item_id=item_id))

    asked = "\n".join(str(message.content) for message in model.seen[0])
    assert 'id="S1"' in asked
    assert 'tier="lab_blog"' in asked
    assert long_text[:GET_ITEM_MAX_CHARS] in asked
    # The unclipped body is not there: the block stops at the tool's own bound.
    assert long_text[: GET_ITEM_MAX_CHARS + 500] not in asked


def test_an_outage_pauses_submits_for_thirty_minutes_then_summarises_both(
    db_session: Session,
) -> None:
    first = seed_item(db_session, "swe")
    second = seed_item(db_session, "survey", title=SURVEY_TITLE, text=SURVEY)
    clock = SimpleClock(NOW)
    summary = checked_summary()

    class Down:
        def __init__(self) -> None:
            self.calls = 0
            self.down = True

        def write(self, *, title: str, text: str) -> Any:
            self.calls += 1
            if self.down:
                raise ModelUnavailable("no provider answered")
            return summary

    writer = Down()
    summary_queue = SummaryQueue(
        lambda: writer,  # type: ignore[arg-type]
        lambda: db_session,
        clock,
        lambda: "test-model",
    )
    summary_queue.submit([first, second])
    summary_queue.join(timeout=10)
    assert writer.calls == 1

    # Ten minutes later the provider may be back, but the queue is still resting.
    writer.down = False
    clock.now = NOW + timedelta(minutes=10)
    summary_queue.submit([first, second])
    summary_queue.join(timeout=10)
    assert writer.calls == 1

    # After the pause both cards are queued again: neither was left in the pending set.
    clock.now = NOW + timedelta(minutes=31)
    summary_queue.submit([first, second])
    summary_queue.join(timeout=10)
    assert writer.calls == 3
    assert db_session.get(ItemSummary, first) is not None
    assert db_session.get(ItemSummary, second) is not None


def test_the_cards_held_behind_the_cap_are_summarised_the_next_day(db_session: Session) -> None:
    first = seed_item(db_session, "swe")
    second = seed_item(db_session, "survey", title=SURVEY_TITLE, text=SURVEY)
    clock = SimpleClock(NOW)
    refusal = Refusal(True)
    script = [AIMessage(content=json.dumps(_raw())) for _ in range(4)]
    summary_queue = queue_for(db_session, script, refuse=refusal, clock=clock)

    summary_queue.submit([first, second])
    summary_queue.join(timeout=10)
    assert db_session.get(ItemSummary, first) is None

    refusal.refuse = False
    clock.now = NOW + timedelta(days=1)
    summary_queue.submit([first, second])
    summary_queue.join(timeout=10)

    assert db_session.get(ItemSummary, first) is not None
    assert db_session.get(ItemSummary, second) is not None
