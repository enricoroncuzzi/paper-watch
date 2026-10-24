"""Threads: a finished answer is saved with its sources, and a follow-up reuses both.

Every answer here is driven by a scripted chat model over the real test database, so the tests pin
the wiring (which turns reach the model, which aliases a turn may cite, what is stored) without a
key or the network. The thread write commits on its own session, outside the test's transaction,
so each test deletes the rows it left behind.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from structlog.testing import capture_logs

from paperwatch.api.app import AgentDeps, create_app, run_question
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
from paperwatch.llm.chat import ChatModel, ScriptedChatModel
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval.hybrid import Filters
from paperwatch.store.db import make_session_factory
from paperwatch.store.models import Thread, ThreadTurn
from paperwatch.store.repository import Repository
from tests.conftest import AGENT_ROWS, SeedRow

pytestmark = pytest.mark.integration
NOW = datetime(2026, 11, 4, 9, tzinfo=UTC)
TEST_SECRET = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret
ANSWER = "SWE-bench Verified measures coding agents on real repository bugs [S1]."
FOLLOW_UP = "And what did it measure exactly?"
Seed = Callable[[list[SeedRow]], dict[str, UUID]]


class FixedMode:
    """A search-mode gate with one answer, so a test needs no run record and no /proc."""

    def current(self) -> ModeDecision:
        return ModeDecision("hybrid", "ok")


class MemoryChatCalls:
    """Usage rows in memory: the answer session is read-only, so the store cannot use it."""

    def reserve_chat_call(self, *, at: datetime, provider: str, cap: int) -> tuple[int, int]:
        return 1, 1

    def fill_chat_call(
        self, call_id: int, *, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        return None

    def fail_chat_call(self, call_id: int) -> None:
        return None


@pytest.fixture
def write_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    """A writable factory, as production has: the answer session is read-only by design."""
    factory = make_session_factory(engine)
    yield factory
    # The thread write commits on its own connection, so the test's rollback cannot remove it.
    with factory() as session:
        session.execute(delete(ThreadTurn))
        session.execute(delete(Thread))
        session.commit()


def deps(
    db_session: Session,
    models: list[ChatModel],
    write_factory: sessionmaker[Session] | None,
    **settings: object,
) -> AgentDeps:
    return AgentDeps(
        session_factory=lambda: db_session,  # type: ignore[arg-type,return-value]
        write_session_factory=write_factory,
        embedder=FakeEmbedder(),
        models=models,
        settings=Settings(_env_file=None, **settings),  # type: ignore[arg-type]
        clock=lambda: NOW,
        chat_calls=MemoryChatCalls(),  # type: ignore[arg-type]
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
    )


def client(app_deps: AgentDeps) -> TestClient:
    return TestClient(create_app(app_deps), base_url="http://localhost")


def ask(app_deps: AgentDeps, payload: dict[str, Any]) -> Any:
    token = mint_hmac_token(TEST_SECRET, "test", time.time())
    return client(app_deps).post("/ask", json=payload, headers={"Authorization": f"Bearer {token}"})


def get(app_deps: AgentDeps, path: str) -> Any:
    token = mint_hmac_token(TEST_SECRET, "test", time.time())
    return client(app_deps).get(path, headers={"Authorization": f"Bearer {token}"})


def events(body: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def scripted(*steps: AIMessage | Exception) -> list[ChatModel]:
    return [ChatModel("openrouter", ScriptedChatModel(script=list(steps)))]


def call(name: str, n: int = 1, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{n}"}])


def final(text: str) -> AIMessage:
    return AIMessage(text)


def stored_turn(factory: sessionmaker[Session], thread_id: UUID) -> list[Any]:
    with factory() as session:
        row = Repository(session).thread(thread_id)
    assert row is not None
    return list(row.turns)


def test_a_first_answer_creates_a_thread_and_sends_its_id(
    db_session: Session, write_factory: sessionmaker[Session], seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    models = scripted(call("search_corpus", query="SWE-bench coding agents"), final(ANSWER))
    frames = events(
        ask(
            deps(db_session, models, write_factory),
            {"question": "how are coding agents evaluated?"},
        ).text
    )
    assert [name for name, _ in frames][-2:] == ["answer", "thread"]
    thread_id = UUID(frames[-1][1]["id"])
    assert frames[-1][1]["turn"] == 1
    with write_factory() as session:
        row = Repository(session).thread(thread_id)
    assert row is not None
    assert row.title == "how are coding agents evaluated?"
    assert [(turn.n, turn.question) for turn in row.turns] == [
        (1, "how are coding agents evaluated?")
    ]


def test_a_follow_up_sends_the_earlier_turns_to_the_model(
    db_session: Session, write_factory: sessionmaker[Session], seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench coding agents"),
            final(ANSWER),
            final("It measured coding agents on real repository bugs [S1]."),
        ]
    )
    app_deps = deps(db_session, [ChatModel("openrouter", model)], write_factory)
    first = events(ask(app_deps, {"question": "how are coding agents evaluated?"}).text)
    thread_id = first[-1][1]["id"]
    second = events(ask(app_deps, {"question": FOLLOW_UP, "thread_id": thread_id}).text)
    assert second[-1][1] == {"id": thread_id, "turn": 2}
    # The follow-up's only model call carries turn 1's question and checked answer.
    contents = [str(message.content) for message in model.seen[2]]
    assert "how are coding agents evaluated?" in contents
    assert ANSWER in contents


def test_aliases_stay_stable_across_turns(
    db_session: Session, write_factory: sessionmaker[Session], seed_items: Seed
) -> None:
    ids = seed_items(AGENT_ROWS)
    model = ScriptedChatModel(
        script=[
            call("search_corpus", query="SWE-bench coding agents"),
            final(ANSWER),
            # S1 is from turn 1: the tool and the checker both read the thread's registry.
            call("get_item", 2, source_id="S1"),
            final("It measured coding agents on real repository bugs [S1]."),
        ]
    )
    app_deps = deps(db_session, [ChatModel("openrouter", model)], write_factory)
    thread_id = events(ask(app_deps, {"question": "q1"}).text)[-1][1]["id"]
    second = events(ask(app_deps, {"question": FOLLOW_UP, "thread_id": thread_id}).text)
    answer = dict(second)["answer"]
    assert answer["supported"] and answer["citations"][0]["id"] == "S1"
    assert answer["citations"][0]["item_id"] == str(ids["swe"])
    turns = stored_turn(write_factory, UUID(thread_id))
    assert [turn.answer["citations"][0]["id"] for turn in turns] == ["S1", "S1"]


def test_a_finished_answer_can_be_fetched_by_thread(
    db_session: Session, write_factory: sessionmaker[Session], seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    models = scripted(
        call("search_corpus", query="SWE-bench coding agents"),
        final(ANSWER),
        final("It measured coding agents on real repository bugs [S1]."),
    )
    app_deps = deps(db_session, models, write_factory)
    first = events(ask(app_deps, {"question": "q1"}).text)
    thread_id = first[-1][1]["id"]
    ask(app_deps, {"question": FOLLOW_UP, "thread_id": thread_id})
    detail = get(app_deps, f"/threads/{thread_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == thread_id
    assert [(turn["n"], turn["question"]) for turn in body["turns"]] == [
        (1, "q1"),
        (2, FOLLOW_UP),
    ]
    assert body["turns"][0]["answer"]["text"] == ANSWER
    assert body["turns"][0]["search_mode"] == "hybrid"
    # The stored turn is exactly what the stream sent, so a re-fetch shows what the reader saw.
    assert body["turns"][0]["answer"] == dict(first)["answer"]
    listed = get(app_deps, "/threads").json()["threads"]
    assert [(row["id"], row["turns"]) for row in listed] == [(thread_id, 2)]
    assert listed[0]["title"] == "q1"


def test_an_aborted_answer_stores_nothing(
    db_session: Session, write_factory: sessionmaker[Session]
) -> None:
    cancel = threading.Event()
    cancel.set()
    received = list(
        run_question(
            deps(db_session, scripted(final(ANSWER)), write_factory), "q", Filters(), cancel
        )
    )
    assert received == []
    with write_factory() as session:
        assert session.scalar(select(func.count()).select_from(Thread)) == 0
        assert session.scalar(select(func.count()).select_from(ThreadTurn)) == 0


def test_an_unknown_thread_is_404_before_any_model_call(
    db_session: Session, write_factory: sessionmaker[Session]
) -> None:
    model = ScriptedChatModel(script=[final(ANSWER)])
    response = ask(
        deps(db_session, [ChatModel("openrouter", model)], write_factory),
        {"question": "q", "thread_id": str(uuid4())},
    )
    assert response.status_code == 404
    assert model.seen == []


def test_history_is_bounded_to_six_turns(
    db_session: Session, write_factory: sessionmaker[Session]
) -> None:
    with write_factory() as session:
        repo = Repository(session)
        thread_id, _ = repo.save_thread_turn(
            thread_id=None,
            title="question 1",
            registry=[],
            question="question 1",
            answer={"text": "answer 1"},
            search_mode="hybrid",
            at=NOW,
        )
        for n in range(2, 9):
            repo.save_thread_turn(
                thread_id=thread_id,
                title="",
                registry=[],
                question=f"question {n}",
                answer={"text": f"answer {n}"},
                search_mode="hybrid",
                at=NOW,
            )
        session.commit()
    model = ScriptedChatModel(script=[final("An answer with no citation.")])
    list(
        run_question(
            deps(db_session, [ChatModel("openrouter", model)], write_factory),
            "the follow-up",
            Filters(),
            thread_id=thread_id,
        )
    )
    contents = [str(message.content) for message in model.seen[0]]
    assert "question 2" not in contents
    assert "question 3" in contents and "answer 3" in contents
    assert "answer 8" in contents


def test_ask_about_an_item_starts_a_thread_with_it_as_s1(
    db_session: Session, write_factory: sessionmaker[Session], seed_items: Seed
) -> None:
    ids = seed_items(AGENT_ROWS)
    # The item read opens and closes its own session, so the seed must be out of this test
    # session's savepoint for the read to see it (as the suggestions tests do).
    db_session.commit()
    models = scripted(final(ANSWER))
    response = ask(
        deps(db_session, models, write_factory),
        {"question": "what does this paper measure?", "item_id": str(ids["swe"])},
    )
    frames = events(response.text)
    assert dict(frames)["answer"]["citations"][0]["id"] == "S1"
    thread_id = UUID(frames[-1][1]["id"])
    with write_factory() as session:
        row = Repository(session).thread(thread_id)
    assert row is not None
    assert [ref["alias"] for ref in row.registry] == ["S1"]
    assert row.registry[0]["item_id"] == str(ids["swe"])


def test_a_failed_turn_write_still_delivers_the_answer(
    db_session: Session, seed_items: Seed
) -> None:
    seed_items(AGENT_ROWS)
    models = scripted(call("search_corpus", query="SWE-bench coding agents"), final(ANSWER))

    def broken() -> Session:
        raise RuntimeError("threads is unwritable")

    with capture_logs() as logs:
        frames = events(ask(deps(db_session, models, broken), {"question": "q"}).text)  # type: ignore[arg-type]
    assert frames[-1][0] == "answer"
    assert "thread" not in {name for name, _ in frames}
    [failure] = [entry for entry in logs if entry["event"] == "thread_save_failed"]
    assert failure["error_type"] == "RuntimeError"
    assert "unwritable" not in str(logs)


def test_a_caller_without_a_write_factory_saves_no_thread(
    db_session: Session, seed_items: Seed
) -> None:
    # A caller that only reads (a test, the contract build) opts out instead of failing a write.
    seed_items(AGENT_ROWS)
    models = scripted(call("search_corpus", query="SWE-bench coding agents"), final(ANSWER))
    frames = events(ask(deps(db_session, models, None), {"question": "q"}).text)
    assert "thread" not in dict(frames)


def test_the_thread_event_is_the_last_frame_and_only_after_a_finished_answer(
    db_session: Session, write_factory: sessionmaker[Session], seed_items: Seed
) -> None:
    # A turn is stored only when the answer finished: a model failure ends the stream with an
    # error event and no thread event, so a client never re-fetches a turn that is not there.
    seed_items(AGENT_ROWS)
    models = scripted(RuntimeError("provider is down"))
    frames = events(ask(deps(db_session, models, write_factory), {"question": "q"}).text)
    assert [name for name, _ in frames] == ["error"]
    with write_factory() as session:
        assert session.scalar(select(func.count()).select_from(ThreadTurn)) == 0


def test_the_thread_list_is_newest_first(
    db_session: Session, write_factory: sessionmaker[Session]
) -> None:
    with write_factory() as session:
        repo = Repository(session)
        older, _ = repo.save_thread_turn(
            thread_id=None,
            title="older",
            registry=[],
            question="older",
            answer={"text": "answer"},
            search_mode="hybrid",
            at=NOW,
        )
        newer, _ = repo.save_thread_turn(
            thread_id=None,
            title="newer",
            registry=[],
            question="newer",
            answer={"text": "answer"},
            search_mode="hybrid",
            at=NOW + timedelta(hours=1),
        )
        session.commit()
    listed = get(deps(db_session, [], write_factory), "/threads").json()["threads"]
    assert [row["title"] for row in listed] == ["newer", "older"]
    assert {row["id"] for row in listed} == {str(older), str(newer)}


def test_an_unknown_thread_detail_is_404(
    db_session: Session, write_factory: sessionmaker[Session]
) -> None:
    assert get(deps(db_session, [], write_factory), f"/threads/{uuid4()}").status_code == 404
