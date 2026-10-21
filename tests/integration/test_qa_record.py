"""The QA recording path: every question answered once, and the tools' sources kept per answer."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from evals.qa_eval import QA_SET_VERSION, record
from paperwatch.agent.prompts import prompt_sha
from paperwatch.llm.chat import ChatModel, ScriptedChatModel

pytestmark = pytest.mark.integration
DATA = Path(__file__).resolve().parents[2] / "evals" / "data"
USAGE = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
ANSWER = "A coding harness edits files and runs tests [S1]."


def question_ids() -> list[str]:
    import json

    return [
        json.loads(line)["id"]
        for line in (DATA / "qa_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]


def search_then_answer() -> list[AIMessage]:
    return [
        AIMessage(
            "",
            tool_calls=[{"name": "search_corpus", "args": {"query": "coding harness"}, "id": "c1"}],
            usage_metadata=USAGE,
        ),
        AIMessage(ANSWER, usage_metadata=USAGE),
    ]


def test_record_covers_every_question_with_its_sources(db_session: Session) -> None:
    model = ScriptedChatModel(script=search_then_answer() * len(question_ids()))
    rows = record(
        db_session,
        [ChatModel("scripted", model)],
        DATA / "qa_questions.jsonl",
        DATA / "corpus.jsonl",
    )
    assert [row.question_id for row in rows] == question_ids()
    assert {row.set_version for row in rows} == {QA_SET_VERSION}
    assert {row.model for row in rows} == {"scripted"}
    for row in rows:
        # Each question gets its own registry, so every row carries the sources its own search
        # returned and the tool it called, not the previous question's.
        assert row.sources and row.sources[0].alias == "S1"
        assert row.sources[0].external_id == "ag-01"
        assert row.tools == ("search_corpus",)
        # Two model calls (the search round and the answer round), 60 usage tokens each.
        assert row.tokens == 120
        assert not row.provider_error


def test_record_marks_a_provider_failure_instead_of_dropping_the_question(
    db_session: Session,
) -> None:
    # One failing candidate per question: the graph raises ModelUnavailable, record keeps the row.
    model = ScriptedChatModel(script=[RuntimeError("provider down") for _ in question_ids()])
    rows = record(
        db_session,
        [ChatModel("scripted", model)],
        DATA / "qa_questions.jsonl",
        DATA / "corpus.jsonl",
    )
    assert [row.question_id for row in rows] == question_ids()
    assert all(row.provider_error for row in rows)
    assert all(row.answer_text == "" and row.sources == () for row in rows)


def test_each_question_starts_a_fresh_registry(db_session: Session) -> None:
    def script(query: str) -> list[AIMessage]:
        call = {"name": "search_corpus", "args": {"query": query}, "id": "c1"}
        return [
            AIMessage("", tool_calls=[call], usage_metadata=USAGE),
            AIMessage(ANSWER, usage_metadata=USAGE),
        ]

    # Even questions search the harness paper, odd ones the ToolBench paper: a registry shared
    # across questions would keep the first question's S1 for all of them.
    steps = [
        message
        for i in range(len(question_ids()))
        for message in script(["coding harness", "ToolBench"][i % 2])
    ]
    rows = record(
        db_session,
        [ChatModel("scripted", ScriptedChatModel(script=steps))],
        DATA / "qa_questions.jsonl",
        DATA / "corpus.jsonl",
    )
    assert [row.sources[0].external_id for row in rows[:2]] == ["ag-01", "ag-02"]


def test_record_stores_the_models_own_text(db_session: Session) -> None:
    # The checker would mask the invented S9 as [?]; the recording keeps what the model wrote, so
    # a later checker change is measured against the original text, not an old verdict.
    invented = "A coding harness edits files and runs tests [S1, S9]."
    steps: list[AIMessage] = []
    for _ in question_ids():
        steps += [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "search_corpus", "args": {"query": "coding harness"}, "id": "c1"}
                ],
                usage_metadata=USAGE,
            ),
            AIMessage(invented, usage_metadata=USAGE),
        ]
    rows = record(
        db_session,
        [ChatModel("scripted", ScriptedChatModel(script=steps))],
        DATA / "qa_questions.jsonl",
        DATA / "corpus.jsonl",
    )
    assert rows[0].answer_text == invented
    assert "S9" in rows[0].answer_text
    assert rows[0].prompt_sha == prompt_sha()


def test_a_provider_failure_row_carries_the_prompt_fingerprint(db_session: Session) -> None:
    # Without it the replay would call a recording with a tolerated provider error stale.
    model = ScriptedChatModel(script=[RuntimeError("down") for _ in question_ids()])
    rows = record(
        db_session,
        [ChatModel("scripted", model)],
        DATA / "qa_questions.jsonl",
        DATA / "corpus.jsonl",
    )
    assert {row.prompt_sha for row in rows} == {prompt_sha()}
