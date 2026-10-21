"""The QA recording path: every question answered once, and the tools' sources kept per answer."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from evals.qa_eval import QA_SET_VERSION, record
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
