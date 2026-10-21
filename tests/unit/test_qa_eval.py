"""QA-eval scoring, the gate, and the frozen question set and recordings' shape."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from evals.qa_eval import (
    GATED,
    QA_SET_VERSION,
    Recorded,
    RecordedSource,
    gate,
    load_recorded,
    score,
    write_recorded,
)

# The repo-root-relative path is resolved from this file, like the retrieval eval's data paths,
# so a run from another working directory still finds the committed set.
DATA = Path(__file__).resolve().parents[2] / "evals" / "data"


def questions(tmp_path: Path) -> Path:
    path = tmp_path / "q.jsonl"
    rows = [
        {"id": "a1", "question": "?", "supporting": ["ag-01"], "kind": "answerable"},
        {"id": "a2", "question": "?", "supporting": ["ag-02"], "kind": "answerable"},
        {"id": "a3", "question": "?", "supporting": ["ag-03"], "kind": "answerable"},
        {"id": "u1", "question": "?", "supporting": [], "kind": "unanswerable"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def rec(qid: str, text: str, *sources: tuple[str, str]) -> Recorded:
    refs = tuple(RecordedSource(alias, eid, "vetted_preprint") for alias, eid in sources)
    return Recorded(qid, "m", QA_SET_VERSION, text, refs, ("search_corpus",), 100)


CLAIM = "The harness edits files and runs the tests automatically"


def test_score_counts_support_recall_and_precision(tmp_path: Path) -> None:
    recorded = [
        rec("a1", f"{CLAIM} [S1].", ("S1", "ag-01")),  # right item, clean
        rec("a2", f"{CLAIM} [S1].", ("S1", "ag-09")),  # wrong item
        rec("a3", f"{CLAIM}.", ("S1", "ag-03")),  # uncited: becomes NO_SUPPORT
        rec("u1", f"{CLAIM}."),  # abstains (no valid citation)
    ]
    m = score(recorded, questions(tmp_path))
    assert m["support_recall"] == pytest.approx(1 / 3)
    assert m["citation_precision"] == pytest.approx(1 / 2)
    assert m["clean_rate"] == pytest.approx(2 / 3)  # a3's NO_SUPPORT carries an uncited_claim flag
    assert m["abstention"] == 1.0
    assert m["mean_tools"] == 1.0 and m["mean_tokens"] == 100.0


def test_answering_an_unanswerable_question_is_not_abstention(tmp_path: Path) -> None:
    recorded = [rec("u1", f"{CLAIM} [S1].", ("S1", "ag-01"))]
    assert score(recorded, questions(tmp_path))["abstention"] == 0.0


def test_rescoring_uses_the_current_checker(tmp_path: Path) -> None:
    # S2 was never returned by a tool, so the checker rejects it and nothing supports a1.
    recorded = [rec("a1", f"{CLAIM} [S2].", ("S1", "ag-01"))]
    assert score(recorded, questions(tmp_path))["support_recall"] == 0.0


def test_gate_refuses_mismatched_keys_and_non_finite_values() -> None:
    base = dict.fromkeys(GATED, 0.5)
    assert gate(base, base) == []
    assert gate({**base, "abstention": 0.4}, base) != []
    with pytest.raises(ValueError):
        gate(base, {**base, "extra": 1.0})
    with pytest.raises(ValueError):
        gate({**base, "clean_rate": math.nan}, base)


def test_gate_allows_the_reported_only_metrics() -> None:
    # mean_tools and mean_tokens are printed but never gated, so extra metric keys are allowed.
    base = dict.fromkeys(GATED, 0.5)
    assert gate({**base, "mean_tools": 3.0}, base) == []


def test_provider_error_is_not_a_clean_answer_or_an_abstention(tmp_path: Path) -> None:
    # A question the provider never answered is a failure, not an answer the model chose to give.
    recorded = [
        replace(rec("a1", f"{CLAIM} [S1].", ("S1", "ag-01")), provider_error=True),
        replace(rec("u1", ""), provider_error=True),
    ]
    m = score(recorded, questions(tmp_path))
    assert m["clean_rate"] == 0.0
    assert m["abstention"] == 0.0
    assert m["provider_errors"] == 2.0


def test_score_refuses_a_duplicated_question(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="more than once"):
        score([rec("a1", CLAIM), rec("a1", CLAIM)], questions(tmp_path))


def test_score_refuses_an_empty_recording(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        score([], questions(tmp_path))


def test_recorded_rows_round_trip_through_jsonl(tmp_path: Path) -> None:
    rows = [rec("a1", f"{CLAIM} [S1].", ("S1", "ag-01"))]
    path = tmp_path / "recording.jsonl"
    write_recorded(path, rows)
    assert load_recorded(path) == rows


def test_question_file_is_well_formed() -> None:
    rows = [
        json.loads(line)
        for line in (DATA / "qa_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    corpus = {
        json.loads(line)["external_id"]
        for line in (DATA / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert len(rows) == 20 and len({r["id"] for r in rows}) == 20
    answerable = [r for r in rows if r["kind"] == "answerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    assert len(answerable) == 16 and len(unanswerable) == 4
    assert all(r["supporting"] and set(r["supporting"]) <= corpus for r in answerable)
    assert all(r["supporting"] == [] for r in unanswerable)
    assert sum(len(r["supporting"]) >= 2 for r in answerable) >= 3


def test_recordings_match_the_question_set() -> None:
    ids = {
        json.loads(line)["id"]
        for line in (DATA / "qa_questions.jsonl").read_text().splitlines()
        if line.strip()
    }
    files = sorted((DATA / "qa_recorded").glob("*.jsonl"))
    assert files
    for path in files:
        rows = load_recorded(path)
        assert {r.question_id for r in rows} == ids
        assert {r.set_version for r in rows} == {QA_SET_VERSION}
