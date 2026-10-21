"""Proposed additions for Task 18: each test pins a mutant that survived the committed suite."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import evals.qa_eval as evals_qa
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
from paperwatch import cli
from paperwatch.agent.prompts import prompt_sha
from paperwatch.cli import app
from tests.unit.test_qa_eval import CLAIM, questions, rec


def test_one_supporting_item_is_enough_for_a_two_item_question(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps(
            {"id": "t1", "question": "?", "supporting": ["ag-01", "ag-02"], "kind": "answerable"}
        )
    )
    assert score([rec("t1", f"{CLAIM} [S1].", ("S1", "ag-01"))], path)["support_recall"] == 1.0


def test_precision_is_one_without_citations_and_ignores_unanswerable(tmp_path: Path) -> None:
    assert score([rec("a1", "")], questions(tmp_path))["citation_precision"] == 1.0
    # A citation on an unanswerable question counts against abstention, not against precision.
    m = score(
        [
            rec("a1", f"{CLAIM} [S1].", ("S1", "ag-01")),
            rec("u1", f"{CLAIM} [S1].", ("S1", "ag-09")),
        ],
        questions(tmp_path),
    )
    assert m["citation_precision"] == 1.0


def test_an_empty_non_error_answer_is_not_clean(tmp_path: Path) -> None:
    assert score([rec("a1", "")], questions(tmp_path))["clean_rate"] == 0.0


def test_gate_tolerance_is_zero() -> None:
    base = dict.fromkeys(GATED, 0.5)
    assert gate({**base, "support_recall": 0.5 - 1e-6}, base) != []


def test_round_trip_keeps_every_field(tmp_path: Path) -> None:
    row = Recorded(
        "q02",
        "m/x:free",
        QA_SET_VERSION + 1,
        "t",
        (
            RecordedSource("S1", "ag-01", "peer_reviewed"),
            RecordedSource("S2", "ag-02", "lab_blog"),
        ),
        ("search_corpus", "get_item"),
        7,
        True,
        prompt_sha(),
    )
    path = tmp_path / "r.jsonl"
    write_recorded(path, [row, replace(row, question_id="q03")])
    assert load_recorded(path) == [row, replace(row, question_id="q03")]


def test_a_tier_claim_is_checked_against_the_recorded_tier(tmp_path: Path) -> None:
    text = "The method was published at a peer-reviewed venue and the harness runs tests [S1]."
    row = rec("a1", text, ("S1", "ag-01"))
    assert score([row], questions(tmp_path))["clean_rate"] == 0.0


def test_score_refuses_a_question_the_set_does_not_hold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not hold"):
        score([rec("zz", CLAIM)], questions(tmp_path))


def test_qa_models_never_include_a_fallback() -> None:
    settings = cli.Settings(openrouter_api_key="k", groq_api_key="g", groq_chat_model="m")  # type: ignore[arg-type]
    assert [m.provider for m in cli._qa_models(settings, "x/y:free")] == ["openrouter"]


def test_exactly_two_provider_errors_are_tolerated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit.test_cli import _qa_cli_paths, qa_row

    path = _qa_cli_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_qa_models", lambda settings, model: [object()])
    monkeypatch.setattr(cli, "_with_throwaway_database", lambda run: run(None))
    rows = [qa_row("q01"), qa_row("q02", provider_error=True), qa_row("q03", provider_error=True)]
    monkeypatch.setattr(evals_qa, "record", lambda session, models, q, c: rows)
    result = CliRunner().invoke(app, ["qa-eval", "--live", "--model", "m/x:free"])
    assert result.exit_code == 0 and (path / "m_x_free.jsonl").exists()


def test_model_without_live_says_why(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.unit.test_cli import _qa_cli_paths

    _qa_cli_paths(monkeypatch, tmp_path)
    result = CliRunner().invoke(app, ["qa-eval", "--model", "m/x:free"])
    assert "only meaningful with --live" in result.stderr


def test_the_default_model_is_the_best_committed_recording() -> None:
    # The plan's order: support_recall, citation_precision, abstention, then lower mean_tokens.
    from paperwatch.config import CHAT_MODEL

    data = Path(__file__).resolve().parents[2] / "evals" / "data"
    best = max(
        (
            m["support_recall"],
            m["citation_precision"],
            m["abstention"],
            -m["mean_tokens"],
            path.stem,
        )
        for path in (data / "qa_recorded").glob("*.jsonl")
        for m in [score(load_recorded(path), data / "qa_questions.jsonl")]
    )
    assert best[-1] == CHAT_MODEL.replace("/", "_").replace(":", "_")
