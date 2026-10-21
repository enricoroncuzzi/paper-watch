"""Shared helpers for the QA-eval tests: a recorded row, a full recording, and CLI paths."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.qa_eval import QA_SET_VERSION, Recorded, RecordedSource
from paperwatch import cli
from paperwatch.agent.prompts import prompt_sha

QA_ANSWER = "A coding harness edits files and runs tests [S1]."


def qa_settings(model: str = "m/x:free") -> SimpleNamespace:
    # The command only reads chat_model on the live path; a real Settings would read .env.
    return SimpleNamespace(chat_model=model, log_level="INFO")


def qa_question_ids() -> list[str]:
    return [
        json.loads(line)["id"]
        for line in (cli._EVAL_DATA / "qa_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]


def qa_row(
    qid: str = "q01",
    external_id: str = "ag-01",
    *,
    provider_error: bool = False,
    model: str = "m/x:free",
    sha: str | None = None,
) -> Recorded:
    return Recorded(
        qid,
        model,
        QA_SET_VERSION,
        QA_ANSWER,
        (RecordedSource("S1", external_id, "peer_reviewed"),),
        ("search_corpus",),
        60,
        provider_error,
        prompt_sha() if sha is None else sha,
    )


def qa_recording() -> list[Recorded]:
    return [qa_row(qid) for qid in qa_question_ids()]


def qa_cli_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(cli, "get_settings", lambda: qa_settings())
    # The replay belongs to the committed default; pin it so the file name and the rows agree.
    monkeypatch.setattr(cli, "CHAT_MODEL", "m/x:free")
    monkeypatch.setattr(cli, "_QA_RECORDED_DIR", tmp_path)
    monkeypatch.setattr(cli, "_QA_BASELINE_PATH", tmp_path / "baseline.json")
    return tmp_path
