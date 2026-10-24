import dataclasses
import json
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import structlog
from structlog.testing import capture_logs
from typer.testing import CliRunner

import evals.qa_eval as evals_qa
import evals.retrieval_eval as retrieval_eval
from evals.retrieval_eval import GATED, DatabaseNotEmpty
from paperwatch import cli
from paperwatch.agent.citations import CheckedAnswer, Citation, Flag
from paperwatch.agent.graph import AnswerEvent, TokenEvent, ToolEvent
from paperwatch.api.app import ErrorEvent
from paperwatch.api.auth import HmacVerifier
from paperwatch.cli import app
from paperwatch.config import Settings
from paperwatch.domain import AdmissionBasis, Tier
from paperwatch.ingest.health import Health
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval.hybrid import Filters
from paperwatch.suggestions.ranking import Candidate
from tests.unit.qa_helpers import qa_cli_paths as _qa_cli_paths
from tests.unit.qa_helpers import qa_recording, qa_row


def boom(*args: object, **kwargs: object) -> object:
    raise AssertionError("must not be built")


def test_ingest_is_a_subcommand_and_requires_live() -> None:
    result = CliRunner().invoke(app, ["ingest"])
    assert result.exit_code == 2 and "pass --live" in result.output


def test_parse_since_accepts_naive_z_and_offset() -> None:
    # Click's parser rejects Z and offsets, so the command parses the text itself.
    assert cli._parse_since(None) is None
    assert cli._parse_since("2026-10-12T06:00:00") == datetime(2026, 10, 12, 6, tzinfo=UTC)
    assert cli._parse_since("2026-10-12T04:00:00Z") == datetime(2026, 10, 12, 4, tzinfo=UTC)
    assert cli._parse_since("2026-10-12T08:00:00+02:00") == datetime(2026, 10, 12, 6, tzinfo=UTC)
    # Aware datetimes compare equal across zones, so assert the offset is truly UTC.
    assert cli._parse_since("2026-10-12T08:00:00+02:00").utcoffset() == timedelta(0)


def test_invalid_since_exits_before_building_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(cli, "get_settings", lambda: called.append(True) or object())
    result = CliRunner().invoke(app, ["ingest", "--live", "--since", "garbage"])
    assert result.exit_code == 2 and "ISO 8601" in result.output
    assert called == []  # parsing happens before any model, client or engine is built


def test_live_ingest_builds_models_through_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # A live run must actually reach make_embedder and make_classifier through _models.
    built = {"embedder": False, "classifier": False}

    def fake_embedder(settings, ledger):
        built["embedder"] = True
        return object()

    def fake_classifier(settings, ledger, profiles, embedder):
        built["classifier"] = True
        return object()

    monkeypatch.setattr(cli, "make_embedder", fake_embedder)
    monkeypatch.setattr(cli, "make_classifier", fake_classifier)
    # Stub everything that would touch the network, the database, or the sources.
    monkeypatch.setattr(cli, "make_http_client", lambda settings: object())
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: object())
    monkeypatch.setattr(
        cli,
        "run_ingestion",
        # The pipeline builds the models from the factory it is given (it may release and rebuild
        # them between follow-up passes), so the stand-in calls it the way run_ingestion does.
        lambda session, sources, build_models, *args, **kwargs: (
            build_models(),
            SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ())),
        )[1],
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0
    assert built == {"embedder": True, "classifier": True}


def _stub_live_ingest(monkeypatch: pytest.MonkeyPatch, result: SimpleNamespace) -> None:
    """Stub every dependency of `ingest --live` so only the command's exit code is under test."""
    monkeypatch.setattr(cli, "make_embedder", lambda settings, ledger: object())
    monkeypatch.setattr(
        cli, "make_classifier", lambda settings, ledger, profiles, embedder: object()
    )
    monkeypatch.setattr(cli, "make_http_client", lambda settings: object())
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: object())
    monkeypatch.setattr(cli, "run_ingestion", lambda *args, **kwargs: result)


def test_ingest_prints_the_skip_and_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Another run holds the lock: the command is a successful no-op, not a failure.
    _stub_live_ingest(
        monkeypatch, SimpleNamespace(status="skipped", counts={}, tokens=0, health=None)
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0
    assert "skip" in result.output.lower()


@pytest.mark.parametrize("status", ["succeeded", "partial"])
def test_ingest_exits_three_and_prints_the_problems_when_unhealthy(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    # A finished run whose health verdict is unhealthy is neither a success nor a crash: the code
    # did its job and judged the day bad, so it gets its own exit code and names the problems.
    health = Health(False, ("arxiv fetched 0 items", "cursor not advanced: arxiv"))
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(status=status, counts={"fetched": 0}, tokens=0, health=health),
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 3
    assert "arxiv fetched 0 items" in result.stderr
    assert "cursor not advanced: arxiv" in result.stderr
    # The run's own line still reaches stdout, so a healthy summary is never swallowed.
    assert f"{status}:" in result.output


def test_ingest_exits_zero_when_a_finished_run_is_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(
            status="succeeded", counts={"fetched": 4}, tokens=0, health=Health(True, ())
        ),
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0 and result.stderr == ""


def test_ingest_prints_a_warning_and_still_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # One missed lab-feed day is a finding worth printing, not a bad morning: the next run retries
    # the same window, so the command says so and exits 0.
    warning = "feed missed once, retried next run: https://lab.example/feed.xml"
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(
            status="succeeded", counts={}, tokens=0, health=Health(True, (), (warning,))
        ),
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0
    assert f"warning: {warning}" in result.stderr
    assert "unhealthy:" not in result.stderr


def test_ingest_prints_warnings_and_problems_together(monkeypatch: pytest.MonkeyPatch) -> None:
    # A continuing outage has both: the problems decide the exit code, the warning still shows.
    warning = "feed missed once, retried next run: https://lab.example/feed.xml"
    health = Health(False, ("status=partial",), (warning,))
    _stub_live_ingest(
        monkeypatch, SimpleNamespace(status="partial", counts={}, tokens=0, health=health)
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 3
    assert f"warning: {warning}" in result.stderr and "unhealthy: status=partial" in result.stderr


def test_ingest_exits_three_when_a_finished_run_carries_no_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pipeline always writes a verdict, so this is defensive: a finished run whose verdict went
    # missing must fail closed rather than pass for a good morning.
    _stub_live_ingest(
        monkeypatch, SimpleNamespace(status="succeeded", counts={}, tokens=0, health=None)
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 3 and "no health verdict" in result.stderr


@pytest.mark.parametrize("status", ["failed", "budget_exhausted"])
def test_ingest_exits_one_when_the_run_did_not_finish_its_work(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    # These are not "finished but unhealthy": the run itself did not complete, so the exit code
    # stays 1 and nothing is mistaken for a day of data that needs a human's judgement.
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(
            status=status, counts={}, tokens=0, health=Health(False, (f"status={status}",))
        ),
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 1


def test_search_is_a_subcommand_and_requires_live(monkeypatch: pytest.MonkeyPatch) -> None:
    # No --live: the command must refuse before building the embedder or the engine.
    monkeypatch.setattr(cli, "make_embedder", boom)
    monkeypatch.setattr(cli, "make_engine", boom)
    result = CliRunner().invoke(app, ["search", "SWE-bench"])
    assert result.exit_code == 2 and "pass --live" in result.output


def test_search_rejects_bad_arguments_before_building_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bad argument must be a usage error (exit 2) raised before any model or connection.
    monkeypatch.setattr(cli, "make_embedder", boom)
    monkeypatch.setattr(cli, "make_engine", boom)
    blank = CliRunner().invoke(app, ["search", "   ", "--live"])
    assert blank.exit_code == 2 and "blank" in blank.output.lower()
    tier = CliRunner().invoke(app, ["search", "SWE-bench", "--live", "--tier", "bogus"])
    assert tier.exit_code == 2 and "tier" in tier.output.lower()
    k = CliRunner().invoke(app, ["search", "SWE-bench", "--live", "--k", "0"])
    assert k.exit_code == 2


def test_live_search_wires_the_filters_and_k_and_prints_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []
    captured: dict[str, object] = {}

    def fake_embedder(settings: object, ledger: object) -> object:
        built.append("embedder")
        return object()

    def fake_hybrid_search(*args: object, **kwargs: object) -> list[object]:
        captured.update(kwargs)
        return [
            SimpleNamespace(score=0.5, tier="vetted_preprint", title="A title", url="https://x/1")
        ]

    closed: list[bool] = []
    session = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(cli, "make_embedder", fake_embedder)
    monkeypatch.setattr(cli, "make_classifier", boom)  # search must never build it
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: session)
    monkeypatch.setattr(cli, "hybrid_search", fake_hybrid_search)
    result = CliRunner().invoke(
        cli.app,
        [
            "search",
            "SWE-bench",
            "--live",
            "--tier",
            "peer_reviewed",
            "--profile",
            "ai-forensics",
            "--k",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert "A title" in result.output and "https://x/1" in result.output
    # The stub received the flags: k and both filters were forwarded.
    assert captured["k"] == 3
    assert captured["filters"] == Filters(frozenset({"peer_reviewed"}), frozenset({"ai-forensics"}))
    assert built == ["embedder"]
    assert closed == [True]


def test_search_help_does_not_mention_sources() -> None:
    # search runs no source, so its --live help must not reuse ingest's wording.
    result = CliRunner().invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "sources" not in result.output.lower()


def _stub_eval(monkeypatch: pytest.MonkeyPatch, evaluate_fn: object) -> None:
    # The command runs the eval in a throwaway database; stub that with a session so nothing is
    # created, and stub evaluate so the metrics are fixed.
    session = SimpleNamespace(rollback=lambda: None, close=lambda: None)
    monkeypatch.setattr(cli, "_with_throwaway_database", lambda run: run(session))
    monkeypatch.setattr(retrieval_eval, "evaluate", evaluate_fn)


def test_eval_passes_when_the_gate_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_eval(monkeypatch, lambda *args, **kwargs: {name: 1.0 for name in GATED})
    monkeypatch.setattr(retrieval_eval, "gate", lambda metrics, baseline: [])
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 0


def test_eval_reports_a_regression_and_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_eval(monkeypatch, lambda *args, **kwargs: {name: 1.0 for name in GATED})
    monkeypatch.setattr(
        retrieval_eval, "gate", lambda metrics, baseline: ["recall_at_k dropped from 0.8 to 0.5"]
    )
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 1
    assert "REGRESSION: recall_at_k dropped from 0.8 to 0.5" in result.output


def test_eval_write_baseline_writes_only_the_gated_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stub returns extra keys; the written baseline must hold exactly GATED, because gate
    # validates that set.
    _stub_eval(
        monkeypatch,
        lambda *args, **kwargs: {**{name: 1.0 for name in GATED}, "text_nonempty": 0.9},
    )
    monkeypatch.setattr(cli, "_BASELINE_PATH", tmp_path / "baseline.json")
    result = CliRunner().invoke(app, ["eval", "--write-baseline"])
    assert result.exit_code == 0
    written = json.loads((tmp_path / "baseline.json").read_text())
    assert set(written) == set(GATED)


def test_eval_data_paths_are_repo_rooted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Running from another directory must still find the data, so the paths are absolute and
    # anchored on the repository root, not the current directory.
    captured: dict[str, Path] = {}

    def fake_evaluate(
        session: object, embedder: object, corpus: Path, questions: Path
    ) -> dict[str, float]:
        captured["corpus"], captured["questions"] = corpus, questions
        return {name: 1.0 for name in GATED}

    monkeypatch.setattr(
        cli,
        "_with_throwaway_database",
        lambda run: run(SimpleNamespace(rollback=lambda: None, close=lambda: None)),
    )
    monkeypatch.setattr(retrieval_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(retrieval_eval, "gate", lambda metrics, baseline: [])
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 0
    assert captured["corpus"] == cli._REPO_ROOT / "evals" / "data" / "corpus.jsonl"
    assert captured["questions"] == cli._REPO_ROOT / "evals" / "data" / "questions.jsonl"


def test_eval_refuses_a_non_empty_database(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> object:
        raise DatabaseNotEmpty("the eval needs an empty database")

    _stub_eval(monkeypatch, refuse)
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "empty database" in result.stderr


def test_eval_names_the_createdb_privilege_when_it_cannot_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(run: object) -> object:
        raise cli.EvalDatabaseError("the database role needs the CREATEDB privilege")

    monkeypatch.setattr(cli, "_with_throwaway_database", refuse)
    result = CliRunner().invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "CREATEDB" in result.stderr


def test_eval_live_refuses_to_write_the_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # --live is a manual checkpoint on the real embedder; the gate and the committed baseline stay
    # pinned to the offline FakeEmbedder, so both flags together is a usage error raised before
    # any model, engine or database is built.
    monkeypatch.setattr(cli, "make_embedder", boom)
    monkeypatch.setattr(cli, "_with_throwaway_database", boom)
    # Redirected so a refusal that fell through to writing could not clobber the real baseline.
    monkeypatch.setattr(cli, "_BASELINE_PATH", tmp_path / "baseline.json")
    result = CliRunner().invoke(app, ["eval", "--live", "--write-baseline"])
    assert result.exit_code == 2
    assert "--live" in result.output and "--write-baseline" in result.output
    assert not (tmp_path / "baseline.json").exists()


def test_eval_live_uses_the_configured_embedder_and_never_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The live path must reach make_embedder (never the offline fake) and must skip the gate, so a
    # real-model number can never be compared against, or become, the committed baseline.
    embedder = object()

    def fake_embedder(settings: object, ledger: object) -> object:
        return embedder

    captured: dict[str, object] = {}

    def fake_evaluate(
        session: object, emb: object, corpus: Path, questions: Path
    ) -> dict[str, float]:
        captured["embedder"] = emb
        return {name: 1.0 for name in GATED}

    monkeypatch.setattr(cli, "make_embedder", fake_embedder)
    # A real Settings would read the developer's .env; the live branch only needs an object.
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(max_tokens_per_run=0))
    _stub_eval(monkeypatch, fake_evaluate)
    monkeypatch.setattr(retrieval_eval, "gate", boom)
    # Redirect the baseline so the test can prove the live path never writes it: without this a
    # `if live: write baseline` regression would pass the test and overwrite the real file.
    monkeypatch.setattr(cli, "_BASELINE_PATH", tmp_path / "baseline.json")
    result = CliRunner().invoke(app, ["eval", "--live"])
    assert result.exit_code == 0
    assert captured["embedder"] is embedder
    assert "recall_at_k" in result.output
    assert not (tmp_path / "baseline.json").exists()


def test_eval_db_name_always_matches_the_sweep_pattern() -> None:
    # The sweep drops names matching `_EVAL_DB_RE`, so the generator must always fullmatch it; a
    # changed separator, a dashed uuid or a millisecond epoch would silently stop the sweep.
    name = cli._eval_db_name(1792070000.0)
    assert cli._EVAL_DB_RE.fullmatch(name)
    assert name.startswith("paperwatch_eval_1792070000_")
    # A fractional `now` is truncated to whole seconds, keeping the 10-digit epoch group.
    assert cli._EVAL_DB_RE.fullmatch(cli._eval_db_name(1792070000.987))


# --- qa-eval (Task 18) ---


def test_qa_eval_refuses_live_with_write_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A live recording is evidence for one model and must never become the gate.
    _qa_cli_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_with_throwaway_database", boom)
    result = CliRunner().invoke(app, ["qa-eval", "--live", "--write-baseline"])
    assert result.exit_code == 2
    assert "--live" in result.output and "--write-baseline" in result.output


def test_qa_eval_model_requires_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _qa_cli_paths(monkeypatch, tmp_path)
    result = CliRunner().invoke(app, ["qa-eval", "--model", "m/x:free"])
    assert result.exit_code == 2 and "--live" in result.output


def test_qa_eval_names_the_missing_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _qa_cli_paths(monkeypatch, tmp_path)
    result = CliRunner().invoke(app, ["qa-eval"])
    assert result.exit_code == 2
    assert "qa-eval --live --model m/x:free" in result.stderr


def test_qa_eval_passes_and_bites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _qa_cli_paths(monkeypatch, tmp_path)
    questions = cli._EVAL_DATA / "qa_questions.jsonl"
    rows = qa_recording()
    evals_qa.write_recorded(path / "m_x_free.jsonl", rows)
    # The baseline is whatever this recording scores; only the wrong-item edit can move it.
    m = evals_qa.score(rows, questions)
    (path / "baseline.json").write_text(json.dumps({name: m[name] for name in evals_qa.GATED}))
    result = CliRunner().invoke(app, ["qa-eval"])
    assert result.exit_code == 0 and "support_recall" in result.output
    rows[0] = qa_row("q01", "ag-02")
    evals_qa.write_recorded(path / "m_x_free.jsonl", rows)
    result = CliRunner().invoke(app, ["qa-eval"])
    assert result.exit_code == 1
    assert "REGRESSION: support_recall" in result.output
    assert "REGRESSION: citation_precision" in result.output


def test_qa_eval_write_baseline_writes_only_the_gated_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _qa_cli_paths(monkeypatch, tmp_path)
    (path / "m_x_free.jsonl").write_text("")
    # The real score is stubbed: this test is about which keys reach the file, not the numbers.
    monkeypatch.setattr(
        evals_qa,
        "score",
        lambda rows, questions: {**dict.fromkeys(evals_qa.GATED, 1.0), "mean_tools": 2.0},
    )
    monkeypatch.setattr(evals_qa, "load_recorded", lambda p: [object()])
    monkeypatch.setattr(evals_qa, "check_complete", lambda *args, **kwargs: None)
    result = CliRunner().invoke(app, ["qa-eval", "--write-baseline"])
    assert result.exit_code == 0
    written = json.loads((path / "baseline.json").read_text())
    assert set(written) == set(evals_qa.GATED)


def test_qa_eval_live_writes_a_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _qa_cli_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_qa_models", lambda settings, model: [object()])
    monkeypatch.setattr(cli, "_with_throwaway_database", lambda run: run(None))
    monkeypatch.setattr(evals_qa, "record", lambda session, models, q, c: [qa_row("q01")])
    result = CliRunner().invoke(app, ["qa-eval", "--live", "--model", "m/x:free"])
    assert result.exit_code == 0
    assert "recorded 1 answers from m/x:free" in result.output
    assert (path / "m_x_free.jsonl").exists()
    assert not (path / "baseline.json").exists()


def test_qa_eval_live_refuses_more_than_two_provider_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _qa_cli_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_qa_models", lambda settings, model: [object()])
    monkeypatch.setattr(cli, "_with_throwaway_database", lambda run: run(None))
    rows = [qa_row(f"q0{i}", provider_error=True) for i in range(1, 4)]
    monkeypatch.setattr(evals_qa, "record", lambda session, models, q, c: rows)
    result = CliRunner().invoke(app, ["qa-eval", "--live", "--model", "m/x:free"])
    assert result.exit_code == 2
    assert "unavailable: 3 of 3" in result.stderr
    assert not (path / "m_x_free.jsonl").exists()


def test_qa_eval_replay_ignores_pw_chat_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A developer's environment must not choose the recording or the baseline: the gate is the
    # committed default's. With PW_CHAT_MODEL set to another model the replay still passes.
    path = _qa_cli_paths(monkeypatch, tmp_path)
    rows = qa_recording()
    evals_qa.write_recorded(path / "m_x_free.jsonl", rows)
    m = evals_qa.score(rows, cli._EVAL_DATA / "qa_questions.jsonl")
    (path / "baseline.json").write_text(json.dumps({name: m[name] for name in evals_qa.GATED}))
    monkeypatch.setenv("PW_CHAT_MODEL", "other/model:free")
    monkeypatch.setattr(cli, "get_settings", lambda: cli.Settings())
    result = CliRunner().invoke(app, ["qa-eval"])
    assert result.exit_code == 0
    assert not (path / "other_model_free.jsonl").exists()


def test_qa_eval_refuses_an_incomplete_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _qa_cli_paths(monkeypatch, tmp_path)
    evals_qa.write_recorded(path / "m_x_free.jsonl", [qa_row("q01")])
    result = CliRunner().invoke(app, ["qa-eval"])
    assert result.exit_code == 2
    assert "covers 1 of 20 questions" in result.stderr
    # The refusal names the file and says exactly how to fix it.
    assert "m_x_free.jsonl" in result.stderr
    assert "uv run paperwatch qa-eval --live --model m/x:free" in result.stderr


# --- ask, serve and the terminal rendering (Task 17, review round 1) ---

HOSTILE = "\u202e"


def scripted_run(*evts: object) -> Callable[..., Iterator[object]]:
    # A generator, not iter(): `ask` closes the stream, which is how the database session is
    # released when an error event ends the loop early. `cancel` is accepted because `ask` passes
    # the event its Ctrl-C handler sets.
    return lambda deps, question, filters, cancel=None: (event for event in evts)


def answer_with(
    text: str = "Coding agents are scored on real bugs [S1].",
    *,
    citations: tuple[Citation, ...] = (),
    flags: tuple[Flag, ...] = (),
    supported: bool = False,
) -> CheckedAnswer:
    return CheckedAnswer(text, citations, flags, supported)


def citation(**overrides: object) -> Citation:
    fields: dict[str, object] = {
        "alias": "S1",
        "item_id": uuid4(),
        "title": "A title marker",
        "url": "https://x/marker",
        "tier": "peer_reviewed",
        "basis": "venue",
    }
    fields.update(overrides)
    return Citation(**fields)  # type: ignore[arg-type]


def test_ask_requires_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", boom)
    result = CliRunner().invoke(app, ["ask", "what is new?"])
    assert result.exit_code == 2 and "pass --live" in result.output


def test_serve_requires_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", boom)
    assert CliRunner().invoke(app, ["serve"]).exit_code == 2


def test_ask_prints_answer_sources_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    flag = Flag("uncited_claim", "Most agents forget everything", "no citation")
    answer = answer_with(
        "Coding agents are scored on real bugs [S1].",
        citations=(citation(title="SWE-bench results", url="https://x/swe"),),
        flags=(flag,),
    )
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(TokenEvent("Coding agents"), AnswerEvent(answer))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "Sources:" in result.output
    assert "[S1] SWE-bench results (peer_reviewed) https://x/swe" in result.output
    assert "warning: uncited_claim: Most agents forget everything" in result.output


def test_ask_exits_1_on_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(ErrorEvent("daily_cap", "800 calls today"))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    # An error is not answer text, so it goes to stderr and stdout stays empty for a pipe.
    assert "800 calls today" in result.stderr
    assert result.stdout == ""


def test_ask_ends_the_streamed_line_before_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli,
        "run_question",
        scripted_run(TokenEvent("half an ans"), ErrorEvent("model_unavailable", "all failed")),
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert result.stdout.endswith("\n")
    assert "half an ans" in result.stdout


def test_ask_exits_1_when_the_stream_ends_without_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(TokenEvent("nothing")))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert "without a checked answer" in result.stderr and "Traceback" not in result.stderr


def test_a_cleanup_cancel_is_not_a_user_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    # `run_question` always calls `cancel.set()` in its finally, so a stream that ends without a
    # checked answer used to look like Ctrl-C and exit 130. Only the handler's own event means the
    # user asked to stop; a wiring bug must still exit 1.
    def stream(
        deps: object, question: str, filters: object, cancel: object = None
    ) -> Iterator[object]:
        try:
            yield TokenEvent("nothing")
        finally:
            cancel.set()  # type: ignore[attr-defined] - what run_question's cleanup does

    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert "without a checked answer" in result.stderr
    assert "interrupted" not in result.stderr.lower()


def test_ask_prints_the_checked_text_again_when_it_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The streamed text can differ from the checked answer: an alias was masked to [?], or the
    # checker replaced the whole answer with NO_SUPPORT. The checked text is authoritative.
    answer = answer_with("Coding agents cite [S1].", supported=True)
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli,
        "run_question",
        scripted_run(TokenEvent("Coding agents cite [?]."), AnswerEvent(answer)),
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "---" in result.output
    assert result.output.count("Coding agents cite [S1].") == 1


def test_ask_does_not_repeat_identical_text(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = answer_with("Coding agents cite [S1].", supported=True)
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli,
        "run_question",
        scripted_run(TokenEvent("Coding agents cite [S1]."), AnswerEvent(answer)),
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "---" not in result.output
    assert result.output.count("Coding agents cite [S1].") == 1


def test_ask_hides_and_logs_flags_beside_no_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from paperwatch.agent.citations import NO_SUPPORT

    flag = Flag("uncited_claim", "Most agents forget everything", "no citation")
    answer = answer_with(NO_SUPPORT, flags=(flag,))
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    # The command configures structlog for the process; keep the capture's configuration so the
    # log call can be read back.
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(cli, "run_question", scripted_run(AnswerEvent(answer)))
    with capture_logs() as logs:
        result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert NO_SUPPORT in result.output
    assert "warning:" not in result.output
    logged = [entry for entry in logs if entry["event"] == "answer_unsupported"]
    assert logged and logged[0]["flags"][0]["sentence"] == flag.sentence


@pytest.mark.parametrize(
    "hostile",
    [
        "\r",  # carriage return: overwrites the line the reader is looking at
        "\u2028",  # line separator
        "\u2029",  # paragraph separator
        "\u200b",  # zero-width space
        "\u200e",  # left-to-right mark: reorders neutral characters
        "\u061c",  # Arabic letter mark
        "\U000e0041",  # a tag character: invisible text that survives a copy
        "\ud800",  # a lone surrogate: typer.echo cannot encode it
        "\x1b",  # ESC: starts every ANSI sequence, OSC 8 links included
        "\x7f",  # DEL
        "\x9b",  # C1 CSI
        "\ufeff",  # byte-order mark
        "\u00ad",  # soft hyphen
    ],
)
def test_terminal_safe_removes_control_and_format_characters(hostile: str) -> None:
    # The one escaping rule for everything `ask` prints: a terminal acts on these, so they become
    # U+FFFD. The categories are Cc, Cf, Cs, Zl and Zp, minus the two formatting characters real
    # text needs (see the next test).
    assert cli.terminal_safe(f"a{hostile}b") == "a\ufffdb"


def test_terminal_safe_keeps_newline_tab_and_the_joiners() -> None:
    # ZWNJ and ZWJ are formatting characters, but Persian, Indic scripts and emoji sequences need
    # them, so removing them would corrupt the text.
    assert cli.terminal_safe("line\ttab\nnext\u200cjoin\u200djoin") == (
        "line\ttab\nnext\u200cjoin\u200djoin"
    )


def test_terminal_safe_keeps_ordinary_unicode_readable() -> None:
    assert cli.terminal_safe("Agents benchmark édité à Zürich, 日本, 42 %\nnext line\tlast") == (
        "Agents benchmark édité à Zürich, 日本, 42 %\nnext line\tlast"
    )


def test_ask_neutralises_an_osc_8_link_in_a_streamed_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OSC 8 makes model-written text a clickable link in iTerm2, kitty and VS Code; every
    # clickable link must come from Citation.url, so control bytes never reach the terminal.
    hostile = "\x1b]8;;http://evil.example\x1b\\click me\x1b]8;;\x1b\\"
    answer = answer_with(hostile, supported=False)
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(TokenEvent(hostile), AnswerEvent(answer)))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "\x1b" not in result.output
    assert "\ufffd" in result.output


@pytest.mark.parametrize("field", ["alias", "title", "tier", "url"])
def test_ask_escapes_every_citation_field(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # One test per printed field: a mutant that drops terminal_safe from one call site must fail.
    ref = citation(**{field: f"bad{HOSTILE}side marker"})
    answer = answer_with("Some answer [S1].", citations=(ref,), supported=True)
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(AnswerEvent(answer)))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "marker" in result.output
    assert HOSTILE not in result.output


@pytest.mark.parametrize("field", ["sentence", "detail"])
def test_ask_escapes_every_flag_field(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    flag = Flag(
        **{
            "kind": "uncited_claim",
            "sentence": "a claim",
            "detail": "why",
            field: f"bad{HOSTILE}side marker",
        }
    )  # type: ignore[arg-type]
    answer = answer_with("Some answer [S1].", flags=(flag,))
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(AnswerEvent(answer)))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "marker" in result.output
    assert HOSTILE not in result.output


def test_ask_escapes_the_checked_text(monkeypatch: pytest.MonkeyPatch) -> None:
    # The checked text replaces the model's, and it is rendered too.
    answer = answer_with(f"right{HOSTILE}to left [S1].", supported=True)
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(AnswerEvent(answer)))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "left" in result.output and HOSTILE not in result.output


def test_ask_escapes_the_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(ErrorEvent("internal", f"bad{HOSTILE}side marker"))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert "marker" in result.stderr and HOSTILE not in result.stderr


def test_ask_escapes_the_tool_name(monkeypatch: pytest.MonkeyPatch) -> None:
    event = ToolEvent(f"search{HOSTILE}_corpus", {"query": "x"})
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(event, AnswerEvent(answer_with("ok [S1].")))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "corpus" in result.stderr and HOSTILE not in result.stderr


def test_ask_quotes_model_written_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    # The tool call is model output too. It is rendered as JSON, whose escaping (ensure_ascii,
    # the json.dumps default) keeps a hostile character as the text `\u202e` instead of a byte.
    event = ToolEvent("search_corpus", {"query": f"bad{HOSTILE}side marker"})
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(event, AnswerEvent(answer_with("ok [S1].")))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "marker" in result.stderr
    assert HOSTILE not in result.stderr
    assert "\\u202e" in result.stderr


def test_ask_prints_a_lone_surrogate_as_replacement_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A provider can send "\ud800" in a JSON stream and the SDK decodes a lone surrogate, which
    # cannot be encoded: the answer must print, not crash with a traceback (Review Focus 3).
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(
        cli,
        "run_question",
        scripted_run(TokenEvent("a\ud800b"), AnswerEvent(answer_with("a\ud800b", supported=False))),
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "\ufffd" in result.output
    assert "Traceback" not in result.output and "UnicodeEncodeError" not in result.stderr


def test_ask_refuses_a_blank_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", boom)
    result = CliRunner().invoke(app, ["ask", "   ", "--live"])
    assert result.exit_code == 2
    assert "blank" in result.output.lower()


def test_ask_refuses_an_unknown_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", boom)
    result = CliRunner().invoke(app, ["ask", "q", "--live", "--tier", "bogus"])
    assert result.exit_code == 2 and "tier" in result.output.lower()


def test_ask_refuses_an_unknown_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", boom)
    result = CliRunner().invoke(app, ["ask", "q", "--live", "--profile", "not-a-profile"])
    assert result.exit_code == 2 and "profile" in result.output.lower()


def test_search_refuses_an_unknown_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    # `search` and `ask` share one validation helper, so a typo is refused by both.
    monkeypatch.setattr(cli, "make_embedder", boom)
    result = CliRunner().invoke(app, ["search", "q", "--live", "--profile", "not-a-profile"])
    assert result.exit_code == 2 and "profile" in result.output.lower()


def test_the_live_commands_find_the_profiles_file_from_another_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Both commands used to read `config/profiles.yaml` relative to the working directory, so the
    # console script died with a FileNotFoundError traceback anywhere else, before the first model
    # or connection. Profile names are configuration, so they resolve from the repository.
    monkeypatch.chdir(tmp_path)
    # The missing chat key is the only thing left for `ask`, and it must be reported as a usage
    # error rather than a traceback: reaching it means the profile file was found and read.
    missing_key = CliRunner().invoke(app, ["ask", "q", "--live", "--profile", "ai-forensics"])
    assert missing_key.exit_code == 2
    assert "PW_OPENROUTER_API_KEY" in missing_key.stderr
    assert "Traceback" not in missing_key.stderr

    # `search` proves the same file was found by validating a name against its contents.
    monkeypatch.setattr(cli, "make_embedder", boom)
    bad = CliRunner().invoke(app, ["search", "q", "--live", "--profile", "not-a-profile"])
    assert bad.exit_code == 2 and "profile" in bad.output.lower()
    assert "Traceback" not in bad.stderr


def test_search_without_a_profile_never_reads_the_profiles_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `search` did not touch the file before Task 17, and a missing or malformed file must not
    # turn a plain search into a traceback: names are loaded only when `--profile` is given.
    monkeypatch.setattr(cli, "_PROFILES_PATH", tmp_path / "absent.yaml")
    monkeypatch.setattr(cli, "make_embedder", lambda settings, ledger: FakeEmbedder())
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(
        cli, "make_session_factory", lambda engine: lambda: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(cli, "hybrid_search", lambda *args, **kwargs: [])
    result = CliRunner().invoke(app, ["search", "q", "--live"])
    assert result.exit_code == 0
    assert "Traceback" not in result.stderr


def test_ask_keeps_log_lines_out_of_its_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    # `ask` configures structlog for the process, and structlog's default logger writes to
    # stdout: the tool call's INFO line landed in the middle of the answer, and `ask q > a.txt`
    # captured it. Log lines belong on stderr, beside the `tool:` progress line.
    def stream(
        deps: object, question: str, filters: object, cancel: object = None
    ) -> Iterator[object]:
        structlog.get_logger().info("search_corpus", query_chars=9)
        yield ToolEvent("search_corpus", {"query": "SWE-bench"})
        yield AnswerEvent(answer_with(supported=True))

    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    # stdout is the answer and its sources only; every log line is on stderr.
    assert result.stdout.strip().endswith("Coding agents are scored on real bugs [S1].")
    assert "search_corpus" not in result.stdout and '"level"' not in result.stdout
    assert any("search_corpus" in line and '"level"' in line for line in result.stderr.splitlines())


def test_serve_without_a_chat_key_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    # `serve` refuses like `ask` does: a usage error that names the variable, before any engine or
    # embedding model is built (the chat chain is first in `live_deps`).
    from paperwatch.api import app as api

    monkeypatch.setattr(api, "make_embedder", boom)
    monkeypatch.setattr(api, "make_engine", boom)
    monkeypatch.setattr(cli, "make_verifier", lambda s: object())
    result = CliRunner().invoke(app, ["serve", "--live"])
    assert result.exit_code == 2
    assert "PW_OPENROUTER_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_serve_cannot_crash_on_a_log_level_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    # `logging` accepts WARN, but uvicorn's Config raises KeyError for it. Settings stores the
    # canonical name, so the level uvicorn gets is one it knows.
    from fastapi import FastAPI

    from paperwatch.config import Settings

    captured: dict[str, object] = {}
    monkeypatch.setenv("PW_LOG_LEVEL", "WARN")
    # `get_settings` is cached per process; build from the test's own environment instead.
    monkeypatch.setattr(cli, "get_settings", lambda: Settings())
    monkeypatch.setattr(cli, "make_verifier", lambda s: object())
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: "deps")
    monkeypatch.setattr(cli, "create_app", lambda deps: FastAPI())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app_, **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(app, ["serve", "--live"])
    assert result.exit_code == 0
    assert captured["log_level"] == "warning"


def test_ask_names_the_missing_chat_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # No key: exit 2 with the variable's name, before any model, engine or embedder is built. The
    # api module's factory is patched too: `live_deps` imports its own, and a regression that
    # built the embedder first would download the real model in a test run.
    from paperwatch.api import app as api

    monkeypatch.setattr(cli, "make_embedder", boom)
    monkeypatch.setattr(cli, "make_engine", boom)
    monkeypatch.setattr(api, "make_embedder", boom)
    monkeypatch.setattr(api, "make_engine", boom)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 2
    assert "PW_OPENROUTER_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_ask_passes_the_filters_to_run_question(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, Filters]] = []

    def stream(
        deps: object, question: str, filters: Filters, cancel: object = None
    ) -> Iterator[object]:
        seen.append((question, filters))
        yield AnswerEvent(answer_with("ok [S1].", supported=True))

    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(
        app,
        ["ask", "q", "--live", "--tier", "peer_reviewed", "--profile", "ai-forensics"],
    )
    assert result.exit_code == 0
    assert seen == [("q", Filters(frozenset({"peer_reviewed"}), frozenset({"ai-forensics"})))]


def test_ask_closes_the_stream_when_an_error_ends_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # An error event ends the loop before the stream is exhausted, and the open generator holds
    # the answer's database session until it is closed.
    closed: list[bool] = []

    def stream(
        deps: object, question: str, filters: object, cancel: object = None
    ) -> Iterator[object]:
        try:
            yield ErrorEvent("internal", "boom")
        finally:
            closed.append(True)

    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert closed == [True]


def test_ask_exits_130_on_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ctrl-C lands while the consumer waits for the next event. Closing the stream aborts the
    # in-flight model call (Task 17 Step 9 item 1b); the command exits like a shell program
    # interrupted by the user, with one line and no traceback.
    closed: list[bool] = []
    received: list[object] = []

    def stream(
        deps: object, question: str, filters: object, cancel: object = None
    ) -> Iterator[object]:
        received.append(cancel)
        try:
            yield TokenEvent("half an answer")
            raise KeyboardInterrupt
        finally:
            closed.append(True)

    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 130
    assert closed == [True]
    # `ask` hands its Ctrl-C event to the answer, so the handler can stop the model call before
    # the interrupt unwinds the stream.
    assert isinstance(received[0], threading.Event)
    assert "interrupted" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_ask_handles_a_real_sigint_and_restores_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real signal, not a raised KeyboardInterrupt: this proves `ask` installs the handler in the
    # main thread, that the command's own event reaches the answer, and that the previous handler
    # is put back on the way out.
    original = signal.getsignal(signal.SIGINT)
    seen: list[object] = []

    def stream(
        deps: object, question: str, filters: object, cancel: object = None
    ) -> Iterator[object]:
        seen.append(cancel)
        yield TokenEvent("half an answer")
        os.kill(os.getpid(), signal.SIGINT)
        # The real agent stops at its next chance once the flag is set; this is that step.
        if not cancel.is_set():  # type: ignore[attr-defined]
            yield AnswerEvent(answer_with(supported=True))

    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: object())
    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 130
    assert isinstance(seen[0], threading.Event) and seen[0].is_set()
    # The first press says what it is doing; the command then stops within the current step.
    assert "stopping; press Ctrl-C again to quit now" in result.stderr
    assert "interrupted: the answer was stopped" in result.stderr
    # The streamed line is ended before the message, so a terminal does not splice them together.
    assert result.stdout.endswith("\n") and result.stdout.strip() == "half an answer"
    assert "Traceback" not in result.stderr
    assert signal.getsignal(signal.SIGINT) is original


def test_the_second_ctrl_c_leaves_at_once(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model call runs in a thread that cannot be interrupted, and Python joins it at exit, so
    # the second press is the only way out of a stalled request: flush and leave.
    cancel, interrupted = threading.Event(), threading.Event()
    exits: list[int] = []
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    handler = cli._interrupt_handler(cancel, interrupted)
    handler(signal.SIGINT, None)
    assert exits == [] and cancel.is_set() and interrupted.is_set()
    handler(signal.SIGINT, None)
    assert exits == [130]
    assert "stopping; press Ctrl-C again to quit now" in capsys.readouterr().err


def test_serve_runs_uvicorn_on_localhost_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The API must not listen on every interface, and uvicorn is told not to configure logging:
    # the app's own structured setup stays in charge.
    captured: dict[str, object] = {}

    def fake_run(app_: object, **kwargs: object) -> None:
        captured.update(app=app_, **kwargs)

    from fastapi import FastAPI

    built = FastAPI()
    seen: list[object] = []
    monkeypatch.setattr(cli, "make_verifier", lambda s: object())
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: "deps")
    monkeypatch.setattr(cli, "create_app", lambda deps: seen.append(deps) or built)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    result = CliRunner().invoke(app, ["serve", "--live"])
    assert result.exit_code == 0
    assert captured["app"] is built
    assert seen == ["deps"]
    assert (captured["host"], captured["port"]) == ("127.0.0.1", 8000)
    assert captured["log_config"] is None
    assert captured["log_level"] == "info"
    # Open streams are cancelled when the server stops instead of being waited for: a Ctrl-C on
    # `serve` must not wait for a whole in-flight answer.
    assert captured["timeout_graceful_shutdown"] == cli.GRACEFUL_SHUTDOWN_S
    assert "warning" not in result.stderr.lower()


def test_serve_refuses_to_start_without_a_verifiable_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default `oidc` mode with no PW_VERCEL_OIDC_* values: a usage error naming the variable,
    # before any engine or model is built.
    monkeypatch.setattr(cli, "live_deps", boom)
    result = CliRunner().invoke(app, ["serve", "--live"])
    assert result.exit_code == 2
    assert "PW_VERCEL_OIDC_ISSUER" in result.stderr
    assert "Traceback" not in result.stderr


def test_ask_runs_without_an_api_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # `ask` serves no route: a .env with only the chat key must reach the answer through the real
    # `live_deps`, not die on the unset PW_VERCEL_OIDC_* values `serve` needs.
    from paperwatch.api import app as api
    from paperwatch.config import Settings

    monkeypatch.setenv("PW_OPENROUTER_API_KEY", "sk-test")
    # `get_settings` is cached per process; build from the test's own environment instead.
    monkeypatch.setattr(cli, "get_settings", lambda: Settings())
    monkeypatch.setattr(api, "load_profiles", lambda path: [])
    monkeypatch.setattr(api, "WorkerEmbedder", lambda should_stop: object())
    built: list[object] = []

    def stream(
        deps: object, question: str, filters: object, cancel: object = None
    ) -> Iterator[object]:
        built.append(deps)
        yield AnswerEvent(answer_with(supported=True))

    monkeypatch.setattr(cli, "run_question", stream)
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0, result.output
    assert isinstance(built[0], api.AgentDeps)
    assert built[0].verifier.verify("any-token") is None


def test_serve_prefetches_the_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    # N1: the key set is warmed in the serve path, not in `live_deps`, so `paperwatch ask` never
    # fetches keys it does not use.
    from fastapi import FastAPI

    class Verifier:
        prefetched = False

        def prefetch(self) -> None:
            self.prefetched = True

    verifier = Verifier()

    class Deps:
        pass

    deps = Deps()
    deps.verifier = verifier  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "create_app", lambda deps_: FastAPI())
    cli._served_app(deps, "127.0.0.1")  # type: ignore[arg-type]
    assert verifier.prefetched is True


def test_an_ipv6_bind_prints_a_bracketed_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # `http://::1:8000` is not a URL a person or a client can use; IPv6 literals need brackets.
    from fastapi import FastAPI

    monkeypatch.setattr(cli, "make_verifier", lambda s: object())
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: "deps")
    monkeypatch.setattr(cli, "create_app", lambda deps: FastAPI())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app_, **kwargs: None)
    result = CliRunner().invoke(app, ["serve", "--live", "--host", "::1"])
    assert result.exit_code == 0
    assert "http://[::1]:8000" in result.stdout


def test_serve_warns_about_a_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI

    built = FastAPI()
    monkeypatch.setattr(cli, "make_verifier", lambda s: object())
    monkeypatch.setattr(cli, "live_deps", lambda s, verifier: "deps")
    monkeypatch.setattr(cli, "create_app", lambda deps: built)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app_, **kwargs: None)
    result = CliRunner().invoke(app, ["serve", "--live", "--host", "192.0.2.7"])
    assert result.exit_code == 0
    assert "warning" in result.stderr.lower()
    assert "behind the tunnel" in result.stderr.lower()


def test_api_token_refuses_the_oidc_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # A token minted under the OIDC setting would never verify, so the command refuses instead of
    # printing something useless.
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    result = CliRunner().invoke(app, ["api-token", "--live"])
    assert result.exit_code == 2
    assert "not hmac" in result.stderr


def test_api_token_prints_a_verifiable_token(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "a-test-secret-that-is-long-enough-to-pass"  # noqa: S105 - a test secret
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(_env_file=None, api_auth="hmac", api_hmac_secret=secret),
    )
    result = CliRunner().invoke(app, ["api-token", "--live"])
    assert result.exit_code == 0
    verifier = HmacVerifier(secret, time.time)
    assert verifier.verify(result.stdout.strip()) is not None


def test_the_cli_never_claims_an_answer_was_verified() -> None:
    # supported means "every visible claim points at a retrieved source", not "verified"
    # (docs/learn/11): a CLI string must not say otherwise.
    for command in (["--help"], ["ask", "--help"], ["serve", "--help"]):
        output = CliRunner().invoke(app, command).output.lower()
        assert "verified" not in output and "fact-check" not in output


def test_ingest_followup_is_the_units_flag_and_carries_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --followup is how the scheduled unit keeps retrying S2 refusals; a manual run does not pass
    # it, so the default must stay off. The bounds come from Settings, so a drop-in can shorten the
    # window without a release.
    seen: dict[str, object] = {}
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ())),
    )

    def stub(*args: object, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ()))

    monkeypatch.setattr(cli, "run_ingestion", stub)
    # An environment-free Settings: the command reads get_settings(), which reads PW_* variables.
    monkeypatch.setenv("PW_FOLLOWUP_MAX_PASSES", "7")
    monkeypatch.setenv("PW_FOLLOWUP_INTERVAL_S", "900")
    monkeypatch.setenv("PW_FOLLOWUP_DEADLINE_HOUR", "20")
    cli.get_settings.cache_clear()
    try:
        result = CliRunner().invoke(
            cli.app, ["ingest", "--live", "--followup"], catch_exceptions=False
        )
    finally:
        cli.get_settings.cache_clear()
    assert result.exit_code == 0
    assert seen["followup"] is True
    bounds = seen["bounds"]
    assert isinstance(bounds, cli.Followup)
    assert (bounds.max_passes, bounds.interval_s, bounds.deadline_hour) == (7, 900, 20)

    seen.clear()
    cli.get_settings.cache_clear()
    try:
        CliRunner().invoke(cli.app, ["ingest", "--live"], catch_exceptions=False)
    finally:
        cli.get_settings.cache_clear()
    # A manual run never waits: the default is off.
    assert seen["followup"] is False


def test_ingest_reads_its_config_from_the_repository_not_the_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The ingest unit sets WorkingDirectory to the checkout, but the command itself must not
    # depend on it: `paperwatch ingest --live` from any other directory still needs
    # config/feeds.yaml and config/admission.yaml, the way config.py resolves profiles.yaml.
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(
            status="succeeded", counts={"fetched": 1}, tokens=0, health=Health(True, ())
        ),
    )
    real_feeds, real_admission = cli.load_feeds, cli.load_admission
    paths: dict[str, Path] = {}

    def feeds(path: Path) -> object:
        paths["feeds"] = path
        return real_feeds(path)

    def admission(path: Path) -> object:
        paths["admission"] = path
        return real_admission(path)

    monkeypatch.setattr(cli, "load_feeds", feeds)
    monkeypatch.setattr(cli, "load_admission", admission)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.app, ["ingest", "--live"], catch_exceptions=False)
    assert result.exit_code == 0
    assert paths["feeds"] == cli.REPO_ROOT / "config" / "feeds.yaml"
    assert paths["admission"] == cli.REPO_ROOT / "config" / "admission.yaml"


def test_sigterm_ends_the_run_instead_of_exiting_silently() -> None:
    # systemd stops a unit with SIGTERM: raising SystemExit is what lets the pipeline close the run
    # row as failed before the process leaves (docs/deploy/vps.md).
    with pytest.raises(SystemExit) as raised:
        cli._exit_on_sigterm(signal.SIGTERM, None)
    assert raised.value.code == 128 + signal.SIGTERM


def test_ingest_takes_over_sigterm_only_for_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # The handler is installed before the run and the previous one is restored after it, so a
    # stopped ingestion closes its row while the rest of the process keeps the default behaviour.
    seen: dict[str, object] = {}
    _stub_live_ingest(
        monkeypatch,
        SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ())),
    )

    def stub(*args: object, **kwargs: object) -> SimpleNamespace:
        seen["handler"] = signal.getsignal(signal.SIGTERM)
        return SimpleNamespace(status="succeeded", counts={}, tokens=0, health=Health(True, ()))

    monkeypatch.setattr(cli, "run_ingestion", stub)
    before = signal.getsignal(signal.SIGTERM)
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0
    assert seen["handler"] is cli._exit_on_sigterm
    assert signal.getsignal(signal.SIGTERM) is before


class _SuggestRepo:
    """The repository the `suggest` command reads: one pick and one Rising item, no summaries."""

    def __init__(self) -> None:
        now = datetime(2026, 11, 2, 9, tzinfo=UTC)
        self.pick = Candidate(
            item_id=uuid4(),
            source="arxiv",
            external_id="2510.00001",
            title="AgentBench-Lite: a small benchmark for tool-using agents",
            url="https://arxiv.org/abs/2510.00001",
            tier=Tier.PEER_REVIEWED,
            basis=AdmissionBasis.VENUE,
            profile="agentic-engineering",
            relevance=0.93,
            published_at=now - timedelta(days=1),
            authors=("Ada Lovelace",),
            org=None,
            github_repo=None,
            github_stars=None,
            project_page=None,
            text_chars=900,
            hf_upvotes=None,
            hf_listed_at=None,
            upvote_series=(),
            upvote_baseline=None,
            promoted_to_peer_at=None,
            repo_first_seen_at=None,
            embedding=(),
            venue="ICLR",
            citation_count=12,
        )
        self.rising = dataclasses.replace(
            self.pick,
            item_id=uuid4(),
            external_id="2510.00002",
            tier=Tier.VETTED_PREPRINT,
            basis=AdmissionBasis.LAB_AUTHOR,
            title="A prevalence study of agent memory",
        )

    def suggestion_candidates(self, **kwargs: object) -> list[Candidate]:
        return [self.pick, self.rising]

    def excluded_items(self) -> frozenset[object]:
        return frozenset()

    def summary_sources(self, ids: object) -> dict[object, tuple[str, str]]:
        return {
            candidate.item_id: (candidate.title, "A body " * 100)
            for candidate in (self.pick, self.rising)
        }

    def summaries(self, ids: object) -> dict[object, object]:
        return {}


def test_suggest_prints_picks_and_rising(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _SuggestRepo()
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings())
    monkeypatch.setattr(cli, "make_engine", lambda url, **kwargs: engine)
    session = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: session)
    monkeypatch.setattr(cli, "Repository", lambda session: repo)

    result = CliRunner().invoke(app, ["suggest", "--live"])

    assert result.exit_code == 0
    assert "[pick] AgentBench-Lite" in result.output
    assert "[rising] A prevalence study of agent memory" in result.output
    assert "peer-reviewed via venue: ICLR" in result.output
    assert "preprint from a recognised lab's authors" in result.output
    assert "summary: pending" in result.output
