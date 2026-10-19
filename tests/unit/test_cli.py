import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import evals.retrieval_eval as retrieval_eval
from evals.retrieval_eval import GATED, DatabaseNotEmpty
from paperwatch import cli
from paperwatch.cli import app
from paperwatch.retrieval.hybrid import Filters


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
        lambda *args, **kwargs: SimpleNamespace(status="succeeded", counts={}, tokens=0),
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0
    assert built == {"embedder": True, "classifier": True}


def test_ingest_prints_the_skip_and_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Another run holds the lock: the command is a successful no-op, not a failure.
    monkeypatch.setattr(cli, "make_embedder", lambda settings, ledger: object())
    monkeypatch.setattr(
        cli, "make_classifier", lambda settings, ledger, profiles, embedder: object()
    )
    monkeypatch.setattr(cli, "make_http_client", lambda settings: object())
    monkeypatch.setattr(cli, "make_engine", lambda url: object())
    monkeypatch.setattr(cli, "make_session_factory", lambda engine: lambda: object())
    monkeypatch.setattr(
        cli,
        "run_ingestion",
        lambda *args, **kwargs: SimpleNamespace(status="skipped", counts={}, tokens=0),
    )
    result = CliRunner().invoke(cli.app, ["ingest", "--live"])
    assert result.exit_code == 0
    assert "skip" in result.output.lower()


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
        ["search", "SWE-bench", "--live", "--tier", "peer_reviewed", "--profile", "p1", "--k", "3"],
    )
    assert result.exit_code == 0
    assert "A title" in result.output and "https://x/1" in result.output
    # The stub received the flags: k and both filters were forwarded.
    assert captured["k"] == 3
    assert captured["filters"] == Filters(frozenset({"peer_reviewed"}), frozenset({"p1"}))
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
