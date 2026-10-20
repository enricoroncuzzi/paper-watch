import json
import signal
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from structlog.testing import capture_logs
from typer.testing import CliRunner

import evals.retrieval_eval as retrieval_eval
from evals.retrieval_eval import GATED, DatabaseNotEmpty
from paperwatch import cli
from paperwatch.agent.citations import CheckedAnswer, Citation, Flag
from paperwatch.agent.graph import AnswerEvent, TokenEvent, ToolEvent
from paperwatch.api.app import ErrorEvent
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(TokenEvent("Coding agents"), AnswerEvent(answer))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "Sources:" in result.output
    assert "[S1] SWE-bench results (peer_reviewed) https://x/swe" in result.output
    assert "warning: uncited_claim: Most agents forget everything" in result.output


def test_ask_exits_1_on_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(ErrorEvent("daily_cap", "800 calls today"))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    # An error is not answer text, so it goes to stderr and stdout stays empty for a pipe.
    assert "800 calls today" in result.stderr
    assert result.stdout == ""


def test_ask_ends_the_streamed_line_before_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(TokenEvent("nothing")))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert "without a checked answer" in result.stderr and "Traceback" not in result.stderr


def test_ask_prints_the_checked_text_again_when_it_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The streamed text can differ from the checked answer: an alias was masked to [?], or the
    # checker replaced the whole answer with NO_SUPPORT. The checked text is authoritative.
    answer = answer_with("Coding agents cite [S1].", supported=True)
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(AnswerEvent(answer)))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "marker" in result.output
    assert HOSTILE not in result.output


def test_ask_escapes_the_checked_text(monkeypatch: pytest.MonkeyPatch) -> None:
    # The checked text replaces the model's, and it is rendered too.
    answer = answer_with(f"right{HOSTILE}to left [S1].", supported=True)
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
    monkeypatch.setattr(cli, "run_question", scripted_run(AnswerEvent(answer)))
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 0
    assert "left" in result.output and HOSTILE not in result.output


def test_ask_escapes_the_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
    monkeypatch.setattr(
        cli, "run_question", scripted_run(ErrorEvent("internal", f"bad{HOSTILE}side marker"))
    )
    result = CliRunner().invoke(app, ["ask", "q", "--live"])
    assert result.exit_code == 1
    assert "marker" in result.stderr and HOSTILE not in result.stderr


def test_ask_escapes_the_tool_name(monkeypatch: pytest.MonkeyPatch) -> None:
    event = ToolEvent(f"search{HOSTILE}_corpus", {"query": "x"})
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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
    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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


def test_ask_names_the_missing_chat_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # No key: exit 2 with the variable's name, before any model, engine or embedder is built.
    monkeypatch.setattr(cli, "make_embedder", boom)
    monkeypatch.setattr(cli, "make_engine", boom)
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

    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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

    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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

    monkeypatch.setattr(cli, "live_deps", lambda s: object())
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


def test_the_interrupt_handler_sets_the_flag_before_it_raises() -> None:
    # This ordering is the fix for Ctrl-C: the graph's cleanup joins the in-flight model call, so
    # the abort flag has to be set before the interrupt starts unwinding.
    cancel = threading.Event()
    handler = cli._interrupt_handler(cancel)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)
    assert cancel.is_set()


def test_serve_runs_uvicorn_on_localhost_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The API has no sign-in until M3's Entra ID, so it must not listen on every interface, and
    # uvicorn is told not to configure logging: the app's own structured setup stays in charge.
    captured: dict[str, object] = {}

    def fake_run(app_: object, **kwargs: object) -> None:
        captured.update(app=app_, **kwargs)

    from fastapi import FastAPI

    built = FastAPI()
    monkeypatch.setattr(cli, "live_deps", lambda s: "deps")
    monkeypatch.setattr(cli, "create_app", lambda deps: built)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    result = CliRunner().invoke(app, ["serve", "--live"])
    assert result.exit_code == 0
    assert captured["app"] is built
    assert (captured["host"], captured["port"]) == ("127.0.0.1", 8000)
    assert captured["log_config"] is None
    assert captured["log_level"] == "info"
    assert any(m.cls.__name__ == "TrustedHostMiddleware" for m in built.user_middleware)
    assert "warning" not in result.stderr.lower()


def test_a_loopback_server_refuses_a_foreign_host() -> None:
    # The middleware must actually reject a rebound Host header, not just be installed.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    built = FastAPI()
    built.add_api_route("/healthz", lambda: {"status": "ok"}, methods=["GET"])
    served = cli._served_app(object(), "127.0.0.1")
    assert TestClient(served).get("/healthz", headers={"host": "127.0.0.1"}).status_code == 200
    assert TestClient(served).get("/healthz", headers={"host": "evil.example"}).status_code == 400


def test_serve_warns_about_a_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI

    built = FastAPI()
    monkeypatch.setattr(cli, "live_deps", lambda s: "deps")
    monkeypatch.setattr(cli, "create_app", lambda deps: built)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app_, **kwargs: None)
    result = CliRunner().invoke(app, ["serve", "--live", "--host", "192.0.2.7"])
    assert result.exit_code == 0
    assert "warning" in result.stderr.lower()
    # A public bind is the operator's choice; the loopback host allow-list would only fail
    # requests there, so it is not installed.
    assert not any(m.cls.__name__ == "TrustedHostMiddleware" for m in built.user_middleware)


def test_the_cli_never_claims_an_answer_was_verified() -> None:
    # supported means "every visible claim points at a retrieved source", not "verified"
    # (docs/learn/11): a CLI string must not say otherwise.
    for command in (["--help"], ["ask", "--help"], ["serve", "--help"]):
        output = CliRunner().invoke(app, command).output.lower()
        assert "verified" not in output and "fact-check" not in output
