"""Relevance eval: threshold choice, scoring, the gate and the committed set. No model."""

import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from evals import relevance_eval
from evals.relevance_eval import (
    BASELINE,
    DATA,
    NONE,
    PROFILES,
    Example,
    best_threshold,
    evaluate,
    fingerprint,
    gate,
    load_examples,
    score,
    write_thresholds,
)
from paperwatch.config import EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL, Settings
from paperwatch.domain import Classification, Profile, SourceItem, load_profiles
from paperwatch.llm.embedder import FakeEmbedder


def _ids(split: str, n: int) -> Iterator[str]:
    # The split is a hash of the id, so pick ids that land where a test needs them.
    found = 0
    for k in range(10_000):
        arxiv_id = f"2610.{k:05d}"
        if Example(arxiv_id, "t", "a", NONE).split == split:
            yield arxiv_id
            found += 1
            if found == n:
                return


def _row(arxiv_id: str, title: str, label: str, abstract: str = "x") -> str:
    return json.dumps({"arxiv_id": arxiv_id, "title": title, "abstract": abstract, "label": label})


def _profile(name: str, threshold: float = 0.5) -> Profile:
    return Profile(name, 1.0, ("cs.AI",), (name,), f"about {name}", threshold)


def test_threshold_sits_halfway_between_the_lowest_admitted_and_highest_rejected() -> None:
    assert best_threshold([0.9, 0.8, 0.7, 0.6], [True, True, False, False]) == 0.75


def test_threshold_prefers_the_higher_cut_on_an_f1_tie() -> None:
    # Cutting at 0.9 or at 0.6 both give F1 2/3; the higher cut admits less and is more precise.
    assert best_threshold([0.9, 0.8, 0.7, 0.6], [True, False, False, True]) == 0.85


def test_threshold_may_admit_everything_and_stays_inside_the_unit_interval() -> None:
    assert best_threshold([0.5, 0.5], [True, False]) == 0.5
    assert best_threshold([0.9999, 0.2], [True, False]) == 0.6
    assert best_threshold([0.9999], [True]) == 0.999
    assert best_threshold([0.0001], [True]) == 0.001


def test_threshold_needs_a_positive_example() -> None:
    with pytest.raises(ValueError, match="no positive"):
        best_threshold([0.9, 0.1], [False, False])


class _Fixed:
    """A classifier that gives each abstract a fixed answer by arXiv id."""

    def __init__(self, answers: dict[str, str | None]) -> None:
        self._answers = answers

    def classify(self, item: SourceItem) -> Classification:
        return Classification(self._answers[item.external_id], 0.5, "fixed")


def test_score_counts_precision_recall_admitted_precision_and_routing() -> None:
    examples = [
        Example("a", "t", "x", "p"),
        Example("b", "t", "x", "p"),
        Example("c", "t", "x", "q"),
        Example("d", "t", "x", NONE),
    ]
    s = score(_Fixed({"a": "p", "b": None, "c": "p", "d": None}), examples, ["p", "q"])
    p = s.profiles["p"]
    assert (p.precision, p.recall, p.f1) == (0.5, 0.5, 0.5)
    assert (s.profiles["q"].f1, s.profiles["q"].support) == (0.0, 1)
    assert (s.macro_precision, s.macro_recall, s.macro_f1) == (0.25, 0.25, 0.25)
    # Two abstracts were admitted (a and c), and one of them to the right profile.
    assert s.admitted_precision == 0.5
    assert s.routing_accuracy == 0.5


def test_gate_flags_only_a_gated_score_that_drops_beyond_the_tolerance() -> None:
    s = score(_Fixed({"a": "p"}), [Example("a", "t", "x", "p")], ["p"])
    assert gate(s, {"macro_f1": 1.0, "admitted_precision": 1.0}) == []
    assert gate(s, {"macro_f1": 1.02, "admitted_precision": 1.0}) == []
    assert gate(s, {"macro_f1": 1.05, "admitted_precision": 1.0}) == [
        "macro_f1 dropped from 1.050 to 1.000"
    ]


def _axis(i: int, off: float = 0.0) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    v[2] = off
    return v


class _ByTitle:
    """Embeds a profile description or an abstract to a fixed vector chosen by its first line."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t.split("\n")[0]] for t in texts]


def test_evaluate_calibrates_on_validation_and_scores_the_test_split(tmp_path: Path) -> None:
    # p0 and p1 are the first two axes; positives sit near their axis, "none" papers on axis 2.
    vectors = {"about p0": _axis(0), "about p1": _axis(1)}
    rows = []
    for split in ("validation", "test"):
        ids = _ids(split, 3)
        for label, vector in (("p0", _axis(0, 0.2)), ("p1", _axis(1, 0.2)), (NONE, _axis(0, 3.0))):
            title = f"{label} {split}"
            vectors[title] = vector
            rows.append(_row(next(ids), title, label))
    data = tmp_path / "abstracts.jsonl"
    data.write_text("\n".join(rows) + "\n")
    profiles = [_profile("p0", 0.99), _profile("p1", 0.99)]
    c = evaluate(_ByTitle(vectors), "stub", profiles, data)
    # cos(positive, its axis) = 1/sqrt(1.04) = 0.981 and cos("none", axis 0) = 1/sqrt(10) = 0.316,
    # so p0's cut sits halfway between them; p1's sits halfway between 0 and 0.981.
    assert c.thresholds == {"p0": 0.648, "p1": 0.49}
    assert (c.n_validation, c.n_test) == (3, 3)
    assert c.test.macro_f1 == 1.0 and c.test.routing_accuracy == 1.0
    # The previous 0.99 is above every positive's cosine, so it admitted nothing.
    assert c.previous_test.admitted_precision == 0.0
    assert c.fingerprint == fingerprint("stub", profiles, data)


def test_load_examples_refuses_an_unknown_label_a_duplicate_id_and_blank_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(_row("1", "t", "p") + "\n" + _row("2", "t", "other") + "\n")
    with pytest.raises(ValueError, match=r":2: unknown label 'other'"):
        load_examples(path, ["p"])
    path.write_text(_row("1", "t", "p") + "\n\n" + _row("1", "t", NONE) + "\n")
    with pytest.raises(ValueError, match=r":3: duplicate arxiv_id 1"):
        load_examples(path, ["p"])
    path.write_text(_row("1", "t", "p", abstract=" ") + "\n")
    with pytest.raises(ValueError, match="blank"):
        load_examples(path, ["p"])


def test_write_thresholds_changes_only_the_threshold_lines(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    shutil.copy(PROFILES, path)
    before = path.read_text().splitlines()
    new = {"agentic-engineering": 0.801, "ai-forensics": 0.822, "ml-architectures": 0.79}
    write_thresholds(path, new)
    assert {p.name: p.relevance_threshold for p in load_profiles(path)} == new
    changed = [(a, b) for a, b in zip(before, path.read_text().splitlines(), strict=True) if a != b]
    assert len(changed) == 3
    assert all("relevance_threshold:" in a and "relevance_threshold:" in b for a, b in changed)


def test_write_thresholds_refuses_a_file_without_one_threshold_line_per_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text("profiles:\n  - name: p\n  - name: q\n    relevance_threshold: 0.5\n")
    with pytest.raises(ValueError, match="one relevance_threshold line per profile"):
        write_thresholds(path, {"p": 0.4, "q": 0.6})
    assert "0.5" in path.read_text()


class _FakeModel:
    """Stands in for fastembed's TextEmbedding with the offline hashed bag-of-words."""

    def embed(self, documents: list[str], batch_size: int) -> Iterable[Iterable[float]]:
        return FakeEmbedder().embed(documents)


def test_command_prints_and_writes_a_consistent_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(Path("config"), tmp_path / "config")
    profiles = load_profiles(PROFILES)
    rows = []
    for split in ("validation", "test"):
        ids = _ids(split, 2 * (len(profiles) + 1))
        for p in [*profiles, None]:
            for _ in range(2):
                text = p.description if p else "Sourdough fermentation and flour hydration."
                rows.append(_row(next(ids), "t", p.name if p else NONE, text))
    (tmp_path / "evals/data/relevance").mkdir(parents=True)
    (tmp_path / DATA).write_text("\n".join(rows) + "\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(relevance_eval, "get_settings", Settings)
    monkeypatch.setattr(relevance_eval, "make_local_model", lambda _settings: _FakeModel())

    assert relevance_eval.main(["--write"]) == 0

    out = capsys.readouterr().out
    assert "8 validation and 8 test abstracts" in out and "test, keyword" in out
    baseline = json.loads(BASELINE.read_text())
    written = load_profiles(PROFILES)
    assert {p.name: p.relevance_threshold for p in written} == baseline["thresholds"]
    assert baseline["fingerprint"] == fingerprint(LOCAL_EMBEDDING_MODEL, written, DATA)
    assert set(baseline["test"]) >= {"macro_f1", "admitted_precision", "profiles"}


def test_committed_set_labels_every_profile_in_both_splits() -> None:
    profiles = load_profiles(PROFILES)
    examples = load_examples(DATA, [p.name for p in profiles])
    assert len(examples) >= 140
    counts = Counter((e.label, e.split) for e in examples)
    for label in [*(p.name for p in profiles), NONE]:
        for split in ("validation", "test"):
            assert counts[label, split] >= 12, (label, split)
    # Base arXiv ids, without the version suffix, so an abstract cannot appear twice.
    assert all(re.fullmatch(r"\d{4}\.\d{4,5}", e.arxiv_id) for e in examples)


def test_committed_thresholds_are_the_calibrated_ones() -> None:
    # Fails when a description, the model or the labelled set changes without a recalibration.
    baseline = json.loads(BASELINE.read_text())
    profiles = load_profiles(PROFILES)
    assert baseline["model"] == LOCAL_EMBEDDING_MODEL
    assert baseline["fingerprint"] == fingerprint(LOCAL_EMBEDDING_MODEL, profiles, DATA), (
        "recalibrate: uv run python -m evals.relevance_eval --write"
    )
    assert {p.name: p.relevance_threshold for p in profiles} == baseline["thresholds"]
