"""Pure retrieval-eval arithmetic, gate validation, and the labelled dataset's shape."""

import json
import math
from pathlib import Path

import pytest

from evals.retrieval_eval import (
    GATED,
    _ndcg,
    _recall,
    _reciprocal_rank,
    gate,
)

DATA = Path(__file__).resolve().parents[2] / "evals" / "data"


def test_recall_counts_the_fraction_of_relevant_found() -> None:
    # The hand values from the Task 11 review: two relevant ids, both in the list, and a partial.
    assert _recall(["x", "b", "y", "a", "z"], {"a", "b"}) == 1.0
    assert _recall(["a"], {"a", "b"}) == 0.5
    assert _recall([], {"a"}) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_hit() -> None:
    assert _reciprocal_rank(["x", "b", "y", "a", "z"], {"a", "b"}) == 0.5
    assert _reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0
    assert _reciprocal_rank([], {"a"}) == 0.0


def test_ndcg_matches_hand_computed_values() -> None:
    # DCG over ranks 2 and 4, normalised by the ideal ordering at ranks 1 and 2.
    assert _ndcg(["x", "b", "y", "a", "z"], {"a", "b"}, 5) == pytest.approx(0.6509, abs=1e-4)
    # Fewer hits than k: one of two relevant items, at rank 1, so the ideal is truncated at 2.
    assert _ndcg(["a"], {"a", "b"}, 5) == pytest.approx(0.6131, abs=1e-4)
    assert _ndcg(["a", "b"], {"a", "b"}, 5) == 1.0
    assert _ndcg([], {"a"}, 5) == 0.0


def _metrics(**overrides: float) -> dict[str, float]:
    # A complete, all-perfect metric set, with the named keys overridden per test.
    base = {name: 1.0 for name in GATED}
    base.update(overrides)
    return base


def test_gate_flags_a_drop_beyond_tolerance() -> None:
    baseline = _metrics(recall_at_k=0.8)
    assert gate(_metrics(recall_at_k=0.5), baseline) == ["recall_at_k dropped from 0.800 to 0.500"]


def test_gate_allows_a_drop_within_tolerance() -> None:
    # 0.79 is 0.01 under an 0.8 baseline, inside the 0.02 tolerance.
    assert gate(_metrics(recall_at_k=0.79), _metrics(recall_at_k=0.8)) == []


def test_gate_requires_exactly_the_gated_baseline_keys() -> None:
    with pytest.raises(ValueError, match="baseline"):
        gate(_metrics(), {})
    with pytest.raises(ValueError, match="baseline"):
        gate(_metrics(), {"recall_at_k": 1.0})


def test_gate_rejects_a_non_finite_metric() -> None:
    with pytest.raises(ValueError, match="mrr"):
        gate(_metrics(mrr=math.nan), _metrics())


def _questions() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (DATA / "questions.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _corpus() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (DATA / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_dataset_item_and_question_counts() -> None:
    corpus, questions = _corpus(), _questions()
    assert len(corpus) == 39
    assert len(questions) == 29
    profiles = {row["external_id"]: row["profile"] for row in corpus}
    assert sum(p == "agentic-engineering" for p in profiles.values()) == 25
    assert sum(p == "ai-forensics" for p in profiles.values()) == 6
    assert sum(p == "ml-architectures" for p in profiles.values()) == 8


def test_dataset_references_and_kinds_are_valid() -> None:
    corpus, questions = _corpus(), _questions()
    ids = {row["external_id"] for row in corpus}
    kinds = {q["kind"] for q in questions}
    assert kinds <= {"exact_name", "paraphrase"}
    # Every relevant id exists and every question has at least one.
    for q in questions:
        relevant = q["relevant"]
        assert relevant and set(relevant) <= ids, q
    # The plan's minimums hold, and some questions carry several relevant ids (fractional recall).
    assert sum(q["kind"] == "exact_name" for q in questions) >= 5
    assert sum(q["kind"] == "paraphrase" for q in questions) >= 5
    assert sum(len(q["relevant"]) > 1 for q in questions) >= 2
