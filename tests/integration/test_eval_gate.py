"""The offline retrieval eval: metrics, per-kind and per-branch reporting, the gate, the guard."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import false, select
from sqlalchemy.orm import Session

from evals.retrieval_eval import GATED, DatabaseNotEmpty, evaluate, gate
from paperwatch.domain import SourceItem
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval import hybrid
from paperwatch.store.models import Item
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration

CORPUS = Path("evals/data/corpus.jsonl")
QUESTIONS = Path("evals/data/questions.jsonl")
BASELINE = Path("evals/baseline.json")

# The full metric surface: hybrid, text-branch-only and vector-branch-only, overall and per kind.
KEYS = {
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "recall_at_k:exact_name",
    "mrr:exact_name",
    "recall_at_k:paraphrase",
    "mrr:paraphrase",
    "recall_at_k:text_only",
    "mrr:text_only",
    "recall_at_k:text_only:exact_name",
    "mrr:text_only:exact_name",
    "recall_at_k:text_only:paraphrase",
    "mrr:text_only:paraphrase",
    "recall_at_k:vector_only",
    "mrr:vector_only",
    "recall_at_k:vector_only:exact_name",
    "mrr:vector_only:exact_name",
    "recall_at_k:vector_only:paraphrase",
    "mrr:vector_only:paraphrase",
    "text_nonempty",
    "text_nonempty:exact_name",
    "text_nonempty:paraphrase",
}


def test_metrics_in_range_and_gate_passes(db_session: Session) -> None:
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    assert set(m) == KEYS
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert gate(m, json.loads(BASELINE.read_text())) == []


def test_metrics_are_pinned_to_the_committed_values(db_session: Session) -> None:
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    baseline = json.loads(BASELINE.read_text())
    # Two-sided and tight: the fake embedder and the data are deterministic on a fresh database,
    # so the committed values can be pinned exactly. This kills a metric accumulated with the
    # wrong helper (for example nDCG or per-kind MRR computed as recall).
    for name in GATED:
        assert m[name] == pytest.approx(baseline[name], abs=1e-9)
    assert m["text_nonempty"] == pytest.approx(15 / 31)
    assert m["text_nonempty:paraphrase"] == pytest.approx(1 / 13)
    assert m["recall_at_k:text_only:exact_name"] == pytest.approx(14 / 18)
    assert m["mrr:paraphrase"] == pytest.approx(7 / 13)


def test_evaluate_is_stable_across_two_runs(db_session: Session) -> None:
    # A second run in the same transaction reuses the seeded rows and must return the same dict.
    first = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    second = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    assert first == second


def test_a_fused_tie_is_rejected(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    # A tie at the k-th place would make the top k depend on the fusion's random-id tie-break.
    def tied(*args: object, **kwargs: object) -> list[hybrid.Hit]:
        return [
            hybrid.Hit(uuid4(), "a", "https://x/a", "vetted_preprint", "seed", 1 / 61, "s"),
            hybrid.Hit(uuid4(), "b", "https://x/b", "vetted_preprint", "seed", 1 / 61, "s"),
        ]

    monkeypatch.setattr(hybrid, "hybrid_search", tied)
    with pytest.raises(ValueError, match="tie"):
        evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)


def test_broken_text_branch_fails_the_gate(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The blocker from the Task 11 review: with the text branch broken, the gate must fail. The
    # patched function is the module attribute both `hybrid_search` and `evaluate` read, so the
    # fusion loses the text list and the branch-only metrics go empty, as a real regression would.
    monkeypatch.setattr(
        hybrid, "text_query", lambda *args, **kwargs: select(Item.id).where(false())
    )
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    assert m["recall_at_k:text_only:exact_name"] == 0.0
    assert gate(m, json.loads(BASELINE.read_text())) != []


def test_gate_flags_regression() -> None:
    metrics = {name: 0.5 for name in GATED}
    baseline = {**metrics, "recall_at_k": 0.8}
    assert gate(metrics, baseline) == ["recall_at_k dropped from 0.800 to 0.500"]


def test_non_empty_database_is_refused(db_session: Session) -> None:
    # A real item would compete with the eval corpus in the ranking, so the eval refuses it.
    Repository(db_session).record_classified(
        SourceItem(
            "arxiv",
            "2510.99999",
            "A real paper",
            ("A",),
            "A real abstract.",
            "https://arxiv.org/abs/2510.99999",
            datetime(2026, 10, 1, tzinfo=UTC),
        ),
        "candidate",
        "agentic-engineering",
        0.9,
    )
    with pytest.raises(DatabaseNotEmpty, match="empty database"):
        evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
