"""The offline retrieval eval: metrics, per-kind reporting and the committed baseline gate."""

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from evals.retrieval_eval import evaluate, gate
from paperwatch.llm.embedder import FakeEmbedder

pytestmark = pytest.mark.integration

CORPUS = Path("evals/data/corpus.jsonl")
QUESTIONS = Path("evals/data/questions.jsonl")
BASELINE = Path("evals/baseline.json")


def test_metrics_in_range_and_gate_passes(db_session: Session) -> None:
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    # The overall metrics gate CI; the per-kind ones show what the lexical fake embedder can and
    # cannot retrieve (natural paraphrase questions versus exact names).
    assert set(m) == {
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "recall_at_k:exact_name",
        "mrr:exact_name",
        "recall_at_k:paraphrase",
        "mrr:paraphrase",
    }
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert gate(m, json.loads(BASELINE.read_text())) == []


def test_gate_flags_regression() -> None:
    assert gate(
        {"recall_at_k": 0.5, "mrr": 0.5, "ndcg_at_k": 0.5},
        {"recall_at_k": 0.8, "mrr": 0.5, "ndcg_at_k": 0.5},
    ) == ["recall_at_k dropped from 0.800 to 0.500"]
