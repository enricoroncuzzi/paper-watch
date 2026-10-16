"""The offline retrieval eval: metrics, per-kind and per-branch reporting, the gate, the floor."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import false, func, select
from sqlalchemy.orm import Session

from evals.retrieval_eval import GATED, DatabaseNotEmpty, evaluate, gate
from paperwatch.domain import SourceItem
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval import hybrid
from paperwatch.retrieval.hybrid import Filters, _filtered
from paperwatch.store.models import Item
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration

CORPUS = Path("evals/data/corpus.jsonl")
QUESTIONS = Path("evals/data/questions.jsonl")
BASELINE = Path("evals/baseline.json")

# The M1 baseline as merged in 35255dc. A later baseline may rise but never fall below it
# without a recorded decision.
M1_BASELINE = {
    "recall_at_k": 0.8225806451612904,
    "mrr": 0.8064516129032258,
    "ndcg_at_k": 0.8070252438969676,
    "recall_at_k:text_only:exact_name": 0.7777777777777778,
    "mrr:text_only:exact_name": 0.7777777777777778,
}

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
    "fused_ties_in_top_k",
}


def test_metrics_in_range_and_gate_passes(db_session: Session) -> None:
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    assert set(m) == KEYS
    # Every metric is a rate except the informational tie count.
    assert all(0.0 <= v <= 1.0 for name, v in m.items() if name != "fused_ties_in_top_k")
    assert gate(m, json.loads(BASELINE.read_text())) == []


def test_metrics_are_pinned_to_the_committed_values(db_session: Session) -> None:
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    baseline = json.loads(BASELINE.read_text())
    # Two-sided and tight: the fake embedder and the data are deterministic on a fresh database,
    # so the committed values can be pinned exactly. This kills a metric accumulated with the
    # wrong helper (for example nDCG or per-kind MRR computed as recall).
    for name in GATED:
        assert m[name] == pytest.approx(baseline[name], abs=1e-9)
    assert m["text_nonempty"] == pytest.approx(30 / 31)
    assert m["text_nonempty:paraphrase"] == pytest.approx(13 / 13)
    assert m["recall_at_k:text_only:exact_name"] == pytest.approx(16 / 18)
    assert m["mrr:paraphrase"] == pytest.approx(35 / 52)
    # Pinning the vector-only branch kills a substitution of the hybrid list for it.
    assert m["recall_at_k:vector_only"] == pytest.approx(22.5 / 31)
    assert m["mrr:vector_only"] == pytest.approx(22 / 31)
    # Informational, but pinned so a counter that stopped counting is noticed.
    assert m["fused_ties_in_top_k"] == 4


def test_baseline_is_not_below_m1() -> None:
    current = json.loads(BASELINE.read_text())
    assert {k: v for k, v in current.items() if v < M1_BASELINE[k]} == {}


def test_worst_first_text_order_fails_the_gate(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def worst_first(query: str, filters: Filters, limit: int) -> object:
        # The stub uses the production OR query, so it still exercises the real branch; the
        # worst-first order is what makes shared_ranks give the worst rows the best ranks.
        tsq = hybrid.any_term_tsquery(query)
        rank = func.ts_rank_cd(Item.tsv, tsq).label("score")
        return _filtered(
            select(Item.id, rank, Item.source, Item.external_id)
            .where(Item.tsv.op("@@")(tsq))
            .order_by(rank.asc(), Item.id)
            .limit(limit),
            filters,
        )

    monkeypatch.setattr(hybrid, "text_query", worst_first)
    m = evaluate(db_session, FakeEmbedder(), CORPUS, QUESTIONS, k=5)
    assert m["mrr:text_only:exact_name"] < 16 / 18
    assert gate(m, json.loads(BASELINE.read_text())) != []


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
