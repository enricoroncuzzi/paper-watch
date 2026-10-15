"""Offline retrieval eval: recall@k, MRR and nDCG@k on a labelled synthetic corpus.

Tests check that code does what it says; an eval checks that retrieval ranks well enough, and a
committed baseline turns a quality drop into a CI failure. This runs with the deterministic
`FakeEmbedder`, so it is fast, offline and reproducible; real-model numbers come from Task 11b.
"""

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from paperwatch.domain import Signals, SourceItem, Tier
from paperwatch.llm.embedder import Embedder
from paperwatch.retrieval.hybrid import hybrid_search
from paperwatch.store.repository import Repository

# The three metrics the gate compares. The per-kind metrics below are reported for the lesson,
# never gated: with a lexical fake embedder a paraphrase question is expected to rank worse.
OVERALL = ("recall_at_k", "mrr", "ndcg_at_k")
# All eval items use this host, so a returned hit's url maps back to its corpus external_id.
_EVAL_URL = "https://eval.local/{external_id}"


def _load(path: Path) -> list[dict[str, Any]]:
    # JSONL: one object per line, blank lines ignored.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(
    session: Session,
    embedder: Embedder,
    corpus: Path,
    questions: Path,
    k: int = 5,
) -> dict[str, float]:
    repo = Repository(session)
    by_url: dict[str, str] = {}
    # Seed the corpus first: every row is admitted exactly as the pipeline would admit it, so the
    # eval exercises the real write path and the partial HNSW index.
    for row in _load(corpus):
        url = _EVAL_URL.format(external_id=row["external_id"])
        item = SourceItem(
            "eval",
            row["external_id"],
            row["title"],
            ("eval",),
            row["text"],
            url,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
        repo.upsert_item(
            item,
            Tier(row["tier"]),
            row["profile"],
            1.0,
            Signals(),
            embedder.embed([f"{row['title']}\n{row['text']}"])[0],
        )
        by_url[url] = row["external_id"]
    session.flush()

    totals: defaultdict[str, float] = defaultdict(float)
    kind_totals: defaultdict[str, float] = defaultdict(float)
    kind_counts: defaultdict[str, int] = defaultdict(int)
    qs = _load(questions)
    for q in qs:
        relevant = set(q["relevant"])
        # A hit that is not a corpus item (a stale row in a shared database) maps to None and is
        # simply not relevant; it still occupies its rank, as it would in a real deployment.
        ranked = [by_url.get(h.url) for h in hybrid_search(session, embedder, q["question"], k=k)]
        totals["recall_at_k"] += len(relevant.intersection(ranked)) / len(relevant)
        # MRR: the reciprocal of the first relevant hit's rank, or 0 when none is found.
        first = next((i for i, r in enumerate(ranked, 1) if r in relevant), None)
        totals["mrr"] += 1 / first if first else 0.0
        # nDCG@k: discounted credit for relevant hits in the top k, normalised by the ideal order.
        dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked, 1) if r in relevant)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        totals["ndcg_at_k"] += dcg / ideal
        # Per-kind recall and MRR show the full-text risks the lexical fake embedder can measure.
        kind = q["kind"]
        kind_totals[f"recall_at_k:{kind}"] += len(relevant.intersection(ranked)) / len(relevant)
        kind_totals[f"mrr:{kind}"] += 1 / first if first else 0.0
        kind_counts[kind] += 1

    n = len(qs)
    metrics = {name: totals[name] / n for name in OVERALL}
    for kind in sorted(kind_counts):
        metrics[f"recall_at_k:{kind}"] = kind_totals[f"recall_at_k:{kind}"] / kind_counts[kind]
        metrics[f"mrr:{kind}"] = kind_totals[f"mrr:{kind}"] / kind_counts[kind]
    return metrics


def gate(
    metrics: dict[str, float], baseline: dict[str, float], tolerance: float = 0.02
) -> list[str]:
    # A metric may drift down by up to `tolerance` (rounding, a corpus rethink) without failing;
    # the committed baseline holds only the overall metrics, so the gate compares only those.
    return [
        f"{name} dropped from {baseline[name]:.3f} to {metrics[name]:.3f}"
        for name in baseline
        if metrics[name] < baseline[name] - tolerance
    ]
