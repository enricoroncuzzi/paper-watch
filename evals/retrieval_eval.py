"""Offline retrieval eval: recall@k, MRR and nDCG@k on a labelled synthetic corpus.

Tests check that code does what it says; an eval checks that retrieval ranks well enough, and a
committed baseline turns a quality drop into a CI failure. This runs with the deterministic
`FakeEmbedder`, so it is fast, offline and reproducible; real-model numbers come from the
corpus-size checkpoints on real data, not from the fake embedder.
"""

import json
import math
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from paperwatch.domain import Signals, SourceItem, Tier
from paperwatch.llm.embedder import Embedder
from paperwatch.retrieval import hybrid
from paperwatch.store.models import Item
from paperwatch.store.repository import Repository

# The three overall hybrid metrics, always gated.
HYBRID = ("recall_at_k", "mrr", "ndcg_at_k")
# The whole gate: the overall metrics plus the text branch's own recall and MRR on exact-name
# questions. The text-only metrics are what catch a broken text branch (they fall to 0) and the
# ordering regression the Task 11 review probed (a worst-first sort drops the MRR).
GATED = (*HYBRID, "recall_at_k:text_only:exact_name", "mrr:text_only:exact_name")
# The branch-only labels in metric keys; hybrid metrics carry no branch suffix.
BRANCHES = ("text_only", "vector_only")
# All eval items use this host, so a returned hit's url maps back to its corpus external_id.
_EVAL_URL = "https://eval.local/{external_id}"


class DatabaseNotEmpty(RuntimeError):
    """Raised when the database holds rows the eval did not seed.

    Those rows would compete with the eval corpus and pollute the ranking, so the eval refuses
    them instead of reporting a false regression.
    """


def _load(path: Path) -> list[dict[str, Any]]:
    # JSONL: one object per line, blank lines ignored.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _recall(ranked: list[str | None], relevant: set[str]) -> float:
    return len(relevant.intersection(ranked)) / len(relevant)


def _reciprocal_rank(ranked: list[str | None], relevant: set[str]) -> float:
    # The reciprocal of the first relevant hit's rank, or 0 when none is found.
    first = next((i for i, r in enumerate(ranked, 1) if r in relevant), None)
    return 1 / first if first else 0.0


def _ndcg(ranked: list[str | None], relevant: set[str], k: int) -> float:
    # Discounted credit for relevant hits in the top k, normalised by the ideal ordering, which
    # is truncated at the number of relevant items or k, whichever is smaller.
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked, 1) if r in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evaluate(
    session: Session,
    embedder: Embedder,
    corpus: Path,
    questions: Path,
    k: int = 5,
) -> dict[str, float]:
    # A real item would compete with the eval corpus in the ranking, so refuse a populated store
    # instead of reporting a false regression. CI runs against its own empty service.
    foreign = session.scalar(select(func.count()).select_from(Item).where(Item.source != "eval"))
    if foreign:
        raise DatabaseNotEmpty(
            f"the eval needs an empty database or the CI service; found {foreign} non-eval items"
        )
    repo = Repository(session)
    by_url: dict[str, str] = {}
    by_item: dict[uuid.UUID, str] = {}
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
        item_id = repo.upsert_item(
            item,
            Tier(row["tier"]),
            row["profile"],
            1.0,
            Signals(),
            embedder.embed([f"{row['title']}\n{row['text']}"])[0],
        )
        by_url[url] = row["external_id"]
        by_item[item_id] = row["external_id"]
    session.flush()

    totals: defaultdict[str, float] = defaultdict(float)
    kind_totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    nonempty: defaultdict[str, int] = defaultdict(int)
    qs = _load(questions)
    for q in qs:
        relevant = set(q["relevant"])
        kind = q["kind"]
        # Request one extra hybrid hit: a tie at the k-th place would make the top k depend on the
        # fusion's random-id tie-break, so the eval refuses to score a question that ties there.
        hits = hybrid.hybrid_search(session, embedder, q["question"], k=k + 1)
        scores = [round(h.score, 12) for h in hits]
        if len(scores) != len(set(scores)):
            raise ValueError(
                f"fused scores tie within the top {k + 1} for {q['question']!r}; "
                "change the question or the data so the ranking is id-independent"
            )
        # The branch lists are the same public queries the fusion uses. The vector branch is run
        # under the same transaction-local settings hybrid_search set (shared helper), not by
        # luck of call order.
        qvec = embedder.embed([q["question"]])[0]
        hybrid.set_search_settings(session, k)
        vector_ranked = [
            by_item.get(i) for i in session.scalars(hybrid.vector_query(qvec, hybrid.Filters(), k))
        ]
        branch_lists = {
            "hybrid": [by_url.get(h.url) for h in hits[:k]],
            "text_only": [
                by_item.get(i)
                for i in session.scalars(hybrid.text_query(q["question"], hybrid.Filters(), k))
            ],
            "vector_only": vector_ranked,
        }
        for branch, ranked in branch_lists.items():
            # Hybrid keys carry no suffix; a branch-only key names the branch, then the kind.
            suffix = "" if branch == "hybrid" else f":{branch}"
            totals[f"recall_at_k{suffix}"] += _recall(ranked, relevant)
            totals[f"mrr{suffix}"] += _reciprocal_rank(ranked, relevant)
            kind_totals[f"recall_at_k{suffix}:{kind}"] += _recall(ranked, relevant)
            kind_totals[f"mrr{suffix}:{kind}"] += _reciprocal_rank(ranked, relevant)
        totals["ndcg_at_k"] += _ndcg(branch_lists["hybrid"], relevant, k)
        if branch_lists["text_only"]:
            nonempty["text_nonempty"] += 1
            nonempty[f"text_nonempty:{kind}"] += 1
        counts[kind] += 1

    n = len(qs)
    metrics = {
        "recall_at_k": totals["recall_at_k"] / n,
        "mrr": totals["mrr"] / n,
        "ndcg_at_k": totals["ndcg_at_k"] / n,
        "text_nonempty": nonempty["text_nonempty"] / n,
    }
    for branch in BRANCHES:
        metrics[f"recall_at_k:{branch}"] = totals[f"recall_at_k:{branch}"] / n
        metrics[f"mrr:{branch}"] = totals[f"mrr:{branch}"] / n
    for kind in sorted(counts):
        metrics[f"recall_at_k:{kind}"] = kind_totals[f"recall_at_k:{kind}"] / counts[kind]
        metrics[f"mrr:{kind}"] = kind_totals[f"mrr:{kind}"] / counts[kind]
        metrics[f"text_nonempty:{kind}"] = nonempty[f"text_nonempty:{kind}"] / counts[kind]
        for branch in BRANCHES:
            metrics[f"recall_at_k:{branch}:{kind}"] = (
                kind_totals[f"recall_at_k:{branch}:{kind}"] / counts[kind]
            )
            metrics[f"mrr:{branch}:{kind}"] = kind_totals[f"mrr:{branch}:{kind}"] / counts[kind]
    return metrics


def gate(
    metrics: dict[str, float], baseline: dict[str, float], tolerance: float = 0.01
) -> list[str]:
    # The committed baseline must carry exactly the gated keys, so a metric cannot be added to (or
    # silently dropped from) the gate without a visible diff, and a non-finite metric is refused
    # rather than compared (every comparison with NaN is False, so it would pass by accident).
    if set(baseline) != set(GATED):
        raise ValueError(
            f"baseline must hold exactly the gated keys {GATED}, got {tuple(sorted(baseline))}"
        )
    missing = [name for name in GATED if name not in metrics]
    if missing:
        raise ValueError(f"metrics are missing gated keys: {missing}")
    for name in GATED:
        if not math.isfinite(metrics[name]):
            raise ValueError(f"{name} is not finite: {metrics[name]!r}")
    # A metric may drift down by up to `tolerance` without failing. With 29+ questions one rank
    # slip on one question costs about 0.017 MRR/nDCG and one lost relevant hit about 0.017 recall,
    # so 0.01 fails either; the embedder and the data are deterministic on a fresh database, so the
    # sharper tolerance cannot flake in CI.
    return [
        f"{name} dropped from {baseline[name]:.3f} to {metrics[name]:.3f}"
        for name in GATED
        if metrics[name] < baseline[name] - tolerance
    ]
