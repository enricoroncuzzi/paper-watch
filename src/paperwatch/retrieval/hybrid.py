"""Hybrid search over the corpus: vector similarity and full-text, fused with RRF.

The vector branch runs cosine distance over the partial HNSW index and the text branch ranks
Postgres full-text matches; their ranked id lists are fused with Reciprocal Rank Fusion, which
needs no score calibration between the two systems. Filters shape the candidates inside both
queries, and only admitted items are searchable.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select, text
from sqlalchemy.orm import Session

from paperwatch.llm.embedder import Embedder
from paperwatch.store.models import ADMITTED_SQL, Item


@dataclass(frozen=True)
class Filters:
    """Narrow the candidates of both branches. None means no filter; an empty

    frozenset matches nothing (the filter is applied, and no value is in it).
    """

    tiers: frozenset[str] | None = None
    profiles: frozenset[str] | None = None
    since: datetime | None = None


@dataclass(frozen=True)
class Hit:
    item_id: uuid.UUID
    title: str
    url: str
    tier: str
    # The admission rule behind the tier (AdmissionBasis value), e.g. "lab_author".
    basis: str
    score: float
    snippet: str


def _filtered(stmt: Select[Any], f: Filters) -> Select[Any]:
    # A literal predicate, identical to the partial index's, keeps a constant the planner
    # can match to the index (a bound parameter is not provably equal under a cached plan).
    stmt = stmt.where(text(f"items.{ADMITTED_SQL}"))
    # `is not None`, not truthiness: an empty frozenset is a filter that matches nothing.
    if f.tiers is not None:
        stmt = stmt.where(Item.tier.in_(f.tiers))
    if f.profiles is not None:
        stmt = stmt.where(Item.profile.in_(f.profiles))
    if f.since is not None:
        stmt = stmt.where(Item.published_at >= f.since)
    return stmt


_NO_FILTERS = Filters()


def vector_query(qvec: list[float], filters: Filters, limit: int) -> Select[Any]:
    # Separate so a test can EXPLAIN exactly the statement search runs. A secondary sort key
    # here would stop the planner using the partial HNSW index; ties inside the vector branch
    # follow the index order, and the fusion sort breaks only equal fused scores by id.
    return _filtered(
        select(Item.id).order_by(Item.embedding.cosine_distance(qvec)).limit(limit), filters
    )


def text_query(query: str, filters: Filters, limit: int) -> Select[Any]:
    # Separate like vector_query, so a test inspects this statement instead of a drifting copy.
    # The id tie-break makes equal ts_rank order deterministic; this statement sorts anyway, so
    # it never affects the GIN index the full-text predicate uses.
    tsq = func.websearch_to_tsquery("english", query)
    return _filtered(
        select(Item.id)
        .where(Item.tsv.op("@@")(tsq))
        .order_by(func.ts_rank_cd(Item.tsv, tsq).desc(), Item.id)
        .limit(limit),
        filters,
    )


def set_search_settings(session: Session, candidates: int = 0) -> None:
    # An HNSW scan returns at most hnsw.ef_search rows and applies other filters after the scan, so
    # a large or filtered candidate set would under-fill. Transaction-local settings widen the scan
    # to `candidates` (never below 40) and let iterative_scan fetch more when a filter rejects a
    # row; strict_order keeps the exact distance order that RRF's ranks rely on. Shared so the
    # eval's vector-only branch searches under the same settings as the hybrid path.
    session.execute(
        text(
            "SELECT set_config('hnsw.ef_search', :n, true), "
            "set_config('hnsw.iterative_scan', 'strict_order', true)"
        ),
        {"n": str(max(candidates, 40))},
    )


def hybrid_search(
    session: Session,
    embedder: Embedder,
    query: str,
    k: int = 10,
    filters: Filters = _NO_FILTERS,
    candidates: int = 50,
    rrf_k: int = 60,
) -> list[Hit]:
    if not query.strip():
        raise ValueError("query must not be blank")
    if k < 1:
        raise ValueError("k must be >= 1")
    if candidates < 1:
        raise ValueError("candidates must be >= 1")
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    # An HNSW scan returns at most hnsw.ef_search rows and applies other filters after the
    # scan, so a large or filtered candidate set would under-fill; this sets it wide enough.
    set_search_settings(session, candidates)
    qvec = embedder.embed([query])[0]
    vec_ids = list(session.scalars(vector_query(qvec, filters, candidates)))
    txt_ids = list(session.scalars(text_query(query, filters, candidates)))
    scores: dict[uuid.UUID, float] = {}
    for ranked in (vec_ids, txt_ids):
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    # Item.id breaks equal-score ties, so the fused order is deterministic too.
    top = sorted(scores, key=lambda i: (-scores[i], str(i)))[:k]
    rows = {r.id: r for r in session.scalars(select(Item).where(Item.id.in_(top)))}
    return [_hit(rows[i], scores[i]) for i in top]


def _hit(row: Item, score: float) -> Hit:
    # Only admitted rows are searchable, and the check constraints give every admitted row a
    # tier and a basis; the assert narrows the column types for mypy and guards that invariant.
    assert row.tier is not None and row.admission_basis is not None
    return Hit(row.id, row.title, row.url, row.tier, row.admission_basis, score, row.text[:300])
