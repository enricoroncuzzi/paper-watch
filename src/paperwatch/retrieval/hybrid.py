"""Hybrid search over the corpus: vector similarity and full-text, fused with RRF.

The vector branch runs cosine distance over the partial HNSW index and the text branch ranks
Postgres full-text matches; their ranked id lists are fused with Reciprocal Rank Fusion, which
needs no score calibration between the two systems. Filters shape the candidates inside both
queries, and only admitted items are searchable.
"""

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, Select, Text, cast, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
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
    # Separate so a test can EXPLAIN exactly the statement search runs. A secondary sort key here
    # would stop the planner using the partial HNSW index, so the score is selected next to the id
    # and equal distances are given a shared rank after the fetch, never a second SQL sort key.
    # A cosine distance of 1 is zero similarity (the vectors share nothing), so the row is not a
    # candidate at all; real English embeddings practically never reach it, but the fake one ties.
    # The stable key rides along so the fusion needs no second query to order an equal score.
    distance = Item.embedding.cosine_distance(qvec)
    return _filtered(
        select(Item.id, distance.label("score"), Item.source, Item.external_id)
        .where(distance < 1.0)
        .order_by(distance)
        .limit(limit),
        filters,
    )


def any_term_tsquery(query: str) -> ColumnElement[Any]:
    # plainto_tsquery joins stems with AND ('harness' & 'agent'); OR-ing them makes any shared
    # stem a match and leaves the ordering to ts_rank_cd. The result is cast back to tsquery, not
    # re-parsed by to_tsquery, which would stem the stems a second time and lose words whose stem
    # is not stable (embedding -> embed, then emb, which no tsvector holds).
    anded = func.plainto_tsquery("english", query)
    return cast(func.replace(cast(anded, Text), " & ", " | "), TSQUERY)


def text_query(query: str, filters: Filters, limit: int) -> Select[Any]:
    # Separate like vector_query, so a test inspects this statement instead of a drifting copy.
    # ts_rank_cd ties are common once the query OR-joins its stems, so the score is selected next
    # to the id and equal ranks get a shared rank after the fetch; the stable key only orders rows
    # the rank cannot separate, and this statement sorts anyway, so the GIN index still serves it.
    # The key is COLLATE "C", whose byte order matches Python's code-point order, so SQL and the
    # fusion cut a tie the same way whatever the database locale.
    tsq = any_term_tsquery(query)
    rank = func.ts_rank_cd(Item.tsv, tsq).label("score")
    return _filtered(
        select(Item.id, rank, Item.source, Item.external_id)
        .where(Item.tsv.op("@@")(tsq))
        .order_by(rank.desc(), Item.source.collate("C"), Item.external_id.collate("C"))
        .limit(limit),
        filters,
    )


def shared_ranks(rows: Iterable[tuple[uuid.UUID, float]]) -> dict[uuid.UUID, int]:
    # Competition ranking over rows already ordered best-score-first: equal scores share the best
    # position of their group (1-2-2-4), so a tie gives the same RRF weight whichever order the
    # database or the HNSW index happened to return it in, and no item gains rank from its id.
    ranks: dict[uuid.UUID, int] = {}
    previous: float | None = None
    rank = 0
    for position, (item_id, score) in enumerate(rows, start=1):
        if score != previous:
            rank = position
        ranks[item_id] = rank
        previous = score
    return ranks


def set_search_settings(session: Session, candidates: int) -> None:
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
    # Each branch returns the id, the score and the stable key. `shared_ranks` gives tied scores
    # one rank each, so the fusion never sees the arbitrary order the index or the text sort
    # returned equal rows in, and the stable key rides along for the equal-score sort below.
    vec_rows = list(session.execute(vector_query(qvec, filters, candidates)))
    txt_rows = list(session.execute(text_query(query, filters, candidates)))
    scores: dict[uuid.UUID, float] = {}
    for branch_rows in (vec_rows, txt_rows):
        for item_id, rank in shared_ranks((r.id, r.score) for r in branch_rows).items():
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    # Equal fused scores are ordered by the item's stable (source, external_id) key, never its
    # random uuid, so the output is reproducible.
    keys = {r.id: (r.source, r.external_id) for r in [*vec_rows, *txt_rows]}
    top = sorted(scores, key=lambda i: (-scores[i], *keys[i]))[:k]
    item_rows = {r.id: r for r in session.scalars(select(Item).where(Item.id.in_(top)))}
    return [_hit(item_rows[i], scores[i]) for i in top]


def _hit(row: Item, score: float) -> Hit:
    # Only admitted rows are searchable, and the check constraints give every admitted row a
    # tier and a basis; the assert narrows the column types for mypy and guards that invariant.
    assert row.tier is not None and row.admission_basis is not None
    return Hit(row.id, row.title, row.url, row.tier, row.admission_basis, score, row.text[:300])
