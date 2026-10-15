from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import Select, func, select, text
from sqlalchemy.orm import Session

from paperwatch.config import EMBEDDING_DIM
from paperwatch.domain import Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval.hybrid import Filters, _filtered, hybrid_search, vector_query
from paperwatch.store.models import Item
from paperwatch.store.repository import Repository

pytestmark = pytest.mark.integration
E = FakeEmbedder()


def seed(session: Session) -> None:
    repo = Repository(session)
    rows = [
        (
            "1",
            "SWE-bench results for coding agents",
            "Coding agent harness evaluated on SWE-bench.",
            Tier.PEER_REVIEWED,
        ),
        (
            "2",
            "Memory for long-horizon agents",
            "Agents that remember across sessions.",
            Tier.VETTED_PREPRINT,
        ),
        (
            "3",
            "Detecting diffusion images",
            "Synthetic image detection with fingerprints.",
            Tier.VETTED_PREPRINT,
        ),
    ]
    # "body", not "text": text would shadow sqlalchemy.text, which this module also imports.
    for eid, title, body, tier in rows:
        it = SourceItem(
            "arxiv",
            eid,
            title,
            ("A",),
            body,
            f"https://arxiv.org/abs/{eid}",
            datetime(2026, 10, 1, tzinfo=UTC),
        )
        repo.upsert_item(
            it, tier, "agentic-engineering", 0.9, Signals(), E.embed([f"{title}\n{body}"])[0]
        )
    session.flush()


def test_exact_term_ranks_first(db_session: Session) -> None:
    seed(db_session)
    hits = hybrid_search(db_session, E, "SWE-bench", k=3)
    assert hits[0].title.startswith("SWE-bench")


def test_tier_filter_applies(db_session: Session) -> None:
    seed(db_session)
    hits = hybrid_search(
        db_session, E, "agents", k=3, filters=Filters(tiers=frozenset({"peer_reviewed"}))
    )
    assert {h.tier for h in hits} == {"peer_reviewed"}


def test_hybrid_falls_back_to_vector_when_no_text_match(db_session: Session) -> None:
    seed(db_session)
    hits = hybrid_search(db_session, E, "remember sessions zzzunmatchedzzz", k=3)
    assert hits and hits[0].title == "Memory for long-horizon agents"


def test_empty_query_rejected(db_session: Session) -> None:
    with pytest.raises(ValueError):
        hybrid_search(db_session, E, "   ", k=3)


def test_candidates_and_irrelevant_items_are_never_returned(db_session: Session) -> None:
    seed(db_session)
    repo = Repository(db_session)
    for eid, status in (("4", "candidate"), ("5", "irrelevant")):
        it = SourceItem(
            "arxiv",
            eid,
            "SWE-bench coding agents",
            ("A",),
            "SWE-bench agents.",
            f"https://arxiv.org/abs/{eid}",
            datetime(2026, 10, 1, tzinfo=UTC),
        )
        repo.record_classified(it, status, "agentic-engineering", 0.9)
    hits = hybrid_search(db_session, E, "SWE-bench coding agents", k=10)
    assert {h.url.rsplit("/", 1)[1] for h in hits} <= {"1", "2", "3"}


def test_vector_search_uses_partial_hnsw_index(db_session: Session) -> None:
    seed(db_session)
    # A three-row table would be scanned sequentially; disable that for this statement so the
    # plan shows whether the planner can match the query to the partial index at all.
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    stmt = vector_query(E.embed(["agents"])[0], Filters(), 10)
    sql = stmt.compile(db_session.get_bind(), compile_kwargs={"literal_binds": True})
    plan = "\n".join(db_session.execute(text(f"EXPLAIN {sql}")).scalars())
    assert "ix_items_embedding_hnsw" in plan


class QueryEmbedder:
    """Returns one fixed vector for any input, so only the text branch can separate documents."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def admitted(
    session: Session,
    eid: str,
    body: str,
    profile: str,
    published: datetime,
    embedding: list[float] | None = None,
) -> None:
    # A minimal admitted row; the embedding defaults to the FakeEmbedder's vector for its text.
    it = SourceItem("arxiv", eid, body, ("A",), body, f"https://arxiv.org/abs/{eid}", published)
    vec = embedding if embedding is not None else E.embed([f"{body}\n{body}"])[0]
    Repository(session).upsert_item(it, Tier.VETTED_PREPRINT, profile, 0.9, Signals(), vec)


def external_id(session: Session, item_id: UUID) -> str:
    return session.get_one(Item, item_id).external_id


def text_query(tsq: Any, filters: Filters) -> Select[Any]:
    # The statement shape hybrid_search runs for its text branch, so a test can inspect it.
    return _filtered(
        select(Item.id)
        .where(Item.tsv.op("@@")(tsq))
        .order_by(func.ts_rank_cd(Item.tsv, tsq).desc()),
        filters,
    )


def test_vector_branch_fills_candidates_under_a_filter(db_session: Session) -> None:
    # More admitted rows than candidates, and only a fifth match the profile filter, so a scan
    # capped at hnsw.ef_search rows and filtered afterwards would return far fewer than 50.
    at = datetime(2026, 10, 1, tzinfo=UTC)
    for i in range(60):
        admitted(db_session, f"match{i}", f"harness number {i}", "agentic-engineering", at)
    for i in range(240):
        admitted(db_session, f"other{i}", f"harness number {i}", "ml-architectures", at)
    db_session.flush()
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    hits = hybrid_search(
        db_session,
        E,
        "zzzunmatchedzzz",
        k=100,
        candidates=50,
        filters=Filters(profiles=frozenset({"agentic-engineering"})),
    )
    # The text branch matches nothing, so the vector branch alone must fill the 50 candidates.
    assert len(hits) == 50


def test_text_branch_ranks_a_token_match_above_a_vector_neighbour(db_session: Session) -> None:
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    at = datetime(2026, 10, 1, tzinfo=UTC)
    # The token match sits far from the query vector; the neighbour is identical to it but never
    # mentions SWE-bench, so only the text branch can lift the token match above the neighbour.
    admitted(db_session, "token", "SWE-bench harness", "p1", at, embedding=far)
    admitted(db_session, "neighbour", "Memory for agents", "p1", at, embedding=near)
    db_session.flush()
    hits = hybrid_search(db_session, QueryEmbedder(near), "SWE-bench", k=2)
    assert hits[0].url.endswith("/token")


def test_rank_one_in_both_lists_scores_two_over_sixty_one(db_session: Session) -> None:
    seed(db_session)
    hits = hybrid_search(db_session, E, "SWE-bench", k=3)
    # The SWE-bench item is rank 1 in the vector list and the text list: 1/61 + 1/61.
    assert hits[0].score == pytest.approx(2 / 61)
    # An item in both lists is fused once, never returned twice.
    assert len({h.item_id for h in hits}) == len(hits)


def test_filters_restrict_both_branches(db_session: Session) -> None:
    at = datetime(2026, 10, 1, tzinfo=UTC)
    admitted(db_session, "1", "agents harness alpha", "p1", at)
    admitted(db_session, "2", "agents harness beta", "p2", at + timedelta(days=10))
    admitted(db_session, "3", "agents harness gamma", "p1", at - timedelta(days=10))
    db_session.flush()
    query = "agents harness"
    tsq = func.websearch_to_tsquery("english", query)
    qvec = E.embed([query])[0]
    cases = (
        (Filters(profiles=frozenset({"p2"})), {"2"}),
        (Filters(since=at + timedelta(days=1)), {"2"}),
        (Filters(since=at), {"1", "2"}),  # the lower bound is inclusive
    )
    for filters, expected in cases:
        vec_ids = {
            external_id(db_session, i) for i in db_session.scalars(vector_query(qvec, filters, 50))
        }
        txt_ids = {external_id(db_session, i) for i in db_session.scalars(text_query(tsq, filters))}
        hybrid = {
            h.url.rsplit("/", 1)[1]
            for h in hybrid_search(db_session, E, query, k=10, filters=filters)
        }
        assert vec_ids == txt_ids == hybrid == expected


def test_empty_filter_set_matches_nothing(db_session: Session) -> None:
    seed(db_session)
    hits = hybrid_search(db_session, E, "agents", k=3, filters=Filters(tiers=frozenset()))
    assert hits == []


def test_k_below_one_rejected(db_session: Session) -> None:
    for bad_k in (0, -1):
        with pytest.raises(ValueError):
            hybrid_search(db_session, E, "agents", k=bad_k)


def test_candidates_and_rrf_k_bounds_rejected(db_session: Session) -> None:
    with pytest.raises(ValueError):
        hybrid_search(db_session, E, "agents", candidates=0)
    with pytest.raises(ValueError):
        hybrid_search(db_session, E, "agents", rrf_k=-1)
