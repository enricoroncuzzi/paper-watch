from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from paperwatch.config import EMBEDDING_DIM
from paperwatch.domain import AdmissionBasis, Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval.hybrid import (
    Filters,
    hybrid_search,
    shared_ranks,
    text_query,
    vector_query,
)
from paperwatch.store.db import make_session_factory
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


def admitted_with_id(
    session: Session, item_id: UUID, eid: str, body: str, embedding: list[float]
) -> None:
    # A chosen id, so a tie-order assertion never depends on uuid4 luck. This writes the same
    # admitted row the repository writes, without a tier-history row.
    session.add(
        Item(
            id=item_id,
            source="arxiv",
            external_id=eid,
            status="admitted",
            tier=Tier.VETTED_PREPRINT.value,
            admission_basis=AdmissionBasis.SEED.value,
            profile="p1",
            relevance=0.9,
            title=body,
            authors=["A"],
            text=body,
            url=f"https://arxiv.org/abs/{eid}",
            published_at=datetime(2026, 10, 1, tzinfo=UTC),
            licence="unknown",
            embedding=embedding,
        )
    )


def test_vector_branch_fills_candidates_under_a_filter(db_session: Session) -> None:
    # More admitted rows than candidates, and only a fifth match the profile filter, so a scan
    # capped at hnsw.ef_search rows and filtered afterwards would return far fewer than 50. A
    # bitmap scan plus sort would return all 50 even with the fix removed, so disable that too:
    # the EXPLAIN then fails loudly if the planner drifts away from the HNSW index.
    at = datetime(2026, 10, 1, tzinfo=UTC)
    for i in range(60):
        admitted(db_session, f"match{i}", f"harness number {i}", "agentic-engineering", at)
    for i in range(240):
        admitted(db_session, f"other{i}", f"harness number {i}", "ml-architectures", at)
    db_session.flush()
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    db_session.execute(text("SET LOCAL enable_bitmapscan = off"))
    filters = Filters(profiles=frozenset({"agentic-engineering"}))
    sql = vector_query(E.embed(["zzzunmatchedzzz"])[0], filters, 50).compile(
        db_session.get_bind(), compile_kwargs={"literal_binds": True}
    )
    plan = "\n".join(db_session.execute(text(f"EXPLAIN {sql}")).scalars())
    assert "ix_items_embedding_hnsw" in plan  # the scan this test is about
    hits = hybrid_search(db_session, E, "zzzunmatchedzzz", k=100, candidates=50, filters=filters)
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
        txt_ids = {
            external_id(db_session, i) for i in db_session.scalars(text_query(query, filters, 50))
        }
        hybrid = {
            h.url.rsplit("/", 1)[1]
            for h in hybrid_search(db_session, E, query, k=10, filters=filters)
        }
        assert vec_ids == txt_ids == hybrid == expected


@pytest.mark.parametrize("key", ["tiers", "profiles"])
def test_empty_filter_set_matches_nothing(db_session: Session, key: str) -> None:
    seed(db_session)
    filters = Filters(**{key: frozenset()})
    assert hybrid_search(db_session, E, "agents", k=3, filters=filters) == []


def test_text_ties_come_back_in_key_order(db_session: Session) -> None:
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    # Three identical bodies tie on ts_rank. Their external ids order (tx, ty, tz) differently
    # from both insertion order and id order, so only the text branch's stable-key tie-break can
    # produce the asserted order.
    for item_id, eid in ((UUID(int=3), "tx"), (UUID(int=1), "tz"), (UUID(int=2), "ty")):
        admitted_with_id(db_session, item_id, eid, "tietoken body", far)
    # Nearer items fill the three candidate slots, so the tied rows are text-only and their
    # fused order is exactly the text branch's order.
    for i in range(3):
        admitted_with_id(db_session, UUID(int=100 + i), f"v{i}", "vector body", near)
    db_session.flush()
    hits = hybrid_search(db_session, QueryEmbedder(near), "tietoken", k=6, candidates=3)
    text_only = [
        external_id(db_session, h.item_id)
        for h in hits
        if h.item_id in {UUID(int=1), UUID(int=2), UUID(int=3)}
    ]
    assert text_only == ["tx", "ty", "tz"]


def test_equal_fused_scores_come_back_in_key_order(db_session: Session) -> None:
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    # A vector-only item and a text-only item both rank first in their list, so they tie at
    # 1/61. Their external ids order ("aaa" before "zzz") differs from their ids, so only the
    # fusion sort's stable-key tie-break can produce the asserted order.
    admitted_with_id(db_session, UUID(int=20), "aaa", "vector body", near)
    admitted_with_id(db_session, UUID(int=10), "zzz", "tietoken body", far)
    db_session.flush()
    hits = hybrid_search(db_session, QueryEmbedder(near), "tietoken", k=2, candidates=1)
    assert [external_id(db_session, h.item_id) for h in hits] == ["aaa", "zzz"]


def test_shared_ranks_skip_after_a_tie() -> None:
    a, b, c, d = UUID(int=1), UUID(int=2), UUID(int=3), UUID(int=4)
    # Competition ranking: the two 0.2 rows share rank 2, so the next row is rank 4, not rank 3.
    ranks = shared_ranks([(a, 0.1), (b, 0.2), (c, 0.2), (d, 0.3)])
    assert [ranks[i] for i in (a, b, c, d)] == [1, 2, 2, 4]


def test_equal_distances_share_a_vector_rank(db_session: Session) -> None:
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    # Identical vectors, and the query matches neither body, so only the vector branch ranks them;
    # equal distances share the top rank whatever their ids and external ids.
    admitted_with_id(db_session, UUID(int=5), "zzz", "vector body", near)
    admitted_with_id(db_session, UUID(int=6), "aaa", "vector body", near)
    db_session.flush()
    hits = hybrid_search(db_session, QueryEmbedder(near), "tietoken", k=2, candidates=2)
    scores = {h.item_id: h.score for h in hits}
    assert scores[UUID(int=5)] == scores[UUID(int=6)] == pytest.approx(1 / 61)


def test_equal_text_ranks_share_a_rank(db_session: Session) -> None:
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    # The tied rows are far from the query vector and the two near rows fill the two candidate
    # slots, so the tied rows are text-only and share the text branch's top rank.
    admitted_with_id(db_session, UUID(int=1), "n1", "vector body", near)
    admitted_with_id(db_session, UUID(int=2), "n2", "vector body", near)
    admitted_with_id(db_session, UUID(int=7), "zzz", "tietoken body", far)
    admitted_with_id(db_session, UUID(int=8), "aaa", "tietoken body", far)
    db_session.flush()
    hits = hybrid_search(db_session, QueryEmbedder(near), "tietoken", k=4, candidates=2)
    scores = {h.item_id: h.score for h in hits}
    assert scores[UUID(int=7)] == scores[UUID(int=8)] == pytest.approx(1 / 61)


def test_search_settings_are_transaction_local(engine: Engine) -> None:
    # A fresh session on a pooled connection, not the savepoint fixture: after the commit
    # ends the transaction, the settings must be at their defaults, so they cannot leak.
    session = make_session_factory(engine)()
    try:
        hybrid_search(session, E, "agents", k=1)
        session.commit()
        iterative = session.scalar(text("SHOW hnsw.iterative_scan"))
        ef = session.scalar(text("SHOW hnsw.ef_search"))
    finally:
        session.close()
    assert iterative == "off"
    assert ef == "40"


def test_k_below_one_rejected(db_session: Session) -> None:
    for bad_k in (0, -1):
        with pytest.raises(ValueError):
            hybrid_search(db_session, E, "agents", k=bad_k)


def test_candidates_and_rrf_k_bounds_rejected(db_session: Session) -> None:
    with pytest.raises(ValueError):
        hybrid_search(db_session, E, "agents", candidates=0)
    with pytest.raises(ValueError):
        hybrid_search(db_session, E, "agents", rrf_k=-1)


def ids_for(session: Session, query: str) -> list[UUID]:
    return list(session.scalars(text_query(query, Filters(), 10)))


def item_id(session: Session, eid: str) -> UUID:
    return session.scalars(select(Item.id).where(Item.external_id == eid)).one()


def test_text_branch_matches_a_paraphrase_with_one_shared_term(db_session: Session) -> None:
    seed(db_session)
    # Shares "harness" and "agents" with item 1 but not "programming": AND finds nothing.
    assert item_id(db_session, "1") in ids_for(
        db_session, "which harness evaluates programming agents"
    )


def test_text_branch_ranks_more_shared_terms_first(db_session: Session) -> None:
    published = datetime(2026, 10, 1, tzinfo=UTC)
    admitted(
        db_session, "two", "sandboxed tool execution for agents", "agentic-engineering", published
    )
    admitted(
        db_session, "one", "tool catalogues for spreadsheets", "agentic-engineering", published
    )
    db_session.flush()
    ranked = ids_for(db_session, "sandboxed tool")
    assert ranked[:2] == [item_id(db_session, "two"), item_id(db_session, "one")]


def test_stopword_only_query_matches_nothing(db_session: Session) -> None:
    seed(db_session)
    assert ids_for(db_session, "the of and") == []


def test_hyphenated_name_still_matches(db_session: Session) -> None:
    seed(db_session)
    assert ids_for(db_session, "SWE-bench")[0] == item_id(db_session, "1")
