from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from paperwatch.domain import Signals, SourceItem, Tier
from paperwatch.llm.embedder import FakeEmbedder
from paperwatch.retrieval.hybrid import Filters, hybrid_search, vector_query
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
