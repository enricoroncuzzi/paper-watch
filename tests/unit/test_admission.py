"""unit tests for the explicit reliability-tier admission rules."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from paperwatch.admission import admit, load_admission, promote
from paperwatch.domain import Classification, Signals, SourceItem, Tier

CFG = load_admission(Path("config/admission.yaml"))
REL = Classification(profile="agentic-engineering", relevance=0.9, reason="r")


def item(source: str = "arxiv", authors: tuple[str, ...] = ("A. Person",)) -> SourceItem:
    return SourceItem(
        source=source,
        external_id="x",
        title="t",
        authors=authors,
        text="a",
        url="u",
        published_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "signals,expected",
    [
        (Signals(venue="ICLR"), Tier.PEER_REVIEWED),
        (Signals(venue="ICLR 2026"), Tier.PEER_REVIEWED),
        (
            Signals(
                venue="International Conference on Learning Representations",
                venue_aliases=(
                    "International Conference on Learning Representations",
                    "Int Conf Learn Represent",
                    "ICLR",
                ),
            ),
            Tier.PEER_REVIEWED,
        ),
        (
            Signals(
                venue="Neural Information Processing Systems",
                venue_aliases=("Neural Information Processing Systems", "NeurIPS"),
            ),
            Tier.PEER_REVIEWED,
        ),
        (Signals(venue="arXiv.org"), None),
        (Signals(hf_upvotes=10), Tier.VETTED_PREPRINT),
        (Signals(citation_count=5), Tier.VETTED_PREPRINT),
        (Signals(hf_upvotes=9, citation_count=4), None),
        (Signals(venue="Some Workshop"), None),
    ],
)
def test_paper_tiers(signals: Signals, expected: Tier | None) -> None:
    assert admit(item(), REL, signals, CFG) == expected


def test_lab_team_authorship_vets_a_preprint() -> None:
    assert admit(item(authors=("DeepSeek-AI",)), REL, Signals(), CFG) == Tier.VETTED_PREPRINT


def test_feed_items_are_lab_blogs() -> None:
    assert admit(item(source="feed"), REL, Signals(), CFG) == Tier.LAB_BLOG


def test_irrelevant_or_unprofiled_is_rejected() -> None:
    assert admit(item(), Classification(None, 0.9, "r"), Signals(venue="ICLR"), CFG) is None
    assert (
        admit(item(), Classification("agentic-engineering", 0.2, "r"), Signals(venue="ICLR"), CFG)
        is None
    )


def test_promote_never_demotes() -> None:
    assert promote(Tier.PEER_REVIEWED, Tier.VETTED_PREPRINT) == Tier.PEER_REVIEWED
    assert promote(Tier.PEER_REVIEWED, None) == Tier.PEER_REVIEWED
    assert promote(Tier.VETTED_PREPRINT, None) == Tier.VETTED_PREPRINT
    assert promote(Tier.LAB_BLOG, None) == Tier.LAB_BLOG


def test_promote_moves_up() -> None:
    assert promote(None, Tier.VETTED_PREPRINT) == Tier.VETTED_PREPRINT
    assert promote(Tier.VETTED_PREPRINT, Tier.PEER_REVIEWED) == Tier.PEER_REVIEWED
    assert promote(None, None) is None
