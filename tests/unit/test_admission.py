"""unit tests for the explicit reliability-tier admission rules."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

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


def _write_admission(tmp_path: Path, body: str) -> Path:
    # Write a small admission file so validation failures can be exercised offline.
    path = tmp_path / "admission.yaml"
    path.write_text(body)
    return path


def _admission_yaml(**overrides: object) -> str:
    # A valid file with one field replaced, so each test isolates one bad value.
    base: dict[str, object] = {
        "min_relevance": 0.5,
        "min_hf_upvotes": 10,
        "min_citations": 5,
        "accepted_venues": ["ICLR"],
        "lab_authors": ["DeepSeek-AI"],
    }
    base.update(overrides)
    return yaml.safe_dump(base)


@pytest.mark.parametrize(
    "signals,expected",
    [
        (Signals(venue="ICLR"), Tier.PEER_REVIEWED),
        (Signals(venue="ICLR 2026"), Tier.PEER_REVIEWED),
        # A workshop title is not the parent conference, even with an acronym alias.
        (Signals(venue="ICLR Workshop"), None),
        (Signals(venue="CVPR Workshops"), None),
        (Signals(venue="NeurIPS 2024 Workshop on Agents"), None),
        (Signals(venue="ICLR Workshop", venue_aliases=("ICLR",)), None),
        # "ACL Anthology" is not the ACL conference.
        (Signals(venue="ACL Anthology"), None),
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
        (Signals(venue="Some Workshop"), None),
        # Matching is case-sensitive, like the lab author strings.
        (Signals(venue="iclr"), None),
        (Signals(hf_upvotes=10), Tier.VETTED_PREPRINT),
        (Signals(citation_count=5), Tier.VETTED_PREPRINT),
        (Signals(hf_upvotes=9, citation_count=4), None),
    ],
)
def test_paper_tiers(signals: Signals, expected: Tier | None) -> None:
    assert admit(item(), REL, signals, CFG) == expected


def test_lab_team_authorship_vets_a_preprint() -> None:
    assert admit(item(authors=("DeepSeek-AI",)), REL, Signals(), CFG) == Tier.VETTED_PREPRINT


def test_lab_author_matching_is_case_sensitive() -> None:
    # arXiv's exact casing is "Bytedance Seed"; the CamelCase spelling is a near miss.
    assert admit(item(authors=("ByteDance Seed",)), REL, Signals(), CFG) is None


def test_feed_items_are_lab_blogs() -> None:
    assert admit(item(source="feed"), REL, Signals(), CFG) == Tier.LAB_BLOG


def test_feed_from_a_known_lab_stays_a_lab_blog() -> None:
    # The feed check runs before the lab-author check, so OpenAI's blog is tier 3.
    assert (
        admit(item(source="feed", authors=("OpenAI",)), REL, Signals(venue="ICLR"), CFG)
        == Tier.LAB_BLOG
    )


def test_irrelevant_or_unprofiled_is_rejected() -> None:
    assert admit(item(), Classification(None, 0.9, "r"), Signals(venue="ICLR"), CFG) is None
    assert (
        admit(item(), Classification("agentic-engineering", 0.2, "r"), Signals(venue="ICLR"), CFG)
        is None
    )


def test_irrelevant_feed_is_rejected() -> None:
    assert (
        admit(item(source="feed"), Classification("agentic-engineering", 0.2, "r"), Signals(), CFG)
        is None
    )


def test_relevance_boundary_is_inclusive() -> None:
    # Task 9 stores a candidate at relevance == min_relevance, so this must admit.
    at_min = Classification("agentic-engineering", 0.5, "r")
    assert admit(item(), at_min, Signals(venue="ICLR"), CFG) == Tier.PEER_REVIEWED


def test_venue_takes_precedence_over_other_signals() -> None:
    assert (
        admit(item(), REL, Signals(venue="ICLR", hf_upvotes=10, citation_count=5), CFG)
        == Tier.PEER_REVIEWED
    )


def test_cited_preprint_is_vetted_even_with_a_preprint_venue() -> None:
    assert (
        admit(item(), REL, Signals(venue="arXiv.org", citation_count=5), CFG)
        == Tier.VETTED_PREPRINT
    )
    assert admit(item(), REL, Signals(venue="arXiv.org", citation_count=4), CFG) is None


def test_promote_never_demotes() -> None:
    assert promote(Tier.PEER_REVIEWED, Tier.VETTED_PREPRINT) == Tier.PEER_REVIEWED
    assert promote(Tier.PEER_REVIEWED, None) == Tier.PEER_REVIEWED
    assert promote(Tier.VETTED_PREPRINT, None) == Tier.VETTED_PREPRINT
    assert promote(Tier.LAB_BLOG, None) == Tier.LAB_BLOG
    # A lab blog is never ranked against paper tiers; the stored tier stays.
    assert promote(Tier.PEER_REVIEWED, Tier.LAB_BLOG) == Tier.PEER_REVIEWED
    assert promote(Tier.LAB_BLOG, Tier.PEER_REVIEWED) == Tier.LAB_BLOG
    assert promote(Tier.VETTED_PREPRINT, Tier.LAB_BLOG) == Tier.VETTED_PREPRINT
    assert promote(Tier.LAB_BLOG, Tier.VETTED_PREPRINT) == Tier.LAB_BLOG


def test_promote_moves_up() -> None:
    assert promote(None, Tier.VETTED_PREPRINT) == Tier.VETTED_PREPRINT
    assert promote(None, Tier.PEER_REVIEWED) == Tier.PEER_REVIEWED
    assert promote(None, Tier.LAB_BLOG) == Tier.LAB_BLOG
    assert promote(Tier.VETTED_PREPRINT, Tier.PEER_REVIEWED) == Tier.PEER_REVIEWED
    assert promote(None, None) is None


def test_load_admission_reads_the_configured_thresholds() -> None:
    assert len(CFG.accepted_venues) == 14
    assert len(CFG.lab_authors) == 18
    assert CFG.min_relevance == 0.5
    assert CFG.min_hf_upvotes == 10
    assert CFG.min_citations == 5


def test_load_admission_rejects_a_string_venue_list(tmp_path: Path) -> None:
    # A bare string would otherwise become a frozenset of single characters.
    path = _write_admission(tmp_path, _admission_yaml(accepted_venues="ICLR"))
    with pytest.raises(ValueError, match="admission"):
        load_admission(path)


def test_load_admission_rejects_a_boolean_threshold(tmp_path: Path) -> None:
    # pydantic would coerce True to 1 for a non-strict int, silently lowering the bar.
    path = _write_admission(tmp_path, _admission_yaml(min_hf_upvotes=True))
    with pytest.raises(ValueError, match="admission"):
        load_admission(path)


def test_load_admission_rejects_a_negative_threshold(tmp_path: Path) -> None:
    path = _write_admission(tmp_path, _admission_yaml(min_citations=-1))
    with pytest.raises(ValueError, match="admission"):
        load_admission(path)
