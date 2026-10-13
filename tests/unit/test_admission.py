"""unit tests for the explicit reliability-tier admission rules."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from paperwatch.admission import Admission, admit, basis_improves, load_admission, promote
from paperwatch.domain import AdmissionBasis, Classification, Signals, SourceItem, Tier

CFG = load_admission(Path("config/admission.yaml"))
REL = Classification(profile="agentic-engineering", relevance=0.9, reason="r")
VENUE = AdmissionBasis.VENUE
HF_UPVOTES = AdmissionBasis.HF_UPVOTES
CITATIONS = AdmissionBasis.CITATIONS
LAB_AUTHOR = AdmissionBasis.LAB_AUTHOR
FEED = AdmissionBasis.FEED


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
        (Signals(venue="ICLR"), Admission(Tier.PEER_REVIEWED, VENUE)),
        (Signals(venue="ICLR 2026"), Admission(Tier.PEER_REVIEWED, VENUE)),
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
            Admission(Tier.PEER_REVIEWED, VENUE),
        ),
        (
            Signals(
                venue="Neural Information Processing Systems",
                venue_aliases=("Neural Information Processing Systems", "NeurIPS"),
            ),
            Admission(Tier.PEER_REVIEWED, VENUE),
        ),
        # S2's ICCV entity carries an "ICCV Workshops" alias. That must not reject a real
        # ICCV paper whose own venue is the conference (venue scout report, 2026-10-13).
        (
            Signals(
                venue="IEEE International Conference on Computer Vision",
                venue_aliases=(
                    "IEEE International Conference on Computer Vision",
                    "ICCV",
                    "ICCV Workshops",
                ),
            ),
            Admission(Tier.PEER_REVIEWED, VENUE),
        ),
        # A paper whose own venue is the workshop stays out, acronym alias or not.
        (
            Signals(
                venue="ICCV Workshops",
                venue_aliases=("IEEE International Conference on Computer Vision", "ICCV"),
            ),
            None,
        ),
        (Signals(venue="arXiv.org"), None),
        (Signals(venue="Some Workshop"), None),
        # Matching is case-sensitive, like the lab author strings.
        (Signals(venue="iclr"), None),
        (Signals(hf_upvotes=10), Admission(Tier.VETTED_PREPRINT, HF_UPVOTES)),
        (Signals(citation_count=5), Admission(Tier.VETTED_PREPRINT, CITATIONS)),
        (Signals(hf_upvotes=9, citation_count=4), None),
    ],
)
def test_paper_tiers(signals: Signals, expected: Admission | None) -> None:
    assert admit(item(), REL, signals, CFG) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Trans. Mach. Learn. Res.",
        "Journal of Machine Learning Research",
        "Journal of machine learning research",
        "IEEE Transactions on Information Forensics and Security",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "USENIX Security Symposium",
        "S&P",
        "ACM Multimedia",
        "Interspeech",
    ],
)
def test_real_s2_venue_strings_are_admitted(name: str) -> None:
    # The scout report's exact S2 strings stand in for the bare acronyms S2 never returns.
    assert admit(item(), REL, Signals(venue=name), CFG) == Admission(Tier.PEER_REVIEWED, VENUE)


def test_lab_team_authorship_vets_a_preprint() -> None:
    assert admit(item(authors=("DeepSeek-AI",)), REL, Signals(), CFG) == Admission(
        Tier.VETTED_PREPRINT, LAB_AUTHOR
    )


def test_community_evidence_outranks_lab_authorship() -> None:
    # A known lab's paper with 10 upvotes records the community basis, not the author one.
    out = admit(item(authors=("DeepSeek-AI",)), REL, Signals(hf_upvotes=10), CFG)
    assert out == Admission(Tier.VETTED_PREPRINT, HF_UPVOTES)


def test_citations_and_upvotes_outrank_lab_authorship() -> None:
    # The rule order is upvotes, then citations, then lab author; both outrank authorship.
    cited = admit(item(authors=("DeepSeek-AI",)), REL, Signals(citation_count=5), CFG)
    assert cited == Admission(Tier.VETTED_PREPRINT, CITATIONS)
    both = admit(item(authors=("DeepSeek-AI",)), REL, Signals(hf_upvotes=10, citation_count=5), CFG)
    assert both == Admission(Tier.VETTED_PREPRINT, HF_UPVOTES)


@pytest.mark.parametrize("name", ["ByteDance Seed", "Gemini Team", "Gemma Team"])
def test_added_lab_author_strings_admit(name: str) -> None:
    assert admit(item(authors=(name,)), REL, Signals(), CFG) == Admission(
        Tier.VETTED_PREPRINT, LAB_AUTHOR
    )


@pytest.mark.parametrize(
    "name", ["Google DeepMind", "Anthropic", "Meta FAIR", "Team GLM", "GLM-4.5 Team"]
)
def test_dropped_lab_author_strings_no_longer_admit(name: str) -> None:
    assert admit(item(authors=(name,)), REL, Signals(), CFG) is None


def test_lab_author_matching_is_case_sensitive() -> None:
    # arXiv's exact casings are "Bytedance Seed" and "ByteDance Seed"; lower case is a miss.
    assert admit(item(authors=("bytedance seed",)), REL, Signals(), CFG) is None


def test_feed_items_are_lab_blogs() -> None:
    assert admit(item(source="feed"), REL, Signals(), CFG) == Admission(Tier.LAB_BLOG, FEED)


def test_feed_from_a_known_lab_stays_a_lab_blog() -> None:
    # The feed check runs before the lab-author check, so OpenAI's blog is tier 3.
    assert admit(item(source="feed", authors=("OpenAI",)), REL, Signals(venue="ICLR"), CFG) == (
        Admission(Tier.LAB_BLOG, FEED)
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
    assert admit(item(), at_min, Signals(venue="ICLR"), CFG) == Admission(Tier.PEER_REVIEWED, VENUE)


def test_venue_takes_precedence_over_other_signals() -> None:
    assert admit(
        item(), REL, Signals(venue="ICLR", hf_upvotes=10, citation_count=5), CFG
    ) == Admission(Tier.PEER_REVIEWED, VENUE)


def test_cited_preprint_is_vetted_even_with_a_preprint_venue() -> None:
    assert admit(item(), REL, Signals(venue="arXiv.org", citation_count=5), CFG) == Admission(
        Tier.VETTED_PREPRINT, CITATIONS
    )
    assert admit(item(), REL, Signals(venue="arXiv.org", citation_count=4), CFG) is None


def test_basis_improves_only_to_community_evidence() -> None:
    # Lab authorship is the weakest basis; community evidence for the same tier upgrades it.
    assert basis_improves(LAB_AUTHOR, HF_UPVOTES)
    assert basis_improves(LAB_AUTHOR, CITATIONS)
    assert basis_improves(None, HF_UPVOTES)
    assert basis_improves(None, CITATIONS)


@pytest.mark.parametrize(
    "current,proposed",
    [
        (LAB_AUTHOR, LAB_AUTHOR),
        (LAB_AUTHOR, VENUE),
        (LAB_AUTHOR, FEED),
        (HF_UPVOTES, CITATIONS),
        (CITATIONS, HF_UPVOTES),
        (HF_UPVOTES, LAB_AUTHOR),
        (CITATIONS, LAB_AUTHOR),
        (VENUE, HF_UPVOTES),
        (FEED, CITATIONS),
        (None, VENUE),
        (None, LAB_AUTHOR),
        (None, FEED),
    ],
)
def test_basis_improves_is_false_for_every_other_pair(
    current: AdmissionBasis | None, proposed: AdmissionBasis
) -> None:
    assert not basis_improves(current, proposed)


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
    assert len(CFG.accepted_venues) == 33
    assert len(CFG.lab_authors) == 16
    assert CFG.min_relevance == 0.5
    assert CFG.min_hf_upvotes == 10
    assert CFG.min_citations == 5


def test_load_admission_rejects_a_string_venue_list(tmp_path: Path) -> None:
    # A bare string would otherwise become a frozenset of single characters.
    path = _write_admission(tmp_path, _admission_yaml(accepted_venues="ICLR"))
    with pytest.raises(ValueError, match=r"(?s)invalid admission config.*accepted_venues"):
        load_admission(path)


def test_load_admission_rejects_a_boolean_threshold(tmp_path: Path) -> None:
    # pydantic would coerce True to 1 for a non-strict int, silently lowering the bar.
    path = _write_admission(tmp_path, _admission_yaml(min_hf_upvotes=True))
    with pytest.raises(ValueError, match=r"(?s)invalid admission config.*min_hf_upvotes"):
        load_admission(path)


def test_load_admission_rejects_a_negative_threshold(tmp_path: Path) -> None:
    path = _write_admission(tmp_path, _admission_yaml(min_citations=-1))
    with pytest.raises(ValueError, match=r"(?s)invalid admission config.*min_citations"):
        load_admission(path)


@pytest.mark.parametrize("value", [True, False, "0.5"])
def test_load_admission_rejects_a_non_strict_relevance(tmp_path: Path, value: object) -> None:
    # Lax mode turns true into 1.0, false into 0.0 (admitting everything) and "0.5" into 0.5.
    path = _write_admission(tmp_path, _admission_yaml(min_relevance=value))
    with pytest.raises(ValueError, match=r"(?s)invalid admission config.*min_relevance"):
        load_admission(path)


def test_load_admission_accepts_an_integer_relevance(tmp_path: Path) -> None:
    path = _write_admission(tmp_path, _admission_yaml(min_relevance=1))
    assert load_admission(path).min_relevance == 1.0


def test_load_admission_wraps_invalid_yaml(tmp_path: Path) -> None:
    path = _write_admission(tmp_path, "min_relevance: [")
    with pytest.raises(ValueError) as excinfo:
        load_admission(path)
    assert "invalid admission config" in str(excinfo.value) and str(path) in str(excinfo.value)


def test_load_admission_wraps_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    with pytest.raises(ValueError) as excinfo:
        load_admission(path)
    assert "invalid admission config" in str(excinfo.value) and str(path) in str(excinfo.value)


def test_load_admission_wraps_a_non_utf8_file(tmp_path: Path) -> None:
    # UnicodeDecodeError is a ValueError, not an OSError, so the path must be wrapped.
    path = tmp_path / "admission.yaml"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError) as excinfo:
        load_admission(path)
    assert "invalid admission config" in str(excinfo.value) and str(path) in str(excinfo.value)
