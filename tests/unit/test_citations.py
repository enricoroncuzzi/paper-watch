from datetime import UTC, datetime
from uuid import uuid4

from paperwatch.agent.citations import NO_SUPPORT, check_answer
from paperwatch.agent.sources import SourceRegistry
from paperwatch.store.repository import ItemDetail


def detail(title: str, tier: str) -> ItemDetail:
    return ItemDetail(
        uuid4(),
        title,
        ("A",),
        "text",
        f"https://x/{title}",
        tier,
        "seed",
        datetime(2026, 10, 1, tzinfo=UTC),
        None,
        0,
        None,
        "agentic-engineering",
    )


def registry() -> SourceRegistry:
    reg = SourceRegistry()
    reg.register(detail("preprint", "vetted_preprint"))  # S1
    reg.register(detail("venue", "peer_reviewed"))  # S2
    return reg


def test_valid_citations_are_listed_once_in_order() -> None:
    a = check_answer(
        "Agents need memory to plan over long tasks [S2]. "
        "Memory also helps tool use across many steps [S1][S2].",
        registry(),
    )
    assert [c.alias for c in a.citations] == ["S2", "S1"]
    assert a.flags == () and a.supported


def test_grouped_markers_are_parsed() -> None:
    a = check_answer("Both papers measure long-horizon agent memory directly [S1, S2].", registry())
    assert [c.alias for c in a.citations] == ["S1", "S2"] and a.supported


def test_marker_after_the_full_stop_belongs_to_its_sentence() -> None:
    a = check_answer("Both papers measure long-horizon agent memory directly. [S1]", registry())
    assert a.flags == () and a.supported


def test_unknown_alias_is_flagged_and_masked() -> None:
    a = check_answer(
        "Memory helps agents plan over long tasks [S1]. "
        "This paper reports a large gain on SWE-bench Verified [S9].",
        registry(),
    )
    assert [f.kind for f in a.flags] == ["unknown_source"]
    assert "[S9]" not in a.text and "[?]" in a.text
    assert [c.alias for c in a.citations] == ["S1"] and not a.supported


def test_uncited_claim_is_flagged() -> None:
    a = check_answer(
        "Memory helps agents plan over long tasks [S1]. "
        "Most production agents today still forget everything between sessions.",
        registry(),
    )
    assert [f.kind for f in a.flags] == ["uncited_claim"] and not a.supported


def test_short_connective_and_questions_are_not_claims() -> None:
    a = check_answer(
        "In short:\nMemory helps agents plan over long tasks [S1].\n"
        "Which one fits your setup best for agents?",
        registry(),
    )
    assert a.flags == ()


def test_bullets_are_sentences() -> None:
    a = check_answer(
        "- Memory helps agents plan over long tasks [S1]\n"
        "- Venue results confirm the effect on real benchmarks\n"
        "1. Both report gains over stateless agent baselines [S2]",
        registry(),
    )
    assert [f.sentence for f in a.flags] == ["Venue results confirm the effect on real benchmarks"]


def test_preprint_called_peer_reviewed_is_flagged() -> None:
    bad = check_answer("This peer-reviewed study shows agents forget early steps [S1].", registry())
    good = check_answer(
        "This peer-reviewed study shows agents forget early steps [S2].", registry()
    )
    assert [f.kind for f in bad.flags] == ["tier_mislabel"] and good.flags == ()


def test_no_valid_citation_gives_the_no_support_answer() -> None:
    for text in (
        "Agents mostly forget what they did a few steps earlier in practice.",
        "Agents mostly forget what they did a few steps earlier in practice [S9].",
    ):
        a = check_answer(text, registry())
        assert (a.text, a.citations, a.supported) == (NO_SUPPORT, (), False)


def test_a_repeated_citation_still_faces_the_tier_check() -> None:
    a = check_answer(
        "Agents need memory to plan over long tasks [S1]. "
        "This peer-reviewed study shows agents forget early steps [S1].",
        registry(),
    )
    assert [f.kind for f in a.flags] == ["tier_mislabel"]
