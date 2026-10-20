import time
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
        "[S1][S2]",
        "See [S1].",
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


def test_zero_width_characters_cannot_hide_an_alias() -> None:
    for gap in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        a = check_answer(f"Memory helps agents plan over long tasks [S1] [S{gap}9].", registry())
        assert [f.kind for f in a.flags] == ["unknown_source"], gap
        assert f"[S{gap}9]" not in a.text and "[?]" in a.text, gap

        b = check_answer(f"Memory helps agents plan over long tasks [S{gap}1].", registry())
        assert b.supported and [c.alias for c in b.citations] == ["S1"], gap
        assert "[S1]" in b.text, gap


def test_paren_and_lowercase_groups_are_flagged_and_masked() -> None:
    for marker in ("(S9)", "[s9]", "[S 9]"):
        a = check_answer(f"Memory helps agents plan over long tasks [S1] {marker}.", registry())
        assert [f.kind for f in a.flags] == ["unknown_source"], marker
        assert marker not in a.text and "[?]" in a.text, marker


def test_a_lowercase_marker_alone_is_accepted_and_normalised() -> None:
    a = check_answer("Memory helps agents plan over long tasks [s1].", registry())
    assert a.supported and [c.alias for c in a.citations] == ["S1"]
    assert "[S1]" in a.text and "[s1]" not in a.text


def test_separators_inside_a_group_are_understood() -> None:
    for marker in ("[S1; S2]", "[S1 and S2]", "[S1 & S2]"):
        a = check_answer(
            f"Both papers measure long-horizon agent memory directly {marker}.", registry()
        )
        assert [c.alias for c in a.citations] == ["S1", "S2"], marker
        # A square group keeps its own wording; only the aliases are normalised.
        assert marker in a.text, marker
    for paren in ("(S1; S2)", "(S1, S2)", "(S1 and S2)"):
        a = check_answer(
            f"Both papers measure long-horizon agent memory directly {paren}.", registry()
        )
        assert [c.alias for c in a.citations] == ["S1", "S2"], paren
        assert "[S1, S2]" in a.text, paren


def test_a_mixed_marker_shows_a_question_mark_in_place() -> None:
    a = check_answer("Memory helps agents plan over long tasks [S1, S9].", registry())
    assert "[S1, ?]" in a.text
    assert [f.kind for f in a.flags] == ["unknown_source"]


def test_honest_negated_labels_are_not_tier_mislabels() -> None:
    for text in (
        "This preprint is not yet peer-reviewed and shows agents forget early steps [S1].",
        "This is a preprint that is not peer-reviewed and shows agents forget early steps [S1].",
        "It is a non-peer-reviewed preprint showing agents forget early steps [S1].",
    ):
        assert check_answer(text, registry()).flags == (), text


def test_tier_synonyms_are_caught() -> None:
    for text in (
        "This paper passed peer review and shows agents forget early steps [S1].",
        "This refereed publication shows agents forget early steps [S1].",
        "It was accepted to a top venue and shows agents forget early steps [S1].",
        "It was accepted for publication and shows agents forget early steps [S1].",
        "It was published at NeurIPS and shows agents forget early steps [S1].",
        "This peer\u2011reviewed study shows agents forget early steps [S1].",
    ):
        assert [f.kind for f in check_answer(text, registry()).flags] == ["tier_mislabel"], text


def test_a_short_uncited_tier_claim_is_flagged() -> None:
    a = check_answer(
        "It was accepted at ICLR. The paper shows memory helps agents plan [S1].", registry()
    )
    assert [f.kind for f in a.flags] == ["uncited_claim"]


def test_claim_min_words_boundary() -> None:
    five = check_answer("Agents forget early steps today.", registry())
    six = check_answer("Agents forget early steps very often.", registry())
    assert five.flags == ()
    assert [f.kind for f in six.flags] == ["uncited_claim"]


def test_a_peer_reviewed_source_supports_the_label() -> None:
    a = check_answer(
        "This peer-reviewed study shows agents forget early steps [S1, S2].", registry()
    )
    assert a.flags == () and a.supported


def test_flags_name_the_sentence_and_the_alias() -> None:
    unknown = check_answer(
        "Memory helps agents plan over long tasks [S1]. Adds a large gain on SWE-bench [S7].",
        registry(),
    )
    assert unknown.flags[0].sentence == "Adds a large gain on SWE-bench"
    assert "S7" in unknown.flags[0].detail

    mislabel = check_answer(
        "This peer-reviewed study shows agents forget early steps [S1].", registry()
    )
    assert mislabel.flags[0].sentence == "This peer-reviewed study shows agents forget early steps"
    assert "S1 is vetted_preprint" in mislabel.flags[0].detail


def test_headings_lead_ins_tables_and_code_are_not_claims() -> None:
    a = check_answer(
        "# Heading about agents memory and long tasks\n"
        "Here are the main findings from the retrieved papers:\n"
        "| a | b | c | d | e | f |\n"
        "**Bold heading words are here for you**\n"
        "```\nMemory helps agents plan over long tasks and beyond forever\n```\n"
        "Memory helps agents plan over long tasks [S1].",
        registry(),
    )
    assert a.flags == () and a.supported


def test_other_bullets_are_stripped() -> None:
    a = check_answer(
        "1) Both report gains over stateless agent baselines [S2]\n"
        "\u2022 Venue results confirm the effect on real benchmarks",
        registry(),
    )
    assert [f.sentence for f in a.flags] == ["Venue results confirm the effect on real benchmarks"]


def test_abbreviations_do_not_end_a_sentence() -> None:
    a = check_answer(
        "Agents such as Claude Code, Cursor, etc. forget early steps in long tasks [S1]. "
        "A study by Wang and Li et al. shows the memory effect on long agent tasks [S2].",
        registry(),
    )
    assert a.flags == () and a.supported


def test_bold_quoted_and_parenthesised_sentences_split() -> None:
    for text in (
        "**Memory helps agents plan over long tasks [S1].** "
        "Most production agents forget everything between sessions.",
        "(Memory helps agents plan over long tasks [S1].) "
        "Most production agents forget everything between sessions.",
        '"Memory helps agents plan over long tasks [S1]." '
        "Most production agents forget everything between sessions.",
    ):
        assert [f.kind for f in check_answer(text, registry()).flags] == ["uncited_claim"], text


def test_a_marker_can_end_a_sentence_without_a_stop() -> None:
    for text in (
        "Memory helps agents plan over long tasks [S1] "
        "Most production agents forget everything between sessions.",
        "Memory helps agents plan over long tasks [S1].Most production agents forget "
        "everything between sessions.",
    ):
        assert [f.kind for f in check_answer(text, registry()).flags] == ["uncited_claim"], text


def test_repeated_unknown_aliases_are_flagged_once() -> None:
    a = check_answer("Memory helps agents plan over long tasks [S9] [S9] [S9].", registry())
    assert [f.kind for f in a.flags] == ["unknown_source"]


def test_a_repeated_alias_is_named_once_in_the_tier_detail() -> None:
    a = check_answer(
        "This peer-reviewed study shows agents forget early steps [S1, S1].", registry()
    )
    assert a.flags[0].detail == "not peer-reviewed: S1 is vetted_preprint"


def test_markers_in_code_and_urls_are_ignored() -> None:
    a = check_answer(
        "Write the marker `[S9]` next to a source list [S1]. "
        "The paper lives at https://example.com/a[S9].html and measures long tasks [S1].",
        registry(),
    )
    assert a.flags == () and [c.alias for c in a.citations] == ["S1"]
    assert "`[S9]`" in a.text and "a[S9].html" in a.text


def test_flag_details_clip_an_echoed_alias() -> None:
    long_alias = "S" + "9" * 200
    a = check_answer(f"Memory helps agents plan over long tasks [S1] [{long_alias}].", registry())
    assert [f.kind for f in a.flags] == ["unknown_source"]
    assert all(len(f.detail) < 120 for f in a.flags)


def test_flags_are_kept_with_the_no_support_answer() -> None:
    a = check_answer("Most production agents forget everything between sessions.", registry())
    assert a.text == NO_SUPPORT
    assert [f.kind for f in a.flags] == ["uncited_claim"]


def test_an_unknown_alias_with_a_tier_phrase_is_only_unknown() -> None:
    a = check_answer("This peer-reviewed study shows agents forget early steps [S9].", registry())
    assert [f.kind for f in a.flags] == ["unknown_source"]


def test_the_evidence_word_boundary() -> None:
    two = check_answer("Agents forget [S1].", registry())
    three = check_answer("Agents forget early [S1].", registry())
    assert (two.text, two.citations, two.supported) == (NO_SUPPORT, (), False)
    assert three.supported and three.flags == ()


def test_a_square_group_with_extra_tokens_is_a_citation() -> None:
    ok = check_answer(
        "Both papers measure long-horizon agent memory directly [S1, S2, and S1].", registry()
    )
    assert ok.supported and ok.flags == ()

    for marker, masked in (
        ("[S1, S2, and S9]", "[S1, S2, and ?]"),
        ("[S1 S9]", "[S1 ?]"),
        ("[S9, p. 4]", "[?, p. 4]"),
        ("[S1-S9]", "[S1-?]"),
    ):
        a = check_answer(f"Memory helps agents plan over long tasks [S1] {marker}.", registry())
        assert [f.kind for f in a.flags] == ["unknown_source"], marker
        assert masked in a.text, marker


def test_an_unclosed_or_inline_fence_is_ordinary_text() -> None:
    for text in (
        "Memory helps agents plan over long tasks [S1].\n```\n"
        "Most production agents forget everything between sessions.",
        "Memory helps agents plan over long tasks [S1].\n```python x = 1```\n"
        "Most production agents forget everything between sessions.",
    ):
        assert [f.kind for f in check_answer(text, registry()).flags] == ["uncited_claim"], text


def test_a_closed_fence_is_untouched() -> None:
    text = "Memory helps agents plan over long tasks [S1].\n```\nx[s1] = data[S9]\n```\n"
    a = check_answer(text, registry())
    assert a.text == text
    assert a.flags == () and a.supported


def test_a_cited_tier_phrase_is_checked_in_any_layout() -> None:
    table = check_answer(
        "Memory helps agents plan over long tasks [S1].\n| Paper | Venue |\n"
        "| Reflexion | accepted at NeurIPS [S1] |",
        registry(),
    )
    assert [f.kind for f in table.flags] == ["tier_mislabel"]

    heading = check_answer(
        "## Peer-reviewed results show agents forget early steps [S1]", registry()
    )
    assert [f.kind for f in heading.flags] == ["tier_mislabel"]

    uncited = check_answer("## Peer-reviewed results show agents forget early steps", registry())
    assert uncited.flags == ()


def test_not_only_is_not_a_negation_and_dated_publication_is_honest() -> None:
    flagged = check_answer(
        "This paper is not only peer-reviewed but also widely cited and shows agents learn [S1].",
        registry(),
    )
    assert [f.kind for f in flagged.flags] == ["tier_mislabel"]

    dated = check_answer(
        "This preprint was published in 2024 and shows agents forget early steps [S1].",
        registry(),
    )
    assert dated.flags == () and dated.supported


def test_every_listed_negation_is_honest() -> None:
    for label in (
        "hasn't been peer-reviewed",
        "isn't peer reviewed yet",
        "was never peer-reviewed",
        "has no peer review",
        "is currently under peer review",
        "was submitted for peer review",
        "is pending peer review",
        "has not yet been formally peer-reviewed",
        "has not been accepted at any venue",
    ):
        text = f"This preprint {label} and shows agents forget early steps [S1]."
        assert check_answer(text, registry()).flags == (), label


def test_a_negation_four_words_away_does_not_apply() -> None:
    text = (
        "This preprint was not ever truly really formally peer-reviewed "
        "and shows agents learn [S1]."
    )
    assert [f.kind for f in check_answer(text, registry()).flags] == ["tier_mislabel"]


def test_etc_before_a_capital_ends_the_sentence() -> None:
    a = check_answer(
        "Agents like Cursor [S1], Copilot, etc. Most production agents forget everything soon.",
        registry(),
    )
    assert [f.kind for f in a.flags] == ["uncited_claim"]


def test_each_abbreviation_keeps_its_full_stop_and_does_not_split() -> None:
    for text, sentence in (
        (
            "Dr. Smith found that agents forget early steps in long production tasks.",
            "Dr. Smith found that agents forget early steps in long production tasks",
        ),
        (
            "Fig. 3 shows that agents forget early steps in long production tasks.",
            "Fig. 3 shows that agents forget early steps in long production tasks",
        ),
        (
            "A study by Wang et al. found that agents forget early steps in long tasks.",
            "A study by Wang et al. found that agents forget early steps in long tasks",
        ),
        (
            "Agents such as e.g. Cursor forget early steps in long production tasks.",
            "Agents such as e.g. Cursor forget early steps in long production tasks",
        ),
        (
            "The runs were fast, i.e. under one second per agent task run.",
            "The runs were fast, i.e. under one second per agent task run",
        ),
        (
            "The preprint vs. the venue paper disagree about long agent memory tasks.",
            "The preprint vs. the venue paper disagree about long agent memory tasks",
        ),
        (
            "Researchers at Google DeepMind in the U.S. found that memory helps agents plan.",
            "Researchers at Google DeepMind in the U.S. found that memory helps agents plan",
        ),
        (
            "In 2024 researchers at Google DeepMind Inc. announced a memory system for agents.",
            "In 2024 researchers at Google DeepMind Inc. announced a memory system for agents",
        ),
    ):
        a = check_answer(text, registry())
        assert [f.sentence for f in a.flags] == [sentence], text


def test_a_cited_table_and_short_bullets_are_evidence() -> None:
    table = check_answer(
        "Comparison:\n| Paper | Result |\n| Reflexion | verbal feedback loops [S1] |\n"
        "| Voyager | skill library growth [S2] |",
        registry(),
    )
    assert table.supported and table.flags == ()

    bullets = check_answer("- Reflexion: verbal RL [S1]\n- Voyager: skill library [S2]", registry())
    assert bullets.supported and bullets.flags == ()


def test_a_wholly_bold_sentence_with_a_citation_counts() -> None:
    a = check_answer("**Memory helps agents plan over long tasks [S1].**", registry())
    assert a.supported and a.flags == ()

    b = check_answer("**Bold heading words are here for you all**", registry())
    assert b.flags == ()


def test_closers_after_the_stop_split_the_sentence() -> None:
    for text in (
        "_Memory helps agents plan over long tasks [S1]._ Next production agents forget "
        "everything between sessions.",
        "(Memory helps agents plan over long tasks [S1].)Next production agents forget "
        "everything between sessions.",
    ):
        assert [f.kind for f in check_answer(text, registry()).flags] == ["uncited_claim"], text


def test_a_non_ascii_digit_alias_is_masked() -> None:
    a = check_answer("Memory helps agents plan over long tasks [S1] [S\u0663].", registry())
    assert [f.kind for f in a.flags] == ["unknown_source"]
    assert "[S\u0663]" not in a.text and "[?]" in a.text


def test_the_alias_clip_boundary_and_the_detail_wording() -> None:
    exact = "S" + "9" * 31  # 32 characters, kept whole
    over = "S" + "9" * 32  # 33 characters, clipped
    kept = check_answer(f"Memory helps agents plan over long tasks [S1] [{exact}].", registry())
    assert kept.flags[0].detail == f"{exact} was not returned by any tool for this answer"
    clipped = check_answer(f"Memory helps agents plan over long tasks [S1] [{over}].", registry())
    assert clipped.flags[0].detail == f"{exact}... was not returned by any tool for this answer"


def test_dash_and_paren_bullets_are_stripped() -> None:
    a = check_answer(
        "1) Venue results confirm the effect on real benchmarks\n"
        "\u00b7 Agent memory studies report similar gains on long tasks",
        registry(),
    )
    assert [f.sentence for f in a.flags] == [
        "Venue results confirm the effect on real benchmarks",
        "Agent memory studies report similar gains on long tasks",
    ]


def test_a_large_repetitive_answer_is_fast() -> None:
    text = "[S1] http://a.b " * 5000
    start = time.perf_counter()
    a = check_answer(text, registry())
    assert a.supported
    assert time.perf_counter() - start < 1.0


def test_a_question_is_not_a_tier_claim() -> None:
    uncited = check_answer("Is this study peer-reviewed and accepted at a venue?", registry())
    assert uncited.flags == ()
    cited = check_answer(
        "Memory helps agents plan over long tasks [S1]. Is this peer-reviewed at all?", registry()
    )
    assert cited.flags == () and cited.supported
