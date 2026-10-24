"""The draft check: every number, quote and link in a draft must come from stored evidence.

The draft is saved as written, so a failure here is a numbered note on the draft, never an edit.
One realistic source and one clean draft carry most tests; every other test plants exactly one bug
in the clean draft, so a flag count of one is evidence the rule fired and nothing else did.
"""

from dataclasses import replace
from uuid import UUID

from paperwatch.drafts.check import check_draft
from paperwatch.drafts.generate import DraftSource, DraftText

TITLE = "Replay Buffers Beat Summaries for Long-Running Coding Agents"
URL = "https://arxiv.org/abs/2601.01234"
REPO = "https://github.com/example/replay-buffers"
TEXT = (
    "We present Replay Buffers, a memory layer for long-running coding agents. Replaying exact "
    "tool results and dropping the ones that no longer matter cut tokens per task by 41% on 40 "
    "long-horizon coding tasks, with no loss in task success. The same policy cut retries by 31% "
    "against a summary-based baseline. Every task is Python, and the benchmark is small. Code and "
    f"logs are at {REPO}."
)
# The point of view is evidence too: the captain's own numbers and first-person experience are
# allowed in the draft because he wrote them, and nothing else is.
POV = (
    "My own harness replays the last twelve tool results. I would test this on our 300-step traces."
)
# The summary's quotes count as stored evidence, so a draft may use a number the excerpt omitted.
SUMMARY = {
    "what": {
        "text": "The authors release Replay Buffers, a memory layer for coding agents.",
        "quote": "Replaying exact tool results and dropping the ones that no longer matter",
        "interpretation": False,
    },
    "result": {
        "text": "Tokens per task fall by 41% on 40 long-horizon coding tasks, and retries by 31%.",
        "quote": "cut tokens per task by 41% on 40 long-horizon coding tasks",
        "interpretation": False,
    },
    "why": {
        "text": "Agent memory is a retrieval problem, not a summary problem.",
        "interpretation": True,
    },
    "catch": {"text": "Every task is Python and the benchmark is small.", "interpretation": True},
}
SOURCE = DraftSource(
    item_id=UUID("5a3f6f2e-0d0b-4c2a-9d55-6f1a2b3c4d5e"),
    title=TITLE,
    url=URL,
    source="arxiv",
    tier="peer_reviewed",
    basis_label="peer-reviewed via venue: ICLR",
    venue="ICLR",
    org=None,
    text=TEXT,
    github_repo=REPO,
    project_page=None,
    summary=SUMMARY,
    early=False,
)
GOOD = DraftText(
    hook="Most agent harnesses forget by summarising. The evidence says that is the expensive way.",
    body=(
        "Replaying tool results, and dropping the ones that stop mattering, cut tokens per task "
        "by 41% on 40 coding tasks and retries by 31%. Every task is Python and the benchmark is "
        "small, so this is a good reason to test the idea, not to adopt it. The code is open."
    ),
    hashtags=("agents", "memory", "llm"),
    source_link=URL,
)


def kinds(draft: DraftText, source: DraftSource = SOURCE, *, pov: str = POV) -> list[str]:
    return [flag.kind for flag in check_draft(draft, source, pov=pov).flags]


def test_a_clean_draft_has_no_flags() -> None:
    check = check_draft(GOOD, SOURCE, pov=POV)
    assert check.flags == ()
    # Counts are what the page prints as "Numbers 3/3": the checks that ran, and those that passed.
    assert check.counts["number"] == (3, 3)
    assert check.counts["link"] == (1, 1)
    assert check.counts["quote"] == (0, 0)


def test_the_checker_catches_planted_bugs() -> None:
    clean = check_draft(GOOD, SOURCE, pov=POV)
    assert clean.flags == ()
    changed_number = replace(GOOD, body=GOOD.body.replace("41%", "45%"))
    invented_quote = replace(GOOD, body=GOOD.body + ' The authors call it "a new law of agents".')
    foreign_link = replace(GOOD, body=GOOD.body + " More at https://example.com/x")
    for draft, kind in (
        (changed_number, "number"),
        (invented_quote, "quote"),
        (foreign_link, "link"),
    ):
        assert [f.kind for f in check_draft(draft, SOURCE, pov=POV).flags] == [kind]


def test_first_person_experience_not_in_the_pov_is_flagged() -> None:
    draft = replace(GOOD, body=GOOD.body + " I tested this on our monorepo.")
    assert kinds(draft, pov="The replay idea is the interesting part.") == ["first_person"]


def test_the_pov_may_supply_first_person_and_numbers() -> None:
    draft = replace(GOOD, body=GOOD.body + " I tested it on our 300-step traces.")
    check = check_draft(draft, SOURCE, pov="I tested it on our 300-step traces.")
    assert check.flags == ()
    assert check.counts["first_person"] == (1, 1)


def test_a_lab_post_without_attribution_is_flagged() -> None:
    lab = replace(
        SOURCE, tier="lab_blog", basis_label="lab post: the lab's own claim", org="OpenAI"
    )
    draft = replace(
        GOOD,
        hook="A memory layer worth a look.",
        body="The replay buffer is a claim about long-running coding agents. The work is early.",
    )
    assert kinds(draft, lab) == ["attribution"]


def test_a_lab_post_that_names_the_lab_passes() -> None:
    lab = replace(
        SOURCE, tier="lab_blog", basis_label="lab post: the lab's own claim", org="OpenAI"
    )
    draft = replace(
        GOOD,
        hook="A memory layer worth a look.",
        body="OpenAI reports that the replay buffer helps long-running coding agents.",
    )
    assert kinds(draft, lab) == []


def test_a_preprint_number_stated_as_fact_is_flagged() -> None:
    preprint = replace(
        SOURCE,
        tier="vetted_preprint",
        venue=None,
        basis_label="vetted preprint via 36 Hugging Face upvotes",
    )
    draft = replace(
        GOOD, body="Replaying tool results cut tokens per task by 41% on 40 coding tasks."
    )
    assert kinds(draft, preprint) == ["preprint_fact"]


def test_an_early_item_without_the_notice_is_flagged() -> None:
    early = replace(SOURCE, tier="vetted_preprint", early=True)
    draft = replace(
        GOOD,
        body=(
            "The authors report that replaying tool results cut tokens per task by 41% on 40 tasks."
        ),
    )
    assert kinds(draft, early) == ["early_notice"]


def test_an_early_item_with_the_notice_passes() -> None:
    early = replace(SOURCE, tier="vetted_preprint", early=True)
    draft = replace(
        GOOD,
        body=(
            "The authors report that replaying tool results cut tokens per task by 41% on 40 tasks."
        ),
        hook="An early replay buffer for coding agents.",
    )
    assert kinds(draft, early) == []


def test_a_long_copied_run_is_flagged() -> None:
    copied = (
        "Replaying exact tool results and dropping the ones that no longer matter cut tokens per "
        "task by 41% on 40 long-horizon coding tasks."
    )
    assert kinds(replace(GOOD, body=copied)) == ["long_copy"]


def test_a_wrong_source_link_is_flagged() -> None:
    assert kinds(replace(GOOD, source_link="https://example.com/other")) == ["link"]


def test_a_link_the_pov_wrote_is_allowed() -> None:
    draft = replace(GOOD, body=GOOD.body + " My notes are at https://example.com/notes.")
    check = check_draft(draft, SOURCE, pov="My notes are at https://example.com/notes.")
    assert check.flags == ()


def test_the_flags_are_numbered_notes_on_the_saved_draft() -> None:
    # The page prints each flag with the sentence it is about, never a bare kind: the reader has
    # to see which claim to fix.
    draft = replace(GOOD, body=GOOD.body + " More at https://example.com/x")
    [flag] = check_draft(draft, SOURCE, pov=POV).flags
    assert "example.com/x" in flag.text
    assert flag.detail
