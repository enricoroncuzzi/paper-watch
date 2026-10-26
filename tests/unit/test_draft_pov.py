"""The point of view is refused when it could not be stored, before a model call is paid for.

A NUL and a lone surrogate both reach Postgres as an error: the first is rejected by the JSONB
write, the second cannot be encoded at all. The draft row is written after the provider call, so
the route must refuse the text on the way in. The CLI shares the same rule, so a terminal paste
fails the same way the page does.
"""

import pytest

from paperwatch.drafts.prompts import clean_pov


def test_a_point_of_view_is_returned_unchanged() -> None:
    text = "I would test this on our own agent, and say so in the post."
    assert clean_pov(text) == text


def test_an_empty_point_of_view_is_allowed() -> None:
    # The dialog sends an empty point of view when the reader writes none.
    assert clean_pov("") == ""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a NUL\x00inside", "NUL"),
        # A lone surrogate is what a "\ud800" JSON escape becomes in Python.
        ("a lone \ud800 surrogate", "surrogate"),
        ("a low \udfff surrogate", "surrogate"),
    ],
)
def test_a_point_of_view_that_cannot_be_stored_is_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        clean_pov(text)
