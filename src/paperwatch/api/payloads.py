"""The checked-answer payload, shared by the `answer` event and a thread's stored turn.

It lives here rather than in `app.py` so the threads route can return the same shape without
importing the composition root that builds it: the stored turn is the `AnswerData` the reader saw,
so the two must be one model or a re-fetch could drift from the stream.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CitationData(BaseModel):
    id: str = Field(description="The per-answer source alias, such as S1.")
    item_id: str
    title: str
    url: str
    tier: str
    basis: str


class FlagData(BaseModel):
    kind: str
    sentence: str
    detail: str


class AnswerData(BaseModel):
    text: str
    search_mode: Literal["hybrid", "text_only"] = Field(
        default="hybrid", description="Hybrid retrieval, or the keyword-only branch during a run."
    )
    supported: bool = Field(
        description=(
            "True when every visible claim points at a retrieved source. That is not a "
            "verification: sentences under six words and numbers are not checked."
        )
    )
    citations: list[CitationData]
    flags: list[FlagData]


__all__ = ["AnswerData", "CitationData", "FlagData"]
