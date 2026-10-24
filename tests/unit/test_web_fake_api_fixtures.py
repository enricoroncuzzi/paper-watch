"""The fake API's fixtures, checked against the API's own response models.

The fake API is what every page test runs against, so a fixture that no longer matches the
contract would let a broken page pass CI. Validating here means the fake cannot drift from the
route it stands in for, and a renamed field fails in Python before it becomes a blank cell.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from paperwatch.api.app import (
    AnswerData,
    ErrorData,
    NoticeData,
    SourceData,
    StatusData,
    StepData,
    TokenData,
    ToolData,
)
from paperwatch.api.responses import RateLimited, Refusal
from paperwatch.api.routes.runs import RunDetail, RunsData, TodayData
from paperwatch.api.routes.suggestions import SuggestionsData

WEB = Path(__file__).resolve().parents[2] / "web"
FIXTURES = WEB / "fake-api" / "fixtures"
STREAMS = WEB / "fake-api" / "streams"
# The fake API's gate refusals, kept in a file so they can be validated against the contract.
REFUSALS = WEB / "fake-api" / "refusals.json"

FIXTURE_MODELS: dict[str, type[BaseModel]] = {
    "today": TodayData,
    "today-unhealthy": TodayData,
    "today-followup": TodayData,
    "today-text-only": TodayData,
    "runs": RunsData,
    "run-detail": RunDetail,
    "status": StatusData,
    "suggestions": SuggestionsData,
    "suggestions-empty": SuggestionsData,
    "suggestions-pending": SuggestionsData,
    "suggestions-javascript": SuggestionsData,
}

EVENT_MODELS: dict[str, type[BaseModel]] = {
    "tool": ToolData,
    "step": StepData,
    "source": SourceData,
    "token": TokenData,
    "notice": NoticeData,
    "answer": AnswerData,
    "error": ErrorData,
}


def test_every_fixture_file_has_a_model() -> None:
    # A new fixture with no model would silently skip validation, so the set must match exactly.
    assert {path.stem for path in FIXTURES.glob("*.json")} == set(FIXTURE_MODELS)


@pytest.mark.parametrize("name", sorted(FIXTURE_MODELS))
def test_each_fixture_matches_its_response_model(name: str) -> None:
    payload: Any = json.loads((FIXTURES / f"{name}.json").read_text())
    model = FIXTURE_MODELS[name]
    parsed = model.model_validate(payload)
    # Round-tripping proves the model keeps every field the fixture sets, so a renamed field is a
    # failure here rather than a blank cell on the page.
    assert parsed.model_dump(mode="json", by_alias=True) == payload


@pytest.mark.parametrize("path", sorted(STREAMS.glob("*.json")), ids=lambda p: p.stem)
def test_each_stream_frame_matches_its_event_model(path: Path) -> None:
    frames: list[dict[str, Any]] = json.loads(path.read_text())
    assert frames, "a stream with no frames tests nothing"
    for frame in frames:
        if "comment" in frame:
            # A comment line is the heartbeat and carries no event payload.
            assert frame["comment"] == ": ping"
            continue
        model = EVENT_MODELS[frame["event"]]
        assert (
            model.model_validate(frame["data"]).model_dump(mode="json", by_alias=True)
            == frame["data"]
        )


def test_the_fake_refusals_match_the_declared_gate_bodies() -> None:
    # The gate declares Refusal for 400/401/403/413/415 and RateLimited for 429; the fake serves
    # exactly these bodies, so a page test that hits one sees the shape the real API sends.
    bodies: dict[str, Any] = json.loads(REFUSALS.read_text())
    assert set(bodies) == {"400", "401", "403", "413", "415", "429"}
    for status in ("400", "401", "403", "413", "415"):
        assert Refusal.model_validate(bodies[status]).model_dump(mode="json") == bodies[status]
    assert RateLimited.model_validate(bodies["429"]).model_dump(mode="json") == bodies["429"]
