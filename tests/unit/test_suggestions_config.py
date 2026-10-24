"""unit tests for config/suggestions.yaml and its loader."""

from pathlib import Path

import pytest
import yaml

from paperwatch.config import ADMISSION_PATH, FEEDS_PATH, SUGGESTIONS_PATH
from paperwatch.suggestions.config import WEIGHT_NAMES, load_suggestions

CFG = load_suggestions(SUGGESTIONS_PATH)


def _body(**overrides: object) -> str:
    # Start from the committed file so a test changes exactly the one key it is about.
    data = yaml.safe_load(SUGGESTIONS_PATH.read_text())
    data.update(overrides)
    return yaml.safe_dump(data)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "suggestions.yaml"
    path.write_text(body)
    return path


def test_the_committed_file_loads() -> None:
    assert CFG.picks == 5
    assert CFG.rising == 3
    assert CFG.momentum_days == 3
    assert CFG.min_text_chars == 400
    assert CFG.event_bonus.promotion_window_days == 14
    # The components must add up to the score, so the weights must add up to one.
    assert sum(getattr(CFG.weights, name) for name in WEIGHT_NAMES) == pytest.approx(1.0, abs=1e-6)
    assert CFG.credibility.peer_reviewed == 1.0


def test_an_unknown_key_is_refused_with_the_path(tmp_path: Path) -> None:
    path = _write(tmp_path, _body(stickiness_days=3))
    with pytest.raises(ValueError, match=str(path)):
        load_suggestions(path)


def test_weights_must_sum_to_one(tmp_path: Path) -> None:
    weights = {name: getattr(CFG.weights, name) for name in WEIGHT_NAMES}
    weights["momentum"] = 0.4
    with pytest.raises(ValueError, match="sum"):
        load_suggestions(_write(tmp_path, _body(weights=weights)))


def test_a_negative_weight_is_refused(tmp_path: Path) -> None:
    weights = {name: getattr(CFG.weights, name) for name in WEIGHT_NAMES}
    weights["momentum"] = -0.1
    weights["recency"] = 0.65
    with pytest.raises(ValueError, match=str(tmp_path)):
        load_suggestions(_write(tmp_path, _body(weights=weights)))


def test_recognisable_names_include_every_active_feed_lab() -> None:
    # The list is proposed from these two files, so a lab added to either must be looked at here
    # rather than silently leaving a known lab's posts out of the picks.
    labs = {feed["lab"] for feed in yaml.safe_load(FEEDS_PATH.read_text())["feeds"]}
    authors = set(yaml.safe_load(ADMISSION_PATH.read_text())["lab_authors"])
    assert labs <= set(CFG.recognisable_names)
    assert authors <= set(CFG.recognisable_names)
