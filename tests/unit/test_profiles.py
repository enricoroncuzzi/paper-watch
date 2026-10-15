"""Topic profile loading: the description and threshold the embedding classifier depends on."""

import json
from pathlib import Path

import pytest

from evals.relevance_eval import BASELINE, DATA, fingerprint
from paperwatch.config import LOCAL_EMBEDDING_MODEL
from paperwatch.domain import load_profiles


def _write(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "profiles:\n  - name: p\n    weight: 1.0\n    categories: [cs.AI]\n    terms: [agent]\n"
        + extra
    )
    return path


def test_committed_profiles_pin_the_calibrated_descriptions_and_thresholds() -> None:
    # A description and its threshold are calibrated together, so changing either is a
    # recalibration (python -m evals.relevance_eval --write) that updates this test on purpose.
    profiles = load_profiles(Path("config/profiles.yaml"))
    assert [(p.name, p.description, p.relevance_threshold) for p in profiles] == [
        (
            "agentic-engineering",
            "Building, evaluating or analysing LLM agents: agent harnesses, tool use, multi-agent "
            "systems, coding agents, agent memory and planning, agent benchmarks and evaluation, "
            "and bots.",
            0.813,
        ),
        (
            "ai-forensics",
            "Detecting, attributing, watermarking or tracing the provenance of synthetic or "
            "manipulated media: deepfake detection, AI-generated image, audio, video and text "
            "detection, and watermarking of generated content.",
            0.799,
        ),
        (
            "ml-architectures",
            "A new neural network architecture or building block for deep learning models: "
            "attention mechanisms, linear attention and recurrent memory, state space models, "
            "mixture-of-experts layers and routing, positional encodings, and looped or hybrid "
            "transformer designs.",
            0.816,
        ),
    ]
    # Provenance: the committed descriptions, model and labelled set, bound by one fingerprint
    # (not the calendar date, so a later identical recalibration does not fail the test).
    baseline = json.loads(BASELINE.read_text())
    assert baseline["model"] == LOCAL_EMBEDDING_MODEL
    assert baseline["fingerprint"] == fingerprint(LOCAL_EMBEDDING_MODEL, profiles, DATA)
    assert baseline["thresholds"] == {p.name: p.relevance_threshold for p in profiles}


def test_description_is_stripped_and_threshold_read_as_float(tmp_path: Path) -> None:
    [p] = load_profiles(
        _write(tmp_path, "    description: '  Agents.  '\n    relevance_threshold: 0.8\n")
    )
    assert (p.description, p.relevance_threshold) == ("Agents.", 0.8)


@pytest.mark.parametrize("description", ["", "'   '", "[a, b]"])
def test_blank_or_non_text_description_is_refused(tmp_path: Path, description: str) -> None:
    path = _write(tmp_path, f"    description: {description}\n    relevance_threshold: 0.8\n")
    with pytest.raises(ValueError, match="description"):
        load_profiles(path)


def test_missing_description_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="description"):
        load_profiles(_write(tmp_path, "    relevance_threshold: 0.8\n"))


@pytest.mark.parametrize("threshold", ["0", "1", "1.5", "-0.2", "true", "'0.8'", "null"])
def test_threshold_outside_the_open_unit_interval_is_refused(
    tmp_path: Path, threshold: str
) -> None:
    path = _write(tmp_path, f"    description: Agents.\n    relevance_threshold: {threshold}\n")
    with pytest.raises(ValueError, match="relevance_threshold"):
        load_profiles(path)
