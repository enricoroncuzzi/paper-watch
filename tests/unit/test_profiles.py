"""Topic profile loading: the description and threshold the embedding classifier depends on."""

from pathlib import Path

import pytest

from paperwatch.domain import load_profiles


def _write(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "profiles:\n  - name: p\n    weight: 1.0\n    categories: [cs.AI]\n    terms: [agent]\n"
        + extra
    )
    return path


def test_committed_profiles_carry_a_description_and_the_scout_threshold() -> None:
    profiles = load_profiles(Path("config/profiles.yaml"))
    assert [p.name for p in profiles] == ["agentic-engineering", "ai-forensics", "ml-architectures"]
    assert all(p.description and p.relevance_threshold == 0.81 for p in profiles)


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
