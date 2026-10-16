"""The real local embedding model. Opt-in: `uv run pytest -m model` (downloads about 180 MB once).

Set PW_LOCAL_EMBEDDING_CACHE_DIR to reuse a downloaded model; default CI never runs these.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from paperwatch.config import EMBEDDING_DIM, Settings
from paperwatch.domain import SourceItem, load_profiles
from paperwatch.llm.classifier import make_classifier
from paperwatch.llm.embedder import Embedder, make_embedder
from paperwatch.llm.ledger import Ledger

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return make_embedder(Settings(), Ledger(0))


def _item(title: str, text: str) -> SourceItem:
    return SourceItem("arxiv", "1", title, ("x",), text, "u", datetime(2026, 10, 1, tzinfo=UTC))


def test_real_model_gives_unit_vectors_of_the_schema_dimension(embedder: Embedder) -> None:
    a, b, c = embedder.embed(
        [
            "A harness that lets a coding agent call tools and verify its own patches.",
            "Tool-using LLM agents that write code and check their changes.",
            "Sourdough fermentation kinetics and flour hydration.",
        ]
    )
    assert len(a) == EMBEDDING_DIM
    assert abs(sum(x * x for x in a) - 1.0) < 1e-5
    cos = lambda u, v: sum(x * y for x, y in zip(u, v, strict=True))  # noqa: E731
    assert cos(a, b) > cos(a, c)
    # Deterministic: the same text gives the same vector on a second call.
    again = embedder.embed(["Sourdough fermentation kinetics and flour hydration."])[0]
    assert again == pytest.approx(c, abs=1e-6)


def test_real_model_routes_clear_abstracts_with_the_committed_thresholds(
    embedder: Embedder,
) -> None:
    profiles = load_profiles(Path("config/profiles.yaml"))
    clf = make_classifier(Settings(), Ledger(0), profiles, embedder)
    agent = _item(
        "SWE-Harness: evaluating tool-using coding agents",
        "We build a harness in which LLM agents plan, call tools and edit repositories, and "
        "evaluate coding agents on long-horizon software tasks with memory across sessions.",
    )
    forensic = _item(
        "Detecting diffusion-generated images",
        "We detect AI-generated and deepfake images and attribute them to the generator that "
        "made them, and test robustness to watermark removal.",
    )
    bread = _item(
        "Sourdough fermentation kinetics",
        "We measure yeast activity, acidity and flour hydration in bread dough over 48 hours.",
    )
    assert clf.classify(agent).profile == "agentic-engineering"
    assert clf.classify(forensic).profile == "ai-forensics"
    assert clf.classify(bread).profile is None
