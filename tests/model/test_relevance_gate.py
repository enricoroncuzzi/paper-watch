"""The relevance gate: the committed thresholds keep their held-out scores with the real model.

Opt-in like every real-model test (`uv run pytest -m model`); the model-evals workflow runs it.
"""

import json

import pytest

from evals.relevance_eval import BASELINE, DATA, PROFILES, gate, load_examples, score
from paperwatch.config import Settings
from paperwatch.domain import load_profiles
from paperwatch.llm.classifier import FakeClassifier, make_classifier
from paperwatch.llm.embedder import make_embedder
from paperwatch.llm.ledger import Ledger

pytestmark = pytest.mark.model


def test_committed_thresholds_keep_their_held_out_scores() -> None:
    settings = Settings()
    profiles = load_profiles(PROFILES)
    names = [p.name for p in profiles]
    test = [e for e in load_examples(DATA, names) if e.split == "test"]
    # The production path: the same factories and guards a live run uses.
    classifier = make_classifier(settings, Ledger(0), profiles, make_embedder(settings, Ledger(0)))
    scores = score(classifier, test, names)
    assert gate(scores, json.loads(BASELINE.read_text())["test"]) == []
    # The model has to earn its place: it must beat matching the profile keywords.
    assert scores.macro_f1 > score(FakeClassifier(profiles), test, names).macro_f1
