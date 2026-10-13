import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paperwatch.config import EMBEDDING_DIM
from paperwatch.domain import SourceItem, load_profiles
from paperwatch.llm.classifier import FakeClassifier
from paperwatch.llm.embedder import (
    EMBED_BATCH,
    EMBED_MAX_BYTES,
    AzureEmbedder,
    FakeEmbedder,
    clip_for_embedding,
)
from paperwatch.llm.ledger import BudgetExceeded, Ledger


class _StubEmbeddings:
    """Records each request the way the Azure SDK would receive it."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]) -> object:
        self.calls.append(input)
        data = [type("D", (), {"embedding": [0.0] * EMBEDDING_DIM})() for _ in input]
        usage = type("U", (), {"total_tokens": sum(len(t) // 4 for t in input)})()
        return type("R", (), {"data": data, "usage": usage})()


def test_long_text_is_clipped_below_the_model_limit() -> None:
    # A full-text blog post of 50,000 characters, with multi-byte characters near the cut.
    post = ("Agent harness design. " * 2200) + "é" * 3000
    clipped = clip_for_embedding(post)
    assert len(clipped.encode("utf-8")) <= EMBED_MAX_BYTES
    assert post.startswith(clipped)  # cut on a character boundary, never mid-character
    stub = _StubEmbeddings()
    client = type("C", (), {"embeddings": stub})()
    AzureEmbedder(client, "emb", Ledger(10**9)).embed([post])  # type: ignore[arg-type]
    assert len(stub.calls[0][0].encode("utf-8")) <= EMBED_MAX_BYTES


def test_requests_are_batched_within_the_request_limit() -> None:
    stub = _StubEmbeddings()
    client = type("C", (), {"embeddings": stub})()
    vectors = AzureEmbedder(client, "emb", Ledger(10**9)).embed(  # type: ignore[arg-type]
        ["text"] * (EMBED_BATCH + 8)
    )
    assert [len(c) for c in stub.calls] == [EMBED_BATCH, 8] and len(vectors) == EMBED_BATCH + 8


def test_budget_exceeded_raises() -> None:
    ledger = Ledger(max_tokens=100)
    ledger.charge("m", 60)
    with pytest.raises(BudgetExceeded):
        ledger.charge("m", 41)
    assert ledger.tokens_used == 60


def test_fake_embedder_is_deterministic_normalised_and_semantic_enough() -> None:
    e = FakeEmbedder()
    a, b, c = e.embed(["agent tool use harness", "harness for agent tool use", "image watermark"])
    assert len(a) == EMBEDDING_DIM
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)
    cos = lambda u, v: sum(x * y for x, y in zip(u, v, strict=True))  # noqa: E731
    assert cos(a, b) > cos(a, c)
    assert e.embed(["agent tool use harness"])[0] == a


def test_fake_classifier_picks_profile_by_terms() -> None:
    clf = FakeClassifier(load_profiles(Path("config/profiles.yaml")))
    item = SourceItem(
        "arxiv",
        "1",
        "A coding agent harness",
        ("x",),
        "We study tool use in LLM agent harnesses.",
        "u",
        datetime(2026, 10, 1, tzinfo=UTC),
    )
    out = clf.classify(item)
    assert out.profile == "agentic-engineering" and out.relevance >= 0.5
