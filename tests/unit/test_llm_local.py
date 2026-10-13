"""Local embedder, embedding classifier and provider factories, with stubs: no model, no network."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paperwatch.config import EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL, Settings
from paperwatch.domain import Profile, SourceItem, load_profiles
from paperwatch.llm import embedder as embedder_module
from paperwatch.llm.classifier import (
    AzureClassifier,
    EmbeddingClassifier,
    FakeClassifier,
    calibrated_relevance,
    make_classifier,
)
from paperwatch.llm.embedder import (
    LOCAL_EMBED_MAX_BYTES,
    AzureEmbedder,
    LocalEmbedder,
    embedding_text,
    make_embedder,
    make_local_model,
)
from paperwatch.llm.ledger import Ledger

AZURE = {
    "azure_openai_endpoint": "https://example.openai.azure.com",
    "azure_openai_api_key": "test-key-not-a-secret",
}


def _vec(*head: float) -> list[float]:
    # A full-length vector whose first components are given and the rest zero.
    return [*head, *([0.0] * (EMBEDDING_DIM - len(head)))]


class _StubModel:
    """Stands in for fastembed's TextEmbedding: records each call, yields one vector per text."""

    def __init__(self, vector: list[float], *, extra: int = 0) -> None:
        self.calls: list[tuple[list[str], int]] = []
        self._vector, self._extra = vector, extra

    def embed(self, documents: list[str], batch_size: int) -> Iterator[list[float]]:
        self.calls.append((list(documents), batch_size))
        for _ in range(len(documents) + self._extra):
            yield self._vector


class _KeyedEmbedder:
    """Returns a fixed vector per exact text and counts the texts it embedded."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [self._vectors[t] for t in texts]


def _profile(name: str, threshold: float) -> Profile:
    return Profile(name, 1.0, ("cs.AI",), (name,), f"about {name}", threshold)


def _item(title: str, text: str = "body") -> SourceItem:
    return SourceItem("arxiv", "1", title, ("x",), text, "u", datetime(2026, 10, 1, tzinfo=UTC))


def _classifier(item_vector: list[float], *thresholds: float) -> EmbeddingClassifier:
    # Profile i is the unit axis i, so the item's cosine to it is its i-th component / norm.
    profiles = [_profile(f"p{i}", t) for i, t in enumerate(thresholds)]
    vectors = {f"about p{i}": _vec(*([0.0] * i), 1.0) for i in range(len(thresholds))}
    vectors[embedding_text(_item("it"))] = item_vector
    return EmbeddingClassifier(_KeyedEmbedder(vectors), profiles)


def test_local_embedder_normalises_clips_and_passes_the_batch_size() -> None:
    model = _StubModel(_vec(3.0, 4.0))
    vectors = LocalEmbedder(model, batch_size=1).embed(["short", "é" * 5000])
    # fastembed does not normalise this model; the embedder must, so a dot product is a cosine.
    assert vectors == [_vec(0.6, 0.8), _vec(0.6, 0.8)]
    [(documents, batch_size)] = model.calls
    assert batch_size == 1 and documents[0] == "short"
    assert len(documents[1].encode("utf-8")) <= LOCAL_EMBED_MAX_BYTES
    assert ("é" * 5000).startswith(documents[1])


def test_local_embedder_skips_the_model_for_no_texts() -> None:
    model = _StubModel(_vec(1.0))
    assert LocalEmbedder(model, batch_size=1).embed([]) == [] and model.calls == []


@pytest.mark.parametrize(
    "model",
    [_StubModel([1.0] * 3), _StubModel(_vec()), _StubModel(_vec(1.0), extra=1)],
    ids=["wrong-dimension", "zero-vector", "extra-vector"],
)
def test_local_embedder_refuses_a_bad_model_answer(model: _StubModel) -> None:
    with pytest.raises(ValueError):
        LocalEmbedder(model, batch_size=1).embed(["text"])


def test_make_local_model_passes_the_memory_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastembed

    seen: dict[str, object] = {}

    def fake_text_embedding(model_name: str, **kwargs: object) -> _StubModel:
        seen.update(kwargs, model_name=model_name)
        return _StubModel(_vec(1.0))

    # No download: the class make_local_model imports is replaced for this test only.
    monkeypatch.setattr(fastembed, "TextEmbedding", fake_text_embedding)
    make_local_model(Settings(local_embedding_cache_dir="/models", local_embedding_threads=3))
    assert seen == {
        "model_name": LOCAL_EMBEDDING_MODEL,
        "cache_dir": "/models",
        "threads": 3,
        "enable_cpu_mem_arena": False,
    }


def test_make_embedder_picks_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _StubModel(_vec(1.0))
    monkeypatch.setattr(embedder_module, "make_local_model", lambda settings: model)
    ledger = Ledger(10)
    local = make_embedder(Settings(local_embedding_batch=4), ledger)
    assert isinstance(local, LocalEmbedder)
    local.embed(["a", "b"])
    # A local call charges nothing: the per-run token cap only guards paid APIs.
    assert model.calls[0][1] == 4 and ledger.tokens_used == 0 and ledger.entries == []
    assert isinstance(
        make_embedder(Settings(embedding_provider="azure", **AZURE), ledger), AzureEmbedder
    )


def test_calibrated_relevance_puts_the_threshold_at_one_half() -> None:
    assert calibrated_relevance(0.81, 0.81) == 0.5
    assert calibrated_relevance(1.0, 0.81) == 1.0
    assert calibrated_relevance(0.405, 0.81) == 0.25
    assert calibrated_relevance(0.0, 0.81) == 0.0
    assert calibrated_relevance(-0.3, 0.81) == 0.0  # a negative cosine is no relevance at all
    assert 0.5 < calibrated_relevance(0.9, 0.81) < 1.0
    assert calibrated_relevance(0.8, 0.81) < 0.5


def test_profile_vectors_are_embedded_once_and_items_as_title_plus_text() -> None:
    embedder = _KeyedEmbedder(
        {"about p0": _vec(1.0), "about p1": _vec(0.0, 1.0), "it\nbody": _vec(1.0)}
    )
    clf = EmbeddingClassifier(embedder, [_profile("p0", 0.5), _profile("p1", 0.5)])
    assert embedder.texts == ["about p0", "about p1"]
    clf.classify(_item("it"))
    clf.classify(_item("it"))
    assert embedder.texts[2:] == ["it\nbody", "it\nbody"]


def test_embedding_classifier_picks_the_profile_that_clears_its_threshold() -> None:
    out = _classifier(_vec(0.2, 0.9, 0.1), 0.81, 0.81, 0.81).classify(_item("it"))
    assert out.profile == "p1" and out.relevance > 0.5 and "cosine" in out.reason


def test_embedding_classifier_returns_no_profile_below_every_threshold() -> None:
    out = _classifier(_vec(0.6, 0.5, 0.0, 0.6245), 0.81, 0.81).classify(_item("it"))
    # The closest score is kept for inspection, below the 0.5 that means "matches".
    assert out.profile is None and 0.0 < out.relevance < 0.5


def test_embedding_classifier_admits_at_exactly_the_threshold() -> None:
    # A 3-4-5 triangle gives an exact cosine of 0.6 against profile p0.
    out = _classifier(_vec(3.0, 0.0, 4.0), 0.6, 0.9).classify(_item("it"))
    assert out.profile == "p0" and out.relevance == 0.5


def test_embedding_classifier_compares_profiles_on_the_calibrated_scale() -> None:
    # Raw cosine favours p0 (0.82 vs 0.58), but only p1 clears its own threshold.
    out = _classifier(_vec(8.5, 6.0), 0.9, 0.5).classify(_item("it"))
    assert out.profile == "p1"


def test_embedding_classifier_breaks_a_tie_by_profile_order() -> None:
    out = _classifier(_vec(1.0, 1.0), 0.5, 0.5).classify(_item("it"))
    assert out.profile == "p0"


def test_make_classifier_picks_the_provider() -> None:
    profiles = load_profiles(Path("config/profiles.yaml"))
    vectors = {p.description: _vec(1.0) for p in profiles}
    embedder = _KeyedEmbedder(vectors)
    ledger = Ledger(10)
    assert isinstance(make_classifier(Settings(), ledger, profiles, embedder), EmbeddingClassifier)
    keyword = make_classifier(Settings(classifier_provider="keyword"), ledger, profiles, embedder)
    assert isinstance(keyword, FakeClassifier)
    azure = make_classifier(
        Settings(classifier_provider="azure", **AZURE), ledger, profiles, embedder
    )
    assert isinstance(azure, AzureClassifier)
    assert ledger.tokens_used == 0  # building a classifier never spends tokens


@pytest.mark.parametrize(
    "settings",
    [
        Settings(embedding_provider="azure", **AZURE),
        Settings(local_embedding_model="BAAI/bge-small-en-v1.5"),
    ],
    ids=["azure-vectors", "fallback-model"],
)
def test_embedding_classifier_refuses_uncalibrated_vectors(settings: Settings) -> None:
    # The thresholds are cosines of one model; on any other they would silently misjudge.
    profiles = load_profiles(Path("config/profiles.yaml"))
    with pytest.raises(ValueError, match="recalibrate"):
        make_classifier(settings, Ledger(10), profiles, _KeyedEmbedder({}))
