import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from paperwatch.config import EMBEDDING_DIM, Settings
from paperwatch.domain import SourceItem, load_profiles
from paperwatch.llm.classifier import AzureClassifier, FakeClassifier, _Out
from paperwatch.llm.embedder import (
    EMBED_BATCH,
    EMBED_MAX_BYTES,
    AzureEmbedder,
    FakeEmbedder,
    clip_for_embedding,
    make_azure_client,
)
from paperwatch.llm.ledger import BudgetExceeded, Ledger


class _StubEmbeddings:
    """Records each request the way the Azure SDK would receive it."""

    def __init__(
        self, *, dim: int = EMBEDDING_DIM, reverse: bool = False, usage: int | None = None
    ) -> None:
        self.calls: list[list[str]] = []
        self.dimensions: list[int] = []
        self._dim = dim
        self._reverse = reverse
        self._usage = usage

    def create(self, model: str, input: list[str], dimensions: int) -> object:
        self.calls.append(input)
        self.dimensions.append(dimensions)
        # Each vector encodes its input's length, so a caller can check order and dimension.
        data = [
            type("D", (), {"embedding": [float(len(t))] * self._dim, "index": i})()
            for i, t in enumerate(input)
        ]
        if self._reverse:
            data = list(reversed(data))
        total = self._usage if self._usage is not None else sum(len(t) // 4 for t in input)
        usage = type("U", (), {"total_tokens": total})()
        return type("R", (), {"data": data, "usage": usage})()


def _embedding_client(stub: _StubEmbeddings) -> object:
    return type("C", (), {"embeddings": stub})()


class _StubParse:
    """Returns one canned parse response, or raises the error it was given."""

    def __init__(self, *, parsed: object = None, error: Exception | None = None) -> None:
        self._parsed = parsed
        self._error = error
        self.calls = 0

    def parse(self, **kwargs: object) -> object:
        self.calls += 1
        if self._error is not None:
            raise self._error
        message = type("M", (), {"parsed": self._parsed})()
        choice = type("C", (), {"message": message})()
        usage = type("U", (), {"total_tokens": 12})()
        return type("R", (), {"choices": [choice], "usage": usage})()


def _classifier_client(stub: _StubParse) -> object:
    completions = type("ChatCompletions", (), {"parse": stub.parse})()
    chat = type("Chat", (), {"completions": completions})()
    return type("C", (), {"chat": chat})()


def _azure_classifier(stub: _StubParse) -> AzureClassifier:
    profiles = load_profiles(Path("config/profiles.yaml"))
    return AzureClassifier(_classifier_client(stub), "deploy", profiles, Ledger(10**9))


def _item(title: str, text: str) -> SourceItem:
    return SourceItem("arxiv", "1", title, ("x",), text, "u", datetime(2026, 10, 1, tzinfo=UTC))


def _validation_error() -> ValidationError:
    # A real pydantic error, the kind the SDK raises when the model answer misses the schema.
    try:
        _Out(profile="x", relevance=2.0, reason="r")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected _Out to reject relevance 2.0")


def test_long_text_is_clipped_below_the_model_limit() -> None:
    # The 8000-byte cut lands in this ASCII prefix; the multi-byte cut is covered by
    # test_clip_handles_empty_multibyte_and_partial_characters.
    post = ("Agent harness design. " * 2200) + "é" * 3000
    clipped = clip_for_embedding(post)
    assert len(clipped.encode("utf-8")) <= EMBED_MAX_BYTES
    assert post.startswith(clipped)  # cut on a character boundary, never mid-character
    stub = _StubEmbeddings()
    AzureEmbedder(_embedding_client(stub), "emb", Ledger(10**9)).embed([post])
    assert len(stub.calls[0][0].encode("utf-8")) <= EMBED_MAX_BYTES


def test_clip_handles_empty_multibyte_and_partial_characters() -> None:
    assert clip_for_embedding("") == ""
    # A character slice would keep all 12000 bytes of these emoji; a byte cut cannot.
    emoji = "😀" * 3000
    emoji_clipped = clip_for_embedding(emoji)
    assert len(emoji_clipped.encode("utf-8")) <= EMBED_MAX_BYTES
    assert emoji.startswith(emoji_clipped)
    # The cut lands inside a multi-byte "é", dropped whole, never as half a character.
    partial = "x" + "é" * 5000
    partial_clipped = clip_for_embedding(partial)
    assert len(partial_clipped.encode("utf-8")) <= EMBED_MAX_BYTES
    assert partial.startswith(partial_clipped)


def test_requests_are_batched_within_the_request_limit() -> None:
    stub = _StubEmbeddings()
    vectors = AzureEmbedder(_embedding_client(stub), "emb", Ledger(10**9)).embed(
        ["text"] * (EMBED_BATCH + 8)
    )
    assert [len(c) for c in stub.calls] == [EMBED_BATCH, 8] and len(vectors) == EMBED_BATCH + 8


def test_azure_embedder_returns_vectors_in_input_order() -> None:
    stub = _StubEmbeddings(reverse=True)
    vectors = AzureEmbedder(_embedding_client(stub), "emb", Ledger(10**9)).embed(
        ["aa", "bbbb", "cccccc"]
    )
    # Each vector encodes its input's length; the response was deliberately reversed.
    assert [v[0] for v in vectors] == [2.0, 4.0, 6.0]


def test_azure_embedder_asks_for_the_schema_dimension() -> None:
    # text-embedding-3 models answer 1536 by default; the column holds EMBEDDING_DIM.
    stub = _StubEmbeddings()
    AzureEmbedder(_embedding_client(stub), "emb", Ledger(10**9)).embed(["text"])
    assert stub.dimensions == [EMBEDDING_DIM]


def test_azure_embedder_rejects_a_vector_of_the_wrong_length() -> None:
    stub = _StubEmbeddings(dim=3)
    with pytest.raises(ValueError):
        AzureEmbedder(_embedding_client(stub), "emb", Ledger(10**9)).embed(["text"])


def test_budget_exceeded_raises() -> None:
    ledger = Ledger(max_tokens=100)
    ledger.charge("m", 60)
    with pytest.raises(BudgetExceeded):
        ledger.charge("m", 41)
    assert ledger.tokens_used == 60


def test_charge_allows_reaching_the_cap_exactly() -> None:
    ledger = Ledger(100)
    ledger.charge("m", 60)
    ledger.charge("m", 40)
    assert ledger.tokens_used == 100
    with pytest.raises(BudgetExceeded):
        ledger.charge("m", 1)


def test_embedder_reserves_a_token_so_a_full_ledger_never_calls_the_api() -> None:
    ledger = Ledger(100)
    ledger.charge("m", 100)
    stub = _StubEmbeddings()
    with pytest.raises(BudgetExceeded):
        AzureEmbedder(_embedding_client(stub), "m", ledger).embed(["ab"])
    assert stub.calls == []


def test_a_positive_estimate_that_does_not_fit_raises_before_the_call() -> None:
    stub = _StubEmbeddings()
    ledger = Ledger(10)
    with pytest.raises(BudgetExceeded):
        AzureEmbedder(_embedding_client(stub), "m", ledger).embed(["a" * 120])
    assert stub.calls == [] and ledger.tokens_used == 0


def test_a_delta_that_does_not_fit_raises_after_the_call() -> None:
    stub = _StubEmbeddings(usage=100)
    ledger = Ledger(50)
    with pytest.raises(BudgetExceeded):
        AzureEmbedder(_embedding_client(stub), "m", ledger).embed(["a" * 120])
    # The estimate (40) is recorded; the 60-token delta cannot fit, so the call is discarded.
    assert len(stub.calls) == 1 and ledger.tokens_used == 40


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


def test_fake_classifier_matches_whole_words_only() -> None:
    clf = FakeClassifier(load_profiles(Path("config/profiles.yaml")))
    # "bot" must not fire on the ordinary word "both", nor inside "robot".
    both = clf.classify(_item("We compare both methods", "The study evaluates both designs."))
    assert both.profile is None and both.relevance == 0.0
    bots = clf.classify(_item("bots", ""))
    assert bots.profile == "agentic-engineering" and bots.relevance == 0.5
    bot = clf.classify(_item("bot", ""))
    assert bot.profile == "agentic-engineering"
    robot = clf.classify(_item("robot", ""))
    assert robot.profile is None


def test_fake_classifier_boundary_hit_is_exactly_half() -> None:
    clf = FakeClassifier(load_profiles(Path("config/profiles.yaml")))
    out = clf.classify(_item("Memory", ""))
    assert out.profile == "agentic-engineering" and out.relevance == 0.5


def test_fake_classifier_returns_none_when_no_terms_match() -> None:
    clf = FakeClassifier(load_profiles(Path("config/profiles.yaml")))
    out = clf.classify(_item("Crystal lattices", "We measure silicate refraction indexes."))
    assert out.profile is None and out.relevance == 0.0


def test_azure_classifier_raises_when_the_model_gives_no_answer() -> None:
    clf = _azure_classifier(_StubParse(parsed=None))
    with pytest.raises(RuntimeError):
        clf.classify(_item("Title", "Body text."))


def test_azure_classifier_raises_on_an_unknown_profile() -> None:
    clf = _azure_classifier(_StubParse(parsed=_Out(profile="nope", relevance=0.5, reason="r")))
    with pytest.raises(RuntimeError):
        clf.classify(_item("Title", "Body text."))


def test_azure_classifier_keeps_a_deliberate_null_profile() -> None:
    parsed = _Out(profile=None, relevance=0.9, reason="no fit")
    clf = _azure_classifier(_StubParse(parsed=parsed))
    out = clf.classify(_item("Title", "Body text."))
    assert out.profile is None and out.relevance == 0.9 and out.reason == "no fit"


def test_azure_classifier_propagates_a_validation_error() -> None:
    clf = _azure_classifier(_StubParse(error=_validation_error()))
    with pytest.raises(ValidationError):
        clf.classify(_item("Title", "Body text."))


def test_make_azure_client_sets_retries_and_a_bounded_timeout() -> None:
    settings = Settings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="test-key-not-a-secret",
    )
    client = make_azure_client(settings)
    try:
        assert client.max_retries == 3
        assert client.timeout == 30.0
    finally:
        client.close()


def test_make_azure_client_rejects_a_whitespace_only_key() -> None:
    settings = Settings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="   ",
    )
    with pytest.raises(RuntimeError):
        make_azure_client(settings)
