import os
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from paperwatch.config import EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL, Settings


@pytest.fixture
def build_settings(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Settings]:
    """Build Settings from the test's own environment: no local .env and no leftover PW_* vars."""
    for name in [n for n in os.environ if n.startswith("PW_")]:
        monkeypatch.delenv(name, raising=False)

    def build() -> Settings:
        # _env_file=None ignores a developer's real .env; env vars the test sets still apply.
        return Settings(_env_file=None)

    return build


def test_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[[], Settings]
) -> None:
    # monkeypatch sets env vars for this test only and undoes them afterwards.
    monkeypatch.setenv("PW_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/pw")
    monkeypatch.setenv("PW_MAX_TOKENS_PER_RUN", "12345")
    s = build_settings()
    assert s.database_url == "postgresql+psycopg://u:p@localhost:5432/pw"
    assert s.max_tokens_per_run == 12345  # the string "12345" is parsed to an int
    assert s.azure_openai_endpoint is None  # unset optional fields default to None
    assert not hasattr(s, "max_cost_eur_per_run")  # EUR accounting arrives in M4


def test_empty_values_fall_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[[], Settings]
) -> None:
    # An unfilled line copied from .env.example (PW_X=) must mean "unset", not "".
    monkeypatch.setenv("PW_MAX_TOKENS_PER_RUN", "")
    monkeypatch.setenv("PW_LOG_LEVEL", "")
    s = build_settings()
    assert s.max_tokens_per_run == 200_000
    assert s.log_level == "INFO"


def test_embedding_dim_is_single_constant() -> None:
    # Guards against the dimension drifting from the DB schema and the model.
    assert EMBEDDING_DIM == 384


def test_model_providers_default_to_local_and_need_no_key(
    build_settings: Callable[[], Settings],
) -> None:
    # The caller's environment and a local .env must not leak into a defaults test.
    s = build_settings()
    assert (s.embedding_provider, s.classifier_provider) == ("local", "embedding")
    assert s.local_embedding_model == LOCAL_EMBEDDING_MODEL
    assert (s.local_embedding_threads, s.local_embedding_batch) == (2, 1)
    assert s.local_embedding_cpu_arena is False and s.local_embedding_cache_dir is None


def test_model_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[[], Settings]
) -> None:
    monkeypatch.setenv("PW_EMBEDDING_PROVIDER", "azure")
    monkeypatch.setenv("PW_CLASSIFIER_PROVIDER", "keyword")
    monkeypatch.setenv("PW_LOCAL_EMBEDDING_CACHE_DIR", "/models")
    monkeypatch.setenv("PW_LOCAL_EMBEDDING_BATCH", "4")
    s = build_settings()
    assert (s.embedding_provider, s.classifier_provider) == ("azure", "keyword")
    assert (s.local_embedding_cache_dir, s.local_embedding_batch) == ("/models", 4)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PW_EMBEDDING_PROVIDER", "openrouter"),
        ("PW_CLASSIFIER_PROVIDER", "llm"),
        ("PW_LOCAL_EMBEDDING_BATCH", "0"),
        ("PW_LOCAL_EMBEDDING_THREADS", "0"),
    ],
)
def test_unknown_provider_or_non_positive_size_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    build_settings: Callable[[], Settings],
    name: str,
    value: str,
) -> None:
    # A typo must fail at startup, not silently pick a default provider.
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        build_settings()
