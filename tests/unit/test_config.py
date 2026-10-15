import pytest
from pydantic import ValidationError

from paperwatch.config import EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL, Settings


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch sets env vars for this test only and undoes them afterwards.
    monkeypatch.setenv("PW_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/pw")
    monkeypatch.setenv("PW_MAX_TOKENS_PER_RUN", "12345")
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@localhost:5432/pw"
    assert s.max_tokens_per_run == 12345  # the string "12345" is parsed to an int
    assert s.azure_openai_endpoint is None  # unset optional fields default to None
    assert not hasattr(s, "max_cost_eur_per_run")  # EUR accounting arrives in M4


def test_empty_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unfilled line copied from .env.example (PW_X=) must mean "unset", not "".
    monkeypatch.setenv("PW_MAX_TOKENS_PER_RUN", "")
    monkeypatch.setenv("PW_LOG_LEVEL", "")
    s = Settings()
    assert s.max_tokens_per_run == 200_000
    assert s.log_level == "INFO"


def test_embedding_dim_is_single_constant() -> None:
    # Guards against the dimension drifting from the DB schema and the model.
    assert EMBEDDING_DIM == 384


def test_model_providers_default_to_local_and_need_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The caller's environment must not leak into a defaults test, for example
    # PW_LOCAL_EMBEDDING_CACHE_DIR exported before a real-model run.
    for name in (
        "PW_EMBEDDING_PROVIDER",
        "PW_CLASSIFIER_PROVIDER",
        "PW_LOCAL_EMBEDDING_MODEL",
        "PW_LOCAL_EMBEDDING_CACHE_DIR",
        "PW_LOCAL_EMBEDDING_THREADS",
        "PW_LOCAL_EMBEDDING_BATCH",
        "PW_LOCAL_EMBEDDING_CPU_ARENA",
    ):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert (s.embedding_provider, s.classifier_provider) == ("local", "embedding")
    assert s.local_embedding_model == LOCAL_EMBEDDING_MODEL
    assert (s.local_embedding_threads, s.local_embedding_batch) == (2, 1)
    assert s.local_embedding_cpu_arena is False and s.local_embedding_cache_dir is None


def test_model_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PW_EMBEDDING_PROVIDER", "azure")
    monkeypatch.setenv("PW_CLASSIFIER_PROVIDER", "keyword")
    monkeypatch.setenv("PW_LOCAL_EMBEDDING_CACHE_DIR", "/models")
    monkeypatch.setenv("PW_LOCAL_EMBEDDING_BATCH", "4")
    s = Settings()
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
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    # A typo must fail at startup, not silently pick a default provider.
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()
