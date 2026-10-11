import pytest

from paperwatch.config import EMBEDDING_DIM, Settings


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
    assert EMBEDDING_DIM == 1536
