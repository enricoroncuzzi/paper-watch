import pytest

from paperwatch.config import EMBEDDING_DIM, Settings


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PW_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/pw")
    monkeypatch.setenv("PW_MAX_TOKENS_PER_RUN", "12345")
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@localhost:5432/pw"
    assert s.max_tokens_per_run == 12345
    assert s.azure_openai_endpoint is None
    assert not hasattr(s, "max_cost_eur_per_run")  # EUR accounting arrives in M4


def test_embedding_dim_is_single_constant() -> None:
    assert EMBEDDING_DIM == 1536
