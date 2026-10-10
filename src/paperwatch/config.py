"""All runtime configuration. The only module that names models and deployments."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIM = 1536


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PW_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://paperwatch:paperwatch@localhost:5432/paperwatch"
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    embedding_deployment: str = "text-embedding-3-small"
    classifier_deployment: str = "gpt-4o-mini"
    max_tokens_per_run: int = 200_000
    semantic_scholar_api_key: str | None = None
    log_level: str = "INFO"
    user_agent: str = "paper-watch/0.1 (+https://github.com/enricoroncuzzi/paper-watch)"


@lru_cache
def get_settings() -> Settings:
    return Settings()
