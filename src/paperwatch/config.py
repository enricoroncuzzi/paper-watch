"""All runtime configuration. The only module that names models and deployments."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Vector size of the embedding model; the DB column and the model must agree on it.
EMBEDDING_DIM = 1536


class Settings(BaseSettings):
    # Each field is read from an env var with the PW_ prefix (database_url <- PW_DATABASE_URL),
    # falling back to a local .env file, then to the default below. Unknown vars are ignored,
    # and empty ones (PW_X= copied from .env.example) count as unset, so the default applies.
    model_config = SettingsConfigDict(
        env_prefix="PW_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    # Database (default matches docker-compose.yml)
    database_url: str = "postgresql+psycopg://paperwatch:paperwatch@localhost:5432/paperwatch"

    # Azure OpenAI: endpoint and key have no default, so they must come from the environment.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    embedding_deployment: str = "text-embedding-3-small"
    classifier_deployment: str = "gpt-4o-mini"

    # Safety cap: a run stops calling the model once it has spent this many tokens.
    max_tokens_per_run: int = 200_000

    # Optional key for Semantic Scholar (works without it, with lower rate limits).
    semantic_scholar_api_key: str | None = None

    log_level: str = "INFO"
    # Sent with every HTTP request so source sites can identify us.
    user_agent: str = "paper-watch/0.1 (+https://github.com/enricoroncuzzi/paper-watch)"


@lru_cache
def get_settings() -> Settings:
    # Cached: settings are built once per process and shared by every caller.
    return Settings()
