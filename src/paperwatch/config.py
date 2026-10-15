"""All runtime configuration. The only module that names models and deployments."""

from datetime import timedelta
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Vector size of the embedding model; the DB column and the model must agree on it. 384 is the
# local production model below and both of its same-size fallbacks (bge-small-en-v1.5,
# all-MiniLM-L6-v2), so a fallback needs a re-embed but no schema change.
EMBEDDING_DIM = 384
# The corpus embedding model (Apache-2.0, run locally through fastembed on ONNX Runtime; captain
# decision 2026-10-13). The relevance thresholds in config/profiles.yaml are calibrated for this
# exact model, so the embedding classifier refuses to run on any other.
LOCAL_EMBEDDING_MODEL = "ibm-granite/granite-embedding-small-english-r2"
# A deterministic failure (for example a text the embedding model refuses) must not be retried
# by every daily run forever: after this many failed attempts the item is dead-lettered.
MAX_ITEM_ATTEMPTS = 3
# Keyless Semantic Scholar (captain decision 2026-10-13): a paper's evidence is re-checked less
# often as it ages. Each pair is (maximum paper age, re-check interval): daily for the first 30
# days, weekly up to 90 days, monthly up to a year. About 0.131 x N checks a day for N papers.
S2_CADENCE = (
    (timedelta(days=30), timedelta(days=1)),
    (timedelta(days=90), timedelta(days=7)),
    (timedelta(days=365), timedelta(days=30)),
)
# arXiv announces papers hours to ~3 days after submission (longer over holidays), and its
# query filters on submission date, so each run re-reads this much before the arXiv cursor.
ARXIV_ANNOUNCEMENT_LAG = timedelta(days=7)
# A feed post can appear after its own date (CDN cache, feed regeneration, scheduled posts
# dated at drafting, zone-less dates read as UTC), so each feed is re-read this far back too.
FEED_OVERLAP = timedelta(days=7)
# A feed that answers but has published nothing for this long is probably dead or moved.
FEED_STALE_AFTER = timedelta(days=90)
# Venue acceptance typically lands 4-9 months after the preprint, so venue and citations are
# re-checked for a year after submission; peer-reviewed papers are never re-checked.
S2_RECHECK_HORIZON = timedelta(days=365)
# Hugging Face listing dates re-read by every run (about 31-62 requests whatever the corpus
# size); upvotes found there apply to any re-check target, whatever its age.
HF_RECHECK_HORIZON = timedelta(days=30)
# Keyless Semantic Scholar: at most this many due papers are looked up per run (25 batches of
# 500, at least 10 s apart); the most overdue go first and the rest stay due for the next run.
S2_MAX_PAPERS_PER_RUN = 12_500
# Single-key Postgres advisory lock (`pg_try_advisory_lock`) that lets one ingestion run at a
# time: a session-level lock on a dedicated connection survives the run's many commits, so a
# second overlapping run sees it held and skips instead of closing the first run's row or
# spending the same token budget. The value is ASCII "PAPRWTCK", arbitrary but fixed.
INGEST_LOCK_KEY = 0x5041_5052_5754_434B


class Settings(BaseSettings):
    # Each field is read from an env var with the PW_ prefix (database_url <- PW_DATABASE_URL),
    # falling back to a local .env file, then to the default below. Unknown vars are ignored,
    # and empty ones (PW_X= copied from .env.example) count as unset, so the default applies.
    model_config = SettingsConfigDict(
        env_prefix="PW_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    # Database (default matches docker-compose.yml)
    database_url: str = "postgresql+psycopg://paperwatch:paperwatch@localhost:5432/paperwatch"

    # Which embedder and which relevance classifier a live run uses. Local needs no key, no
    # network after the model download, and no token budget; azure is kept for the Azure proof.
    embedding_provider: Literal["local", "azure"] = "local"
    classifier_provider: Literal["embedding", "keyword", "azure"] = "embedding"

    # Local embedding model. The cache dir is where fastembed keeps the downloaded model; set it
    # to a pre-filled directory (for example one baked into the image) so a run never downloads.
    # Unset means fastembed's default, a folder in the system temp directory.
    local_embedding_model: str = LOCAL_EMBEDDING_MODEL
    local_embedding_cache_dir: str | None = None
    # Sized for the 2-vCPU, 1.9 GB VPS: memory grows with the batch, so embed one text at a time,
    # and the ONNX Runtime arena (which keeps freed memory for reuse) stays off.
    local_embedding_threads: int = Field(default=2, ge=1)
    local_embedding_batch: int = Field(default=1, ge=1)
    local_embedding_cpu_arena: bool = False

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
