"""All runtime configuration. The only module that names models and deployments."""

from datetime import timedelta
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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
# arXiv's bulk-harvest interface (OAI-PMH) lists records by last-modification datestamp, one
# day at a time, so a paper is seen the day it is announced and again when it is revised.
ARXIV_OAI_URL = "https://oaipmh.arxiv.org/oai"
# arXivRaw carries every version with its exact date, unlike the plain arXiv format.
ARXIV_OAI_PREFIX = "arXivRaw"
# Each run re-reads one day before its cursor so a datestamp's time-zone boundary cannot miss a
# record; the store's unique key absorbs the overlap.
ARXIV_OAI_OVERLAP = timedelta(days=1)
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

# Chat providers speak the OpenAI chat-completions shape, so only the base URL, the key and the
# model id change between them (ADR 0006). OpenRouter is primary because its free tier needs no
# card; Groq is an optional fallback for the day a free model is withdrawn or rate-limited.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Provisional default, checked against the public OpenRouter catalogue on 2026-10-19: present,
# priced at 0, and listing `tools` in `supported_parameters`, which the agent's tools need.
# Task 18 measures three candidates on the QA set and may replace this id.
CHAT_MODEL = "google/gemma-4-31b-it:free"
# Tool rounds the agent may take before it is called once more without tools and must answer;
# the cap turns a model that loops on its tools into a final answer instead of a hung request.
MAX_TOOL_ROUNDS = 4

# Connection budget for `paperwatch serve` (one local user, a few tabs at most). One answer holds
# a pooled connection for its whole life, which can be minutes while the model writes, so the pool
# allows several at once and a short wait: a full pool becomes an `internal` event quickly instead
# of a request hanging for the driver's 30 s default.
ANSWER_POOL_SIZE = 6
ANSWER_POOL_OVERFLOW = 2
ANSWER_POOL_TIMEOUT_S = 3.0
# The cap check and the usage write never wait behind an answer: they get their own tiny pool, and
# a row is written in milliseconds.
USAGE_POOL_SIZE = 2
# libpq's connect timeout: a paused or unreachable database fails a probe or a request in seconds.
CONNECT_TIMEOUT_S = 3
# Readiness checks are cheap and must stay cheap: two may run at once, each bounded by this
# timeout. A check past the bound answers 503 and is abandoned, so a stalled database cannot hold
# the probe for the driver's own timeout.
READY_CONCURRENCY = 2
READY_TIMEOUT_S = 2.0


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
    # Keys are SecretStr so a Settings object in a log or a traceback shows `**********`, not the
    # key; every call site reads them with .get_secret_value().
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    embedding_deployment: str = "text-embedding-3-small"
    classifier_deployment: str = "gpt-4o-mini"

    # Safety cap: a run stops calling the model once it has spent this many tokens.
    max_tokens_per_run: int = 200_000

    # Chat (M2). The key is unset by default and comes only from the environment; without it a
    # chat call refuses to start rather than failing deep inside an HTTP request.
    openrouter_api_key: SecretStr | None = None
    chat_model: str = CHAT_MODEL
    # Groq answers only when both the key and a model id are set: Groq's free models change too
    # often to pin a default here, and a guessed id would fail on every fallback attempt.
    groq_api_key: SecretStr | None = None
    groq_chat_model: str | None = None
    # A free model can stall; a timeout turns that into a fallback instead of a hang. There is no
    # retry setting: the SDK's own retries would be invisible to the daily cap (ADR 0006).
    chat_timeout_s: float = Field(default=60.0, gt=0)
    # One response's output is capped too: the ledger is only charged after a call, and a streamed
    # response is not bounded by `chat_timeout_s` (httpx applies it per read, not to the whole
    # stream), so without this one runaway generation can pass the answer budget in a single call
    # or stream until the context limit.
    chat_max_output_tokens: int = Field(default=4096, ge=1)
    # Per-answer cap, charged through the same Ledger as the ingestion runs: one answer is one
    # graph run, so a tool loop cannot spend without bound.
    chat_max_tokens_per_answer: int = Field(default=60_000, ge=1)
    # 800 sits under the 1,000 free requests a day OpenRouter allows with the captain's credit:
    # the gap is headroom for questions asked at the same time, not for hidden SDK retries.
    chat_max_calls_per_day: int = Field(default=800, ge=1)

    # Optional key for Semantic Scholar (works without it, with lower rate limits).
    semantic_scholar_api_key: SecretStr | None = None

    log_level: str = "INFO"
    # Sent with every HTTP request so source sites can identify us.
    user_agent: str = "paper-watch/0.1 (+https://github.com/enricoroncuzzi/paper-watch)"


@lru_cache
def get_settings() -> Settings:
    # Cached: settings are built once per process and shared by every caller.
    return Settings()
