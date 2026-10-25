"""All runtime configuration. The only module that names models and deployments."""

# Absolute import of the standard library module, not paperwatch/logging.py.
import logging
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository-relative paths resolve from this file (src/paperwatch/config.py), never from the
# working directory: `paperwatch eval` already works from anywhere, and a live command that read
# `config/profiles.yaml` from the cwd died with a FileNotFoundError everywhere else.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = REPO_ROOT / "config" / "profiles.yaml"
FEEDS_PATH = REPO_ROOT / "config" / "feeds.yaml"
ADMISSION_PATH = REPO_ROOT / "config" / "admission.yaml"
# The suggestion weights and windows (Task 34). A file rather than constants, because what the list
# prefers is an editorial choice the captain changes without a release.
SUGGESTIONS_PATH = REPO_ROOT / "config" / "suggestions.yaml"

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
# The width of `items.venue`. Real venue names are short, but a malformed signal can carry a whole
# paragraph, and a write the column cannot hold fails the admission and dead-letters the item, so
# the merge clips to this width. The model and the clipping share the one number.
ITEM_VENUE_MAX_CHARS = 256
# The width of `items.hf_org`. Hugging Face's organisation is free text, and a write the column
# cannot hold would cost the whole listing step for every paper, so the client clips to this width
# and the model shares the number (review round 1).
HF_ORG_MAX_CHARS = 256
# The largest value an int4 column holds. A listing count beyond it is dropped like any other
# malformed extra instead of failing the write, and the client and the store share this number.
INT4_MAX = 2_147_483_647
# Keyless Semantic Scholar (captain decision 2026-10-13): a paper's evidence is re-checked less
# often as it ages. Each pair is (maximum paper age, re-check interval): daily for the first 30
# days, weekly up to 90 days, monthly up to a year. About 0.131 x N checks a day for N papers.
S2_CADENCE = (
    (timedelta(days=30), timedelta(days=1)),
    (timedelta(days=90), timedelta(days=7)),
    (timedelta(days=365), timedelta(days=30)),
)
# A daily timer that starts a few seconds earlier than yesterday's must not skip a day: a paper
# checked at 06:00:20 yesterday is due at 06:00:00 today, 23 h 59 min 40 s later. So a paper has to
# clear its interval minus this slack. Two hours is far shorter than a day, so a weekly paper is
# pulled in by at most two hours.
S2_RECHECK_SLACK = timedelta(hours=2)
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
# Follow-up passes (ADR 0008): the keyless pool sometimes refuses every attempt for hours
# (measured 06:03 and 09:52 on 2026-10-23), and a refused paper stays due, so a scheduled run may
# wait and ask again for exactly those papers instead of losing a day of evidence. The first
# retries come soon, because a burst of minutes is common; the waits then grow one step at a
# time, one scale up from the client's own 5/10/20/40 s backoff.
FOLLOWUP_DELAYS_S = (300, 600, 1200, 2400)  # waits before passes 1 to 4: 5, 10, 20, 40 minutes
FOLLOWUP_INTERVAL_S = 1800  # every later pass waits 30 minutes
# 40 passes is a bound, not a plan: 06:00 to 22:00 is 16 hours, which the time-of-day limit ends
# first (the ramp takes 1 h 15 min, then 29 more passes fill the rest), so a fully refused day
# is bounded by the clock the captain set and this cap only guards a misconfigured deadline.
FOLLOWUP_MAX_PASSES = 40
# No pass starts at or after this hour in Europe/Rome (the server's own zone, named because a
# 22:00 local limit must not move with a DST change). A late Persistent catch-up must not retry
# into the night on a machine another service shares during the day.
FOLLOWUP_DEADLINE_HOUR = 22
# The window is also capped relative to the start, not only by the clock face: a catch-up that
# starts at 00:30 would otherwise sleep until 19:45 and outlive the unit's timeout in the middle
# of a wait. 16 h is the 06:00 to 22:00 span itself, and `TimeoutStartSec` must stay above it.
FOLLOWUP_MAX_WINDOW = timedelta(hours=16)
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
# The default, chosen by measurement on 2026-10-21 (Task 18): the best recorded QA scores of the
# free models with tools (ADR 0006 holds the table). A model id is configuration, so a withdrawn
# model is a settings change, and `paperwatch qa-eval` is what detects an answer-quality loss.
CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Tool rounds the agent may take before it is called once more without tools and must answer;
# the cap turns a model that loops on its tools into a final answer instead of a hung request.
MAX_TOOL_ROUNDS = 4
# The width of `chat_calls.model`: a usage row stores the provider's model id, and the guard clips
# it so a provider that names its model on every streamed chunk cannot fail the write.
CHAT_CALL_MODEL_MAX_CHARS = 128
# Defaults for the two chat limits. They are constants as well as settings so the QA eval can
# build its own guard from the same numbers without reading the developer's `.env`; the settings
# below still override both by name.
CHAT_MAX_TOKENS_PER_ANSWER = 60_000
CHAT_MAX_CALLS_PER_DAY = 800
# The single-key advisory lock that serialises the daily-cap check and its insert across the API
# and the CLI. Two answers reading "799 calls" at the same instant would both pass the cap and both
# reach the provider, so the count and the reservation happen under one transaction lock. The value
# is ASCII "PAWCHAT!", arbitrary but fixed.
CHAT_CALLS_LOCK_KEY = 0x5041_5743_4841_5421

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

# Answer pacing (Task 27). Six model-spending requests a minute and 100 a UTC day bound what one
# caller can spend, two concurrent answers bound the memory they hold, and the deadline is well
# under Vercel Hobby's 300 s function limit. iOS and proxies drop a stream that says nothing, so a
# silent answer gets a comment line every 15 s.
REQUESTS_PER_MINUTE = 6
REQUESTS_PER_DAY = 100
CONCURRENT_ANSWERS = 2
ANSWER_DEADLINE_S = 120
SSE_HEARTBEAT_S = 15

# The embedding model in a child process (Task 28). With the model resident the API held 380-520 MB
# all day next to an ingestion that peaks near 707 MB, and freeing it in process gave nothing back
# to the kernel; exiting is the one release that always works. It exits after ten idle minutes,
# never while an answer is open, and one question is clipped to 600 bytes because that is a query,
# not a document.
EMBED_WORKER_IDLE_S = 600
EMBED_WORKER_TIMEOUT_S = 8.0
QUERY_EMBED_MAX_BYTES = 600
# The search mode is re-read at most this often, so a minute of answers does not query the run
# table and /proc on every tool call. The ingestion's memory pressure can hit within that window;
# the memory caps, not this cache, are the hard guarantee.
SEARCH_MODE_CACHE_S = 2.0
SEARCH_MODE_MIN_AVAILABLE_MB = 800
SEARCH_MODE_MAX_PSI_SOME_AVG10 = 5.0
# A row still marked running after this long is a killed run, not a pass to wait out; it is the
# morning check's own stale-row limit.
MAIN_PASS_MAX_AGE = timedelta(hours=3)
# The API's read-only pool for one tool call: one per concurrent answer plus slack, because the
# session is closed as soon as the tool returns.
READ_POOL_SIZE = 4

# Conversation memory (Task 40). A follow-up sees the last finished turns only, and at most this
# much of their answers: the per-answer token cap bounds one answer, not a conversation, so the
# history is bounded before it reaches the model. The registry is not bounded: every alias a
# thread ever issued stays citable, which is what keeps S2 from turn 1 valid in turn 5.
THREAD_HISTORY_TURNS = 6
THREAD_HISTORY_CHARS = 6_000
# A thread is named after its first question; the Recent menu shows one line, so it is clipped.
THREAD_TITLE_MAX_CHARS = 80

# The request gate (Task 26). 16 KB is far above the largest request the API takes (a points of
# view box is capped at 2,000 characters), so a bigger body is a client bug or an attack.
API_MAX_BODY_BYTES = 16_384
# Vercel rotates the keys that sign its OIDC tokens; an unknown key id triggers a refetch, but
# never more often than this, so an unauthenticated caller cannot use the API to hammer Vercel's
# key endpoint.
JWKS_REFETCH_MIN_S = 300
# Clock skew between Vercel and this server, applied to the token's exp, nbf and iat claims.
TOKEN_LEEWAY_S = 30
# An HMAC token is minted for one check and expires at once: it is the fallback credential for
# local operation and the tunnel rehearsal, not a long-lived key.
HMAC_TOKEN_TTL_S = 60
HMAC_AUDIENCE = "paperwatch-api"
# The names a browser uses for this machine; the tunnel's own name comes from PW_API_PUBLIC_HOST.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


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
    # An optional override for the draft chain's model; unset means `chat_model`. A draft sends no
    # training opt-out, so the provider may train on and retain its prompt, point of view included,
    # which the owner accepted (ADR 0015).
    draft_model: str | None = None
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
    chat_max_tokens_per_answer: int = Field(default=CHAT_MAX_TOKENS_PER_ANSWER, ge=1)
    # 800 sits under the 1,000 free requests a day OpenRouter allows with the captain's credit:
    # the gap is headroom for questions asked at the same time, not for hidden SDK retries.
    chat_max_calls_per_day: int = Field(default=CHAT_MAX_CALLS_PER_DAY, ge=1)

    # Optional key for Semantic Scholar (works without it, with lower rate limits).
    semantic_scholar_api_key: SecretStr | None = None

    # Follow-up pass bounds (ADR 0008), so a drop-in on the server can shorten or lengthen the
    # window without a release. The ramp of the first four waits is a constant: it is the shape
    # of a burst, not a knob. The deadline hour can be moved earlier but never past 22:00, which
    # is the captain's decision; the 16 h cap in `Followup` is what keeps every start inside the
    # unit's timeout.
    followup_max_passes: int = Field(default=FOLLOWUP_MAX_PASSES, ge=1)
    followup_interval_s: int = Field(default=FOLLOWUP_INTERVAL_S, ge=60)
    followup_deadline_hour: int = Field(default=FOLLOWUP_DEADLINE_HOUR, ge=1, le=22)

    # The API credential (Task 26): a token Vercel signs for one project's production deployment,
    # verified against Vercel's published keys, or the HMAC fallback for local operation and the
    # tunnel rehearsal. One credential at a time, never both.
    api_auth: Literal["oidc", "hmac"] = "oidc"
    vercel_oidc_issuer: str | None = None
    vercel_oidc_audience: str | None = None
    vercel_oidc_subject: str | None = None
    api_hmac_secret: SecretStr | None = None
    # The tunnel's host name, added to the loopback names the gate accepts.
    api_public_host: str | None = None

    log_level: str = "INFO"
    # Sent with every HTTP request so source sites can identify us.
    user_agent: str = "paper-watch/0.1 (+https://github.com/enricoroncuzzi/paper-watch)"

    @field_validator("log_level")
    @classmethod
    def _canonical_log_level(cls, value: str) -> str:
        # `logging` accepts WARN and FATAL as aliases, but uvicorn looks the name up in its own
        # table and raises KeyError for them, so `serve` would crash where `ask` works. Store the
        # canonical name so both agree, and refuse a typo instead of failing at the first log.
        number = logging.getLevelName(value.strip().upper())
        if not isinstance(number, int):
            raise ValueError(f"unknown log level: {value}")
        return logging.getLevelName(number)


@lru_cache
def get_settings() -> Settings:
    # Cached: settings are built once per process and shared by every caller.
    return Settings()
