"""Embedding layer: the offline fake used by tests and evals, the local model, and Azure.

The `Embedder` protocol keeps the pipeline free of any provider SDK. Production embeds
locally with an ONNX model (no key, no network once downloaded, no token cost); the
Azure implementation clips and batches so no single request breaks the model's input
limits. `make_embedder` picks one from settings.
"""

import contextlib
import hashlib
import json
import math
import re
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import openai
import structlog

from paperwatch.config import (
    EMBED_WORKER_IDLE_S,
    EMBED_WORKER_TIMEOUT_S,
    EMBEDDING_DIM,
    QUERY_EMBED_MAX_BYTES,
    Settings,
)
from paperwatch.domain import SourceItem
from paperwatch.llm.ledger import Ledger

_WORD = re.compile(r"[a-z0-9]+")
# Azure's embedding model reads at most 8,191 tokens per input, and a token is at least one UTF-8
# byte, so 8,000 bytes can never exceed it. This is exact without a tokenizer download (default
# CI is offline) and keeps roughly the first 1,500 words of a post, which carry its topic; the
# full text stays stored for keyword search. Chunking long posts is a later milestone.
EMBED_MAX_BYTES = 8000
# The API also caps the tokens of one request; 32 inputs of at most 8,000 tokens stay far below it.
EMBED_BATCH = 32
# The local model's memory grows with input length: measured at batch 1 with 2 threads and the
# arena off (Apple M2, 2026-10-13), peak RSS was about 400 MB at 3,000 bytes, 520 MB at 4,000
# and 1,200 MB at 8,000, too much next to Postgres on a 1.9 GB VPS. 3,000 bytes keeps a whole
# arXiv title and abstract (abstracts are capped at 1,920 characters) and the opening ~500 words
# of a blog post; the full text stays stored for keyword search.
LOCAL_EMBED_MAX_BYTES = 3000


log = structlog.get_logger()
# The reaper checks the child every this many seconds; it is a bound on how long a stopped child
# can hold the memory, not a precision timer.
REAPER_INTERVAL_S = 2.0
# How long a kill waits for the child to be reaped before giving up on it.
KILL_TIMEOUT_S = 5.0


def _read_status(pid: int) -> str | None:
    # /proc exists on the Linux server only; elsewhere the child reports no figure.
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            return status.read()
    except OSError:
        return None


def rss_anon_mb(status: str | None) -> float | None:
    """The child's own anonymous memory in MB, or None when it cannot be read."""
    if status is None:
        return None
    for line in status.splitlines():
        if line.startswith("RssAnon:"):
            try:
                return int(line.split()[1]) / 1024
            except (IndexError, ValueError):
                return None
    return None


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbedderUnavailable(RuntimeError):
    """The embedding worker cannot answer now, so a search must fall back to keywords."""


def _read_line(stream: Any, timeout_s: float) -> str | None:
    # `select` on the pipe, then one read: a blocking `readline` would ignore the timeout and a
    # hung model would hold the answer until the deadline. POSIX only, like the deployment.
    ready, _, _ = select.select([stream], [], [], timeout_s)
    if not ready:
        return None
    line = stream.readline()
    return line if isinstance(line, str) else None


class WorkerEmbedder:
    """Embed through a child process that loads the model and exits when it is no longer needed.

    The parent holds no model, so the model's memory is returned by the kernel the moment the
    child exits. `prewarm` starts it in the background while the chat model thinks, `embed` sends
    one clipped query, and the reaper kills it when the caller says stop or when it has been idle
    for `idle_s` and no answer is open.
    """

    def __init__(
        self,
        command: Sequence[str] = (sys.executable, "-m", "paperwatch.llm.embed_worker"),
        *,
        idle_s: float = EMBED_WORKER_IDLE_S,
        timeout_s: float = EMBED_WORKER_TIMEOUT_S,
        query_max_bytes: int = QUERY_EMBED_MAX_BYTES,
        should_stop: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
        read_status: Callable[[int], str | None] = _read_status,
    ) -> None:
        self._command = list(command)
        self._idle_s = idle_s
        self._timeout_s = timeout_s
        self._query_max_bytes = query_max_bytes
        self._should_stop = should_stop
        self._clock = clock
        self._read_status = read_status
        self._state = threading.Lock()
        self._io = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._answers = 0
        self._last_used: float | None = None
        self._reaper: threading.Thread | None = None
        self._stop = False

    def prewarm(self) -> None:
        """Start the child without waiting for it: the model load happens during the chat call."""
        with self._state:
            if self._alive_locked():
                return
        threading.Thread(target=self._spawn, name="embed-prewarm", daemon=True).start()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clipped = [clip_for_embedding(text, self._query_max_bytes) for text in texts]
        # One request/response at a time: the child reads whole lines, and two threads writing
        # them at once would interleave a request and its answer.
        with self._io:
            process = self._spawn()
            if process is None or process.stdin is None or process.stdout is None:
                raise EmbedderUnavailable("the embedding worker is not running")
            try:
                process.stdin.write(json.dumps({"texts": clipped}) + "\n")
                process.stdin.flush()
                line = _read_line(process.stdout, self._timeout_s)
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._kill()
                raise EmbedderUnavailable("the embedding worker failed") from exc
            if line is None:
                self._kill()
                raise EmbedderUnavailable("the embedding worker timed out")
            if not line:
                self._kill()
                raise EmbedderUnavailable("the embedding worker died")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self._kill()
                raise EmbedderUnavailable("the embedding worker sent no JSON") from exc
            if "error" in response:
                # The child is healthy and the request was not; keep it for the next query.
                raise EmbedderUnavailable(f"the embedding worker refused: {response['error']}")
            vectors = response.get("vectors")
            if (
                not isinstance(vectors, list)
                or len(vectors) != len(texts)
                or not all(isinstance(vector, list) for vector in vectors)
            ):
                raise EmbedderUnavailable("the embedding worker returned a bad shape")
            with self._state:
                self._last_used = self._clock()
            return vectors

    @contextmanager
    def answer_open(self) -> Iterator[None]:
        """Count an open answer: the child may grow, but it may not exit mid-answer."""
        with self._state:
            self._answers += 1
        try:
            yield
        finally:
            with self._state:
                self._answers = max(0, self._answers - 1)
                # Idle time counts from the end of the last answer, not from its last embed.
                self._last_used = self._clock()

    def reap_once(self) -> None:
        """Kill the child when it is no longer needed. The reaper calls this every 2 s."""
        with self._state:
            process = self._process
            if process is None:
                return
            if process.poll() is not None:
                self._process = None
                return
            if self._answers > 0:
                return
            now = self._clock()
            idle = self._last_used is not None and now - self._last_used >= self._idle_s
        if not idle:
            try:
                stop = self._should_stop()
            except Exception as exc:  # noqa: BLE001 - a mode read must not kill the reaper
                # The child must still exit when idle even if the mode check is broken.
                log.warning("embed_worker_stop_check_failed", error_type=type(exc).__name__)
                stop = False
            if not stop:
                return
        with self._state:
            # Another thread may have replaced the child, and an answer may have opened, while the
            # mode check ran; the child may not exit mid-answer.
            if self._process is process and self._answers == 0:
                self._terminate_locked(process)

    def stop(self) -> None:
        with self._state:
            self._stop = True
        self._kill()

    def alive(self) -> bool:
        with self._state:
            return self._live_locked() is not None

    def child_rss_mb(self) -> float | None:
        """The live child's own anonymous memory: what an exit would give back.

        None when no child is alive or its status cannot be read, so the gate counts nothing it
        cannot see; RssFile is left out because that page cache is already reclaimable.
        """
        with self._state:
            process = self._live_locked()
            if process is None:
                return None
            # Read while the lock is held: a reaped child's pid can be reused, and the figure has
            # to be this child's memory, not a stranger's.
            return rss_anon_mb(self._read_status(process.pid))

    def _alive_locked(self) -> bool:
        return self._live_locked() is not None

    def _live_locked(self) -> subprocess.Popen[str] | None:
        process = self._process
        if process is None:
            return None
        if process.poll() is not None:
            self._process = None
            return None
        return process

    def _spawn(self) -> subprocess.Popen[str] | None:
        with self._state:
            if self._stop:
                return None
            if self._alive_locked():
                return self._process
            try:
                # S603: the command is this interpreter plus a fixed module path, never caller
                # input; the child is ours.
                process = subprocess.Popen(  # noqa: S603
                    self._command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    close_fds=True,
                    text=True,
                )
            except OSError as exc:
                # A missing interpreter or module must not take the API down; the search falls
                # back to keywords for this call.
                log.warning("embed_worker_spawn_failed", error_type=type(exc).__name__)
                return None
            self._process = process
            self._last_used = self._clock()
            if self._reaper is None or not self._reaper.is_alive():
                self._reaper = threading.Thread(
                    target=self._reap_loop, name="embed-reaper", daemon=True
                )
                self._reaper.start()
            return process

    def _reap_loop(self) -> None:
        while True:
            time.sleep(REAPER_INTERVAL_S)
            self.reap_once()
            with self._state:
                if self._stop or self._process is None:
                    return

    def _kill(self) -> None:
        with self._state:
            process = self._process
            if process is not None:
                self._terminate_locked(process)

    def _terminate_locked(self, process: subprocess.Popen[str]) -> None:
        # Closing stdin lets a healthy child see EOF and exit on its own; kill() is for a child
        # that is hung or already gone. stderr is inherited, so the journal keeps its traceback.
        with contextlib.suppress(OSError):
            if process.stdin is not None:
                process.stdin.close()
        with contextlib.suppress(OSError):
            if process.poll() is None:
                process.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=KILL_TIMEOUT_S)
        self._process = None


def clip_for_embedding(text: str, max_bytes: int = EMBED_MAX_BYTES) -> str:
    # Cut on bytes, then drop a partial trailing character, so the result is valid text.
    return text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def embedding_text(item: SourceItem) -> str:
    # The one text embedded for an item, by the classifier and at admission alike, so its
    # stored vector and its relevance score describe the same thing.
    return f"{item.title}\n{item.text}"


class FakeEmbedder:
    """Deterministic hashed bag-of-words. Offline stand-in for tests and CI evals only."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * EMBEDDING_DIM
            for w in _WORD.findall(text.lower()):
                h = int.from_bytes(hashlib.sha256(w.encode()).digest()[:8], "big")
                v[h % EMBEDDING_DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def make_azure_client(settings: Settings) -> openai.AzureOpenAI:
    # A whitespace-only key is as useless as an empty one, so strip before the check.
    secret = settings.azure_openai_api_key
    key = secret.get_secret_value() if secret else None
    if not (settings.azure_openai_endpoint and key and key.strip()):
        raise RuntimeError("PW_AZURE_OPENAI_ENDPOINT and PW_AZURE_OPENAI_API_KEY are required")
    return openai.AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=key,
        api_version=settings.azure_openai_api_version,
        max_retries=3,
        # One embedding or classification, not the SDK default of 10 minutes per read.
        timeout=30.0,
    )


class AzureEmbedder:
    def __init__(self, client: openai.AzureOpenAI, deployment: str, ledger: Ledger) -> None:
        self._client, self._deployment, self._ledger = client, deployment, ledger

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = [clip_for_embedding(t) for t in texts[i : i + EMBED_BATCH]]
            # Reserve at least one token so a ledger already at its cap cannot charge 0 and
            # still reach the API; the provider's reported usage then tops the ledger up to
            # the real cost.
            estimate = max(1, sum(len(t) for t in batch) // 3)
            self._ledger.charge(self._deployment, estimate)
            # text-embedding-3 models answer 1536 dimensions unless asked for fewer; the store's
            # column is EMBEDDING_DIM. Shortened vectors are still in Azure's own space, never
            # comparable with the local model's, so one corpus uses one provider.
            resp = self._client.embeddings.create(
                model=self._deployment, input=batch, dimensions=EMBEDDING_DIM
            )
            self._ledger.charge(self._deployment, max(0, resp.usage.total_tokens - estimate))
            # The SDK may answer out of order; restore input order and refuse a bad shape.
            for embedding in sorted(resp.data, key=lambda d: d.index):
                if len(embedding.embedding) != EMBEDDING_DIM:
                    raise ValueError(
                        f"expected {EMBEDDING_DIM} dimensions, got {len(embedding.embedding)}"
                    )
                out.append(embedding.embedding)
        return out


class TextModel(Protocol):
    """The part of fastembed's TextEmbedding that LocalEmbedder uses, so tests can stub it."""

    def embed(self, documents: list[str], batch_size: int) -> Iterable[Iterable[float]]: ...


def make_local_model(settings: Settings) -> TextModel:
    # Imported here so the offline tests and the Azure path never load ONNX Runtime. Building
    # the model downloads it into the cache dir unless it is already there (about 180 MB);
    # with HF_HUB_OFFLINE=1 a missing model is an error instead of a download.
    from fastembed import TextEmbedding

    return TextEmbedding(
        settings.local_embedding_model,
        cache_dir=settings.local_embedding_cache_dir,
        threads=settings.local_embedding_threads,
        enable_cpu_mem_arena=settings.local_embedding_cpu_arena,
    )


class LocalEmbedder:
    """Embeds on this machine: no network, no API key, and nothing charged to the ledger."""

    def __init__(self, model: TextModel, batch_size: int) -> None:
        self._model, self._batch = model, batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clipped = [clip_for_embedding(t, LOCAL_EMBED_MAX_BYTES) for t in texts]
        out: list[list[float]] = []
        for raw in self._model.embed(clipped, batch_size=self._batch):
            vector = [float(x) for x in raw]
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(f"expected {EMBEDDING_DIM} dimensions, got {len(vector)}")
            # The model does not normalise; unit length makes a dot product a cosine, as with
            # every other embedder here.
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0.0:
                raise ValueError("the model returned a zero vector")
            out.append([x / norm for x in vector])
        if len(out) != len(texts):
            raise ValueError(f"expected {len(texts)} vectors, got {len(out)}")
        return out


def make_embedder(settings: Settings, ledger: Ledger) -> Embedder:
    # One switch for every live command; only the Azure embedder spends ledger tokens.
    if settings.embedding_provider == "azure":
        return AzureEmbedder(make_azure_client(settings), settings.embedding_deployment, ledger)
    return LocalEmbedder(make_local_model(settings), settings.local_embedding_batch)
