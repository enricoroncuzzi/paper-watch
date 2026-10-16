"""Embedding layer: the offline fake used by tests and evals, the local model, and Azure.

The `Embedder` protocol keeps the pipeline free of any provider SDK. Production embeds
locally with an ONNX model (no key, no network once downloaded, no token cost); the
Azure implementation clips and batches so no single request breaks the model's input
limits. `make_embedder` picks one from settings.
"""

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol

import openai

from paperwatch.config import EMBEDDING_DIM, Settings
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


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


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
    key = settings.azure_openai_api_key
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
