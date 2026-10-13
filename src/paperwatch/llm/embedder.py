"""Embedding layer: the offline fake used by tests and evals, and the Azure client.

The `Embedder` protocol keeps the pipeline free of the provider SDK, and the
Azure implementation clips and batches so no single request breaks the model's
input limits.
"""

import hashlib
import math
import re
from typing import Protocol

import openai

from paperwatch.config import EMBEDDING_DIM, Settings
from paperwatch.llm.ledger import Ledger

_WORD = re.compile(r"[a-z0-9]+")
# The embedding model reads at most 8,191 tokens per input, and a token is at least one UTF-8
# byte, so 8,000 bytes can never exceed it. This is exact without a tokenizer download (default
# CI is offline) and keeps roughly the first 1,500 words of a post, which carry its topic; the
# full text stays stored for keyword search. Chunking long posts is a later milestone.
EMBED_MAX_BYTES = 8000
# The API also caps the tokens of one request; 32 inputs of at most 8,000 tokens stay far below it.
EMBED_BATCH = 32


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def clip_for_embedding(text: str) -> str:
    # Cut on bytes, then drop a partial trailing character, so the result is valid text.
    return text.encode("utf-8")[:EMBED_MAX_BYTES].decode("utf-8", "ignore")


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
            resp = self._client.embeddings.create(model=self._deployment, input=batch)
            self._ledger.charge(self._deployment, max(0, resp.usage.total_tokens - estimate))
            # The SDK may answer out of order; restore input order and refuse a bad shape.
            for embedding in sorted(resp.data, key=lambda d: d.index):
                if len(embedding.embedding) != EMBEDDING_DIM:
                    raise ValueError(
                        f"expected {EMBEDDING_DIM} dimensions, got {len(embedding.embedding)}"
                    )
                out.append(embedding.embedding)
        return out
