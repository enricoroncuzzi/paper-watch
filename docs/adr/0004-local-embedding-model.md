# 4. Corpus embeddings run locally with one pinned 384-dimensional model

Date: 2026-10-13

## Status

Accepted

## Context

The corpus stores one embedding per admitted item, and vectors from different models cannot be
compared, so the embedding model is fixed for the life of the corpus: changing it means
re-embedding everything. The cloud budget is zero, and production runs on a single VPS (2 vCPU,
1.9 GB RAM, no swap) next to Postgres and a scraper that uses about 150 MB during the day. A scout
on 2026-10-13 (`data/awakening-zero-cost-scout/report.md`) measured seven local models on 139 real
arXiv abstracts. `ibm-granite/granite-embedding-small-english-r2` (384 dimensions, Apache-2.0) was
the best relevance classifier (held-out macro F1 0.866, against 0.760 for the keyword classifier)
and statistically tied for retrieval (recall@5 0.452, MRR 0.844). The captain's own VPS benchmark
at batch 1 and 2 threads measured it at 25.3 documents per second with peak memory at most 517 MB.
Hosted embedding APIs, free or paid, can withdraw or re-version a model without notice, and the
smallest local language model that classifies at all needs about 1.1 GB and scores below both.

## Decision

Corpus embeddings are computed locally with `ibm-granite/granite-embedding-small-english-r2`
through fastembed on ONNX Runtime, with 2 threads, batch 1, the CPU memory arena off, and inputs
clipped to 3,000 UTF-8 bytes (measured: about 400 MB peak at 3,000 bytes, 1,200 MB at 8,000).
`EMBEDDING_DIM` is 384. Relevance is classified by cosine similarity between an item and a written
description of each profile, with a per-profile threshold in `config/profiles.yaml` calibrated for
this model; the code refuses to use those thresholds with any other embedding model. The keyword
classifier stays as the offline test default and a fallback, and the Azure OpenAI embedder and
classifier stay behind a provider setting for the time-boxed Azure proof, which runs the same
local model inside its container. Nothing local is charged to the token ledger. The model files
are downloaded once into a configurable cache directory (`PW_LOCAL_EMBEDDING_CACHE_DIR`), so a
deployment can ship them pre-filled and set `HF_HUB_OFFLINE=1`.

## Consequences

- No API key, network call or per-item cost at ingestion or query time, and no provider can change
  the model under the corpus.
- Same-dimension fallbacks exist (`BAAI/bge-small-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`):
  switching needs a re-embed of the corpus and new thresholds, but no schema change.
- A long blog post is represented by its first 3,000 bytes (about 500 words) instead of the 8,000
  an Azure call would take; the full text stays in full-text search.
- The thresholds started from the scout's value with descriptions rewritten from its labelling
  criteria, so they must be recalibrated on a labelled set before relevance is trusted as a gate.
- Default CI never loads the 180 MB model; tests that do are opt-in (`pytest -m model`).
