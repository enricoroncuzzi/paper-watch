# Architecture

Paper Watch is one Python service around one Postgres store. Each unit has one purpose and a
narrow interface, and every external dependency sits behind a small protocol with a deterministic
fake, so the whole test and eval suite runs offline against a real Postgres + pgvector container.

## Units after M1

| Unit | Purpose | Depends on |
|---|---|---|
| `sources/*` | One client per source; returns normalised `SourceItem` records; owns pacing and retries (arXiv harvests OAI-PMH datestamp days) | HTTP only |
| `ingest` | Window, dedupe, classify, admit, embed, upsert; writes run records and a health verdict | sources, store, llm |
| `store` | Schema, Alembic migrations, repository; the only module that writes SQL | Postgres |
| `retrieval` | Hybrid search (pgvector + Postgres full-text, fused with reciprocal rank fusion), filters | store |
| `llm` | Local embedder and embedding-similarity classifier; the OpenAI SDK wrappers kept for the Azure proof; provider selected by configuration; token ledger and budget enforcement | fastembed (ONNX Runtime), OpenAI SDK |
| `evals` | Labelled synthetic corpus and questions; recall@k, MRR and nDCG@k against a committed baseline | retrieval, llm, store |

`agent`, `drafts`, `api`, `web`, `bench` and `infra` arrive in later milestones; only the local
`docker-compose.yml` exists so far. Domain units never import a framework directly, so the
framework comparison stays fair when `bench` is built.

## One ingestion run

`paperwatch ingest --live` reads `Settings` and builds one client per source plus the local
embedder and classifier, then `run_ingestion` takes the single-run Postgres advisory lock so two
runs cannot overlap. arXiv is harvested one datestamp day at a time through OAI-PMH from the day
after its cursor, one overlap day is re-read because a datestamp's time-zone boundary can move a
record between days, and each response page is parked durably before the next request is sent; the
cursor advances to the next day only once a whole past day is complete, so today is fetched but
never marked complete. Each feed is fetched from its cursor minus the same kind of overlap. The
run then dedupes against the store, classifies each new item once with the embedding-similarity
classifier, admits it by tier and quality signal, embeds the admitted text locally, and upserts it
idempotently; the unique `(source, external_id)` key makes a replay insert nothing. After
fetching, the run re-checks the evidence of candidate and vetted-preprint papers whose age makes
them due and promotes them if it improved, then writes a run row with counts, tokens and a health
verdict, advances each source's cursor only when its fetch fully succeeded, and releases the lock.
A source outage stops that source for the run, is recorded as a dead letter over the whole missed
window, and makes the run `partial` without stopping the others; a malformed entry is skipped and
recorded once.

## The retrieval path

`paperwatch search QUERY --live` embeds the query with the same local model, then `hybrid_search`
runs two branches inside one transaction. The vector branch orders admitted rows by cosine
distance over the partial HNSW index; the text branch ranks full-text matches from the generated
`tsvector` with `websearch_to_tsquery`. Filters (tier, profile, date) are applied inside both
queries so they shape the candidates, not just the final list. The two ranked id lists are fused
with reciprocal rank fusion, which needs no score calibration between the two systems, and the
top hits are read back with their tier and admission basis. Only admitted items are searchable; a
query with no full-text match still returns the vector results. `paperwatch eval` measures the
quality of this path against `evals/baseline.json` and gates CI on it.
