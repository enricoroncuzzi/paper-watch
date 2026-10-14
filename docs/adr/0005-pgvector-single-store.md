# 5. One Postgres + pgvector store for items, vectors, and full text

Date: 2026-10-14

## Status

Accepted

## Context

The design (section 4.2) needs four things kept together: the item records, their embeddings, the
full-text index over titles and bodies, and (later, in M4) run checkpoints. A managed vector
database would split the corpus in two: an item and its vector could disagree after a failed write,
a backup would cover only one half, and a query that combines a keyword filter with a similarity
search would need a round trip between systems. The corpus is small (thousands to tens of thousands
of admitted items) and the cloud budget is zero, so the operational cost of a second data store is
not paid back by scale. Hosted options considered: Pinecone (managed, no self-hosting on the
budget, and a second system to keep consistent) and Azure AI Search (vector plus keyword, but it
re-enters the Azure dependency the local-model decision avoids and was already deferred with the
Azure proof). Both are recorded as deferred, not rejected on merit.

## Decision

Items, embeddings, the full-text index, and later run checkpoints live in one Postgres database
with the `pgvector` extension. Every classified item is stored with a status: `irrelevant` (kept
only so it is never classified again), `candidate` (relevant, waiting for evidence, never
searchable), or `admitted` (has a tier and an embedding, searchable). Embeddings are written only
at admission, so a candidate or an irrelevant item never keeps a vector. Search is hybrid: a
cosine-distance HNSW index (`vector_cosine_ops`, `m=16`, `ef_construction=64`) over the admitted
rows plus a GIN index over a generated `tsvector`. The HNSW index is **partial** on
`WHERE status = 'admitted'`, and the retrieval query in Task 10 repeats that predicate literally so
the planner can use it. The schema keeps the record honest with database constraints rather than
application checks: a unique `(source, external_id)` key makes ingestion idempotent, and check
constraints refuse an admitted item without a tier or without an embedding, or a tier without a
recorded `admission_basis`. The migration writes the embedding column as the literal `Vector(384)`
and creates the extension itself.

Two limits are accepted. One embedding per item is computed over at most the first 3,000 bytes of
text (`LOCAL_EMBED_MAX_BYTES`, Task 7b, sized for the 1.9 GB VPS; 8,000 bytes with the Azure
embedder), so a long blog post is represented by its opening while its full text stays in the
`tsvector` index. Chunking long documents into passages with their own vectors, and the migration
version or `EMBEDDING_DIM` change that requires, is deferred to a later milestone and decided with
the retrieval evals. The design's `content_hash` column is also deferred until a later milestone
needs change detection beyond the arXiv version, which the `version` column already covers for
papers.

## Consequences

- One transactional store, one backup, and one connection string: an item, its tier history, and
  its embedding are written or rolled back together.
- An item that has only been classified can become relevant later without ever being classified
  again, because it is kept as `candidate` or `irrelevant` instead of discarded.
- The unique key, not application code, absorbs the overlap from re-fetched windows and racing
  runs; a duplicate insert is refused even under two concurrent runs.
- The schema must be migrated with Alembic, and a change to `EMBEDDING_DIM` or the embedding model
  is a new migration plus a full re-embed and recalibration, never an edit of `0001_initial.py`.
- Long documents are searchable by their full text but embedded only by their opening; passage
  chunking stays an open item for a later milestone.
- `content_hash` change detection is not available yet; a feed post that changes is not re-read.
