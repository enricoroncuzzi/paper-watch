# Architecture

Paper Watch is one Python service around one Postgres store. Each unit has one purpose and a
narrow interface, and every external dependency sits behind a small protocol with a deterministic
fake, so the whole test and eval suite runs offline against a real Postgres + pgvector container.

## Units after M2

| Unit | Purpose | Depends on |
|---|---|---|
| `sources/*` | One client per source, owning pacing and retries (arXiv harvests OAI-PMH datestamp days); every request goes through one shared client that caps each body, bounds each request by a wall-clock deadline and follows a redirect only under its own policy (ADR 0010); arXiv and the feeds return normalised `SourceItem`s, Hugging Face and Semantic Scholar return quality signals for arXiv ids | HTTP only |
| `ingest` | Window, dedupe, classify, admit, embed, upsert; writes run records and a health verdict | sources, store, llm |
| `store` | Schema, Alembic migrations, repository; the only module that writes SQL | Postgres |
| `retrieval` | Hybrid search (pgvector + Postgres full-text, fused with reciprocal rank fusion), filters | store |
| `llm` | Local embedder and embedding-similarity classifier; chat models on a free OpenAI-compatible API behind the daily call cap and the token ledger; the Azure wrappers kept for the Azure proof | fastembed (ONNX Runtime), OpenAI SDK |
| `agent` | The LangGraph question-answering graph, the three read-only corpus tools, the per-answer source aliases and the deterministic citation check; the only place LangGraph is used | retrieval, llm, store |
| `api` | FastAPI routes, the streamed answer, health and readiness, the OpenAPI contract; the one place an agent failure becomes an error event | agent, store, llm |
| `evals` | Labelled synthetic corpus and questions; recall@k, MRR and nDCG@k against a committed baseline | retrieval, llm, store |
| `deploy/vps` | systemd user units for the daily ingestion (06:00 Europe/Rome, `--followup`) and the 05:15 backup, `backup.sh`, the morning `check.sh` | the `paperwatch ingest` CLI, Postgres |

`drafts`, `web` and `bench` arrive in later milestones. `infra` exists in part:
`docker-compose.yml` for local Postgres and `deploy/vps/` (systemd user units, `backup.sh`,
`check.sh`, described in `docs/deploy/vps.md`) for the daily ingestion on the shared VPS, which
ADR 0009 records. The API deployment and the Azure proof are M3 and M4. Domain units never import
a framework directly, so the framework comparison stays fair when `bench` is built; `agent` and
`llm/chat.py` are the only units that import LangChain or LangGraph.

## One ingestion run

`paperwatch ingest --live` reads `Settings` and builds one client per source plus the local
embedder and classifier, then `run_ingestion` takes the single-run Postgres advisory lock so two
runs cannot overlap. arXiv is harvested one datestamp day at a time through OAI-PMH starting at
the cursor day minus the one-day overlap (a datestamp's time-zone boundary can move a record
between days), and each response page is parked durably before the next request is sent; the
cursor advances per completed day to the next day's midnight, so today is fetched but never marked
complete. Each feed is fetched from its cursor minus the same kind of overlap. The
run then dedupes against the store, classifies each new item once with the embedding-similarity
classifier, admits it by tier and quality signal, embeds the admitted text locally, and upserts it
idempotently; the unique `(source, external_id)` key makes a replay insert nothing. After
fetching, the run re-checks the evidence of candidate and vetted-preprint papers whose age makes
them due and promotes them if it improved. The same Hugging Face listing carries more than
upvotes, so after admission the run stores each listed paper's GitHub repo and stars, project
page, comment count, organisation and listing time on the item, and appends a `signal_snapshots`
row with the numbers (upvotes, comments, stars, citations) whenever they changed since that item's
latest snapshot; a failure there is a warning, never an item's failed admission. It then writes a
run row with counts, tokens and a health verdict, advances each feed's cursor only when it fully
succeeded and the arXiv cursor after each completed day, and releases the lock.
A source outage stops that source for the run, is recorded as a dead letter over the whole missed
window, and makes the run `partial` without stopping the others; for a lab feed only a second miss
in a row does, because the first miss is a health warning and the next run retries the same window,
so one missed day loses nothing. A malformed entry is skipped and recorded once.

A run started with `--followup` (the scheduled unit does; a manual run does not) then finishes the
papers Semantic Scholar refused: it waits 5, 10, 20 and 40 minutes and then every 30, until nothing
is refused, 40 passes have run, the next pass would start at or after 22:00 Europe/Rome, or 16 hours
have passed since the start, whichever comes first. A pass asks S2 only for those ids and re-runs
only their admission, with the run's `now` as their check time; it never re-fetches arXiv, the feeds
or Hugging Face, and it records its state in `runs.health.followup` after every pass so the morning
check can tell a run that is waiting from one that hung (ADR 0008).

## The retrieval path

`paperwatch search QUERY --live` embeds the query with the same local model, then `hybrid_search`
runs two branches inside one transaction. The vector branch orders admitted rows by cosine
distance over the partial HNSW index; the text branch ranks full-text matches from the generated
`tsvector` with `ts_rank_cd`, over a query that ORs the stems of the question
(`any_term_tsquery`), so a paraphrase that shares one term still scores. Filters (tier, profile,
date) are applied inside both queries so they shape the candidates, not just the final list. The
two ranked id lists are fused with reciprocal rank fusion, which needs no score calibration
between the two systems, and the top hits are read back with their tier and admission basis. Only
admitted items are searchable; a query with no full-text match still returns the vector results.
`paperwatch eval` measures the quality of this path against `evals/baseline.json` and gates CI on
it.

## One question

`paperwatch ask "QUESTION" --live` and `POST /ask` run the same path. `run_question` opens one
database session, marks it `SET TRANSACTION READ ONLY` (the tools only read, so a bug cannot write
to the corpus), and builds this answer's `SourceRegistry`, token ledger and `CallGuard`. Then the
LangGraph graph loops between the chat model and the three read-only tools: `search_corpus` runs
the hybrid retrieval above with tier/profile/date filters, `get_item` opens one admitted item and
`compare_items` shows two to four of them. A tool registers each item it returns and hands the
model a short alias (`S1`, `S2`, ...), so a citation can only point at an item a tool actually
returned during this answer. Source text reaches the model only inside delimited `<source>` blocks
with markup escaped.

The loop is capped at `MAX_TOOL_ROUNDS`: after four tool rounds the model is called once more with
tools bound but `tool_choice="none"` and the `FINAL_ROUND_NOTE`, so it must answer from what it
has. Every model attempt passes the `CallGuard` first, which checks the daily `chat_calls` cap and
charges the per-answer token ledger after the call; a failed provider attempt is recorded too, and
the next configured model (Groq, when configured) answers.

The graph ends in a deterministic `check` node: `check_answer` compares every cited alias with the
registry, flags claims without a citation and tier mislabels, masks an unknown alias as `[?]`, and
returns the fixed no-support message when nothing valid is cited. `run_question` then yields that
`AnswerEvent` last and releases the session. The API formats the events as server-sent events
(`tool`, `token`, `answer`, `error`); the CLI prints the tokens, then the checked text under a rule
when it differs from what was streamed, then the sources and any flags. Only the `answer` event is
authoritative, and everything it carries is rendered as plain text with every link taken from
`Citation.url`. Errors become `error` events (`daily_cap`, `budget`, `model_unavailable`,
`internal`) rather than tracebacks, and the provider's own error text never reaches a client or the
log.
