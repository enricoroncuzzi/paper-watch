# Architecture

Paper Watch is one Python service around one Postgres store, reached over one HTTP API and read
through one Next.js page. Each unit has one purpose and a narrow interface, and every external
dependency sits behind a small protocol with a deterministic fake, so the whole test and eval suite
runs offline against a real Postgres + pgvector container.

## Units after M3

| Unit | Purpose | Depends on |
|---|---|---|
| `sources/*` | One client per source, owning pacing and retries (arXiv harvests OAI-PMH datestamp days); every request goes through one shared client that caps each body, bounds each request by a wall-clock deadline and follows a redirect only under its own policy (ADR 0010); arXiv and the feeds return normalised `SourceItem`s, Hugging Face returns an `HfPaper` per arXiv id (upvotes plus the listing extras) and Semantic Scholar returns venue and citation signals | HTTP only |
| `ingest` | Window, dedupe, classify, admit, embed, upsert; writes run records and a health verdict | sources, store, llm |
| `store` | Schema, Alembic migrations, repository; the only module that writes SQL | Postgres |
| `retrieval` | Hybrid search (pgvector + Postgres full-text, fused with reciprocal rank fusion), filters | store |
| `llm` | Local embedder and embedding-similarity classifier; the child process that hosts the embedder for the API (`embed_worker.py`); chat models on a free OpenAI-compatible API behind the daily call cap and the token ledger; the Azure wrappers kept for the Azure proof | fastembed (ONNX Runtime), OpenAI SDK |
| `suggestions` | The weekly picks: a pure gate-score-diversify ranking weighted from `config/suggestions.yaml`, the 30-second summary writer and the code checks on its quotes, numbers and names, and the card shapes the route serves (ADR 0014) | store, llm, domain |
| `drafts` | The LinkedIn draft: one OpenRouter-only call on the chat model or `PW_DRAFT_MODEL`, the prompt, and the code check over its numbers, quotes, links, first-person claims, attribution and copying; the point of view is stored with the draft and never logged (ADR 0015) | store, llm, suggestions |
| `agent` | The LangGraph question-answering graph, the three read-only corpus tools, the per-answer source aliases (restored from a thread for a follow-up) and the deterministic citation check; the only place LangGraph is used | retrieval, llm, store |
| `api` | The credential and host gate, the per-client limits, the paced SSE stream, the search-mode gate, the FastAPI routes (`/healthz`, `/readyz`, `/status`, `POST /ask` with `thread_id`, `GET /today` and the runs reads, `GET /suggestions` with its 5-minute cache, its background summary queue and its event route, `POST /drafts`, its regenerate route and the Drafts reads, `GET /threads` and one thread's turns, `GET /items` and one item's page), health and readiness, the committed OpenAPI contract; the one place an agent failure becomes an error event and a finished turn is stored (ADR 0016) | agent, store, llm |
| `web` | The Next.js page in `web/`: server-rendered sections (the today strip, the picks and Rising, Drafts, Corpus, Runs), the GitHub session guard on every server component and route handler, the streaming ask proxy, the thread reads and the conversation the sheet keeps with its Recent list, the corpus table and its filter and search proxy, the item page, the drafts dialog, and the server-only API client; the only caller of the API | the API over HTTPS |
| `evals` | Labelled synthetic corpus and questions; recall@k, MRR and nDCG@k for retrieval, and the four recorded-answer scores the QA eval replays through the citation check; each against a committed baseline | retrieval, llm, store |
| `deploy/vps` | systemd user units for the daily ingestion (06:00 Europe/Rome, `--followup`), the 05:15 backup, the API and its tunnel, the shared `paperwatch.slice` and the ingestion drop-in, `backup.sh`, the morning `check.sh`, the API `api-check.sh`, and the tunnel soak's `tunnel-soak.sh` probe with its `soak-stream.py` SSE test server and `paperwatch-soak-stream.service` unit | the `paperwatch ingest` and `serve` CLIs, Postgres |

`bench` and the Azure proof arrive in a later milestone. `infra` exists in part:
`docker-compose.yml` for local Postgres, `web/` for the Vercel project and `deploy/vps/` (systemd
user units for the ingestion, the API and its tunnel, the shared `paperwatch.slice`, the ingestion
drop-in, `backup.sh`, `check.sh`, `api-check.sh` and the soak's `tunnel-soak.sh` and
`soak-stream.py`, described in `docs/deploy/vps.md`), which ADR 0009 records and ADRs 0012 and 0013
extend. Domain units never import a framework directly, so the framework comparison stays fair when
`bench` is built; `agent` and `llm/chat.py` are the only units that import LangChain or LangGraph.

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

`paperwatch ask "QUESTION" --live` and `POST /ask` run the same path. `run_question` first asks
`SearchModeGate` for the search mode: a main-pass row in `runs` younger than three hours, or
`MemAvailable` plus the live model child's own memory below 800 MB (ADR 0013), or PSI `some avg10` above 5.0, switches this answer to keyword-only;
the page is sent a `notice` and the answer carries `search_mode: text_only`. On the API, each tool
call then opens its own short `SET TRANSACTION READ ONLY` session and closes it when the tool
returns, so an answer waiting for the model holds no database connection; the CLI keeps one
read-only session. Either way the answer builds its own `SourceRegistry`, token ledger and
`CallGuard`, and the LangGraph graph loops between the chat model and the three read-only tools:
`search_corpus` runs the hybrid retrieval above with tier/profile/date filters (the text branch
alone in keyword-only mode), `get_item` opens one admitted item and `compare_items` shows two to
four of them. A tool registers each item it returns and hands the
model a short alias (`S1`, `S2`, ...), so a citation can only point at an item a tool actually
returned during this answer. Source text reaches the model only inside delimited `<source>` blocks
with markup escaped.

The loop is capped at `MAX_TOOL_ROUNDS`: after four tool rounds the model is called once more with
tools bound but `tool_choice="none"` and the `FINAL_ROUND_NOTE`, so it must answer from what it
has. Every model attempt passes the `CallGuard` first, which checks the daily `chat_calls` cap and
charges the per-answer token ledger after the call; a failed provider attempt is recorded too, and
the next configured model (Groq, when configured) answers.

The embedding model is not in the API process: `WorkerEmbedder` starts
`python -m paperwatch.llm.embed_worker` when a hybrid answer arrives (prewarmed while the chat
model thinks), sends one clipped query per search, and lets the child exit after ten idle minutes,
never while an answer is open. When the ingestion needs the memory the gate turns text-only and
the reaper stops the child (`docs/learn/16`).

The graph ends in a deterministic `check` node: `check_answer` compares every cited alias with the
registry, flags claims without a citation and tier mislabels, masks an unknown alias as `[?]`, and
returns the fixed no-support message when nothing valid is cited. `run_question` then yields that
`AnswerEvent` last and releases the session. The API formats the events as server-sent events
(`tool`, `step`, `source`, `token`, `notice`, `answer`, `error`), pacing them with a `: ping`
comment every 15 s of silence and a 120 s deadline; the CLI prints the tokens, then the checked
text under a rule when it differs from what was streamed, then the sources and any flags. Only the
`answer` event is authoritative, and everything it carries is rendered as plain text with every
link taken from `Citation.url`. Errors become `error` events (`daily_cap`, `budget`,
`model_unavailable`, `internal`, `deadline`) rather than tracebacks, and the provider's own error
text never reaches a client or the log.

## The API on a public URL

`paperwatch serve --live` defaults to `127.0.0.1:8000` and warns on any other bind, and `ApiGate`
decides every request before FastAPI parses it: the credential, then the `Host` (loopback or
`PW_API_PUBLIC_HOST`), then a refusal of any `Origin`, then the method and content type, then the
16 KB body cap. `GET /healthz` is the one route without a credential, and it touches neither the
database nor a model.

`PW_API_AUTH=oidc` verifies a Vercel OIDC token against Vercel's published keys with the issuer,
audience and subject pinned; `hmac` verifies a 60-second token minted by `paperwatch api-token
--live`, and each one works once. `RequestLimiter` bounds every model-spending request per
credential subject per minute and per day, and two shared answer slots protect the process's
memory: a third answer gets 429 with a `Retry-After` instead of a queued connection.

The deployment publishes that loopback port through Tailscale in userspace
(`tailscaled --tun=userspace-networking`, no root and no TUN device), in its own
`paperwatch-tunnel.service` so a tunnel restart cannot stop an answer; the Funnel port mapping is
tailscaled state, set once by the runbook (`docs/deploy/vps.md`, ADR 0012). `api-check.sh` is the
acceptance check over the gate, the limits and the units. The read routes (`/today`, the runs,
`/threads`, `/items`) open one short read-only session each and never touch a model;
`GET /suggestions` serves a five-minute cache and fills a missing summary in the background; only
`POST /ask`, `POST /drafts` and a regenerate spend a model call on the request's own path, and
every one of those calls counts against the same daily cap and the same per-client limit.

## The page

`web/` is the Next.js App Router app and the only caller of the API. Every server component and
route handler checks the GitHub session itself, comparing the numeric GitHub id with the one
allowed id, and the browser never holds the API's address or credential: reads go through
`lib/api.ts` in a server component, a question through the `/api/ask` route handler, a thread read
through `/api/threads`, and `lib/api-token.ts` mints the credential on the server (a Vercel OIDC
token by default, HMAC when `PW_API_AUTH=hmac`). Vercel builds only a release commit on the
production branch (`web/scripts/vercel-ignore.sh`), so the page and the server change together and
a preview URL is never created.

One page holds the sections: the today strip from `/today` (the last run's verdict, its follow-up
state and the search mode), the picks and the Rising shelf from `/suggestions`, Drafts, the Corpus
table whose first page is rendered with the page, and Runs. A run detail and an item page are their
own routes. The ask sheet is one dialog that keeps the conversation, streams the answer with its
sources and trace, offers three fixed follow-ups and reads its thread back after a dropped stream,
and its Recent list reads `/threads`. The drafts dialog takes the point of view and shows the
check's numbered notes. Every end-to-end test runs against `web/fake-api/`, whose fixtures are
validated against the API's own models, and Playwright drives the same flows in Chromium and
WebKit, including an iPhone 15 (`docs/learn/17`).

## The weekly picks

`GET /suggestions` gates, scores, merges near-duplicates and diversifies on request. Reliability is
the gate; `momentum`, `recency`, `shareability`, `topic_fit` and `credibility` are weighted from
`config/suggestions.yaml`; a one-off event bonus is added for a promotion or a repository that
appeared; maximal marginal relevance then chooses five cards, the first being the lead, and three
Rising cards. The response is cached in the process for five minutes. A card without a summary
shows the abstract's excerpt while `SummaryQueue` writes the real one on one daemon thread,
pausing for thirty minutes when the provider is down and until UTC midnight when the daily cap is
spent. `POST /suggestions/{item_id}/events` records opened and dismissed; a shown card is recorded
by the route itself at most once an hour, with the rank, the score and the six components it ranked
by, so a stored list stays explainable. A dismissal or a draft removes the item from every later
list. `paperwatch suggest --live` prints the same list from the CLI.

## One LinkedIn draft

`POST /drafts` makes exactly one call, on the chat model or `PW_DRAFT_MODEL` when set, from the
item's stored text and the owner's point of view; `check.py` then checks the reply and the draft is
saved in the same request, so generating is saving and there is no separate save call. The check
counts its numbers, quotes, links, first-person claims, attribution, early notice and copied
sentences into numbered notes, and a note is something the reader judges, not a refusal. A failed
provider call saves nothing and answers 503 with a reason the page can show; the point of view is
stored with its draft, returned only by `GET /drafts/{draft_id}`, and never logged. A saved draft
is never edited: `POST /drafts/{draft_id}/regenerate` re-runs the stored inputs and point of view
into a new draft that links the old one (ADR 0015). `paperwatch draft ITEM_ID --live` runs the same
path and prints the notes.

## One conversation

A thread is our own two tables, not a LangGraph checkpointer (ADR 0016). `POST /ask` without a
`thread_id` starts one, and a finished answer is stored as the thread's next turn on the write
session factory `AgentDeps` carries; an aborted, dropped or failed answer stores nothing. A
follow-up sends the last six finished turns and at most 6,000 characters of their answers, and the
thread's source registry is persisted with it, so `S1` from the first answer is still citable in
the fifth. `GET /threads` is the Recent list and `GET /threads/{thread_id}` returns the finished
turns, each `answer` exactly the checked payload the stream sent, so a re-fetch after a dropped
stream shows what the server decided. The page's ask sheet keeps that conversation open across
turns and does the re-fetch itself (`web/src/lib/use-ask.ts`).

## The corpus table

`GET /items` pages admitted items by keyset on `(published_at, id)`, so a tie never repeats or
hides a row, or walks the full-text ranking by offset, because a ranking is a score and not a key.
It filters by tier, source and profile, and it never embeds: the corpus stays readable in
keyword-only mode and while the ingestion holds the model's memory. `GET /items/{item_id}` returns
one admitted item with its stored text, its extras, its tier history, its signal series, its
cached summary and its drafts. The page sends its filter, search and paging calls through
`/api/corpus`, and `/items/[id]` renders one item and reuses the drafts dialog.
