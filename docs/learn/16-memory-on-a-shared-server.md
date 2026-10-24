# Memory on a shared server

On a small shared machine the expensive part of a service is often one library, here the embedding
model. Freeing a model inside a long-lived process does not reliably give its pages back to the
operating system, so the model lives in a child process that the API starts when a question
arrives and lets exit when idle: exiting is the one release the kernel always honours. The start
is hidden by starting the child while the chat model thinks. When the daily ingestion needs the
memory, the API does not stop: it answers from full-text search alone and says so. Memory caps then
bound the worst case: each unit has a ceiling, the shared slice has one for everything Paper Watch
runs, and the API side carries the higher OOM score so it is the one the kernel picks.

Measured on the shared VPS before this slice: the API with the model resident held 380 to 520 MB
all day, next to an ingestion that peaks near 707 MB, on a machine with about 1.3 GB available. An
in-process `del` plus a garbage collection did not return the pages, which is why the design is a
process, not a cache.

## The child process

`src/paperwatch/llm/embed_worker.py` is the child. `serve(embedder, inp, out)` reads one JSON line
(`{"texts": [...]}`) and writes one line back (`{"vectors": [[...]]}` or `{"error":
"<ExceptionType>"}`), and returns on EOF. The exception type only, never its message: a model error
can echo the question. `main()` raises its own `/proc/self/oom_score_adj` to 1000 first (Linux only;
a Mac has no `/proc`), then loads `LocalEmbedder` from the settings, then serves. The high score is
the child volunteering: when the shared slice runs out of memory the kernel kills it rather than
the ingestion.

`WorkerEmbedder` is the parent side, in `llm/embedder.py`:

- `prewarm()` starts the child in a background thread and returns at once, so the load (about a
  second cold) happens while the chat model is still thinking.
- `embed(texts)` clips each query to 600 bytes (`QUERY_EMBED_MAX_BYTES`; a document is 3,000, but a
  question is short), writes one line, and waits at most eight seconds (`select` on the pipe,
  then one read). A timeout, a dead child or an error line raises `EmbedderUnavailable`; a timeout
  and a crash kill the child, an error line does not, because the request was bad and the child is
  healthy. The next call spawns a fresh child, so one crash costs one search its semantic half.
- `answer_open()` counts the answers in flight. The idle clock starts at the end of the last
  embed and is reset when an answer closes.
- `reap_once()` kills the child when `should_stop()` is true (the search mode turned text-only) or
  when it has been idle for `EMBED_WORKER_IDLE_S` (600 s) with no answer open. A daemon thread
  calls it every two seconds while a child is alive.
- `stop()` is for shutdown and tests.

The protocol is deliberately line-oriented: the child is a separate process with its own memory,
so the only thing the parent needs is a request and an answer. `tests/helpers/fake_embed_worker.py`
speaks the same protocol with the deterministic `FakeEmbedder`, and `FAKE_WORKER_MODE=hang` or
`crash` reproduces the two failures, which is how the timeout and respawn tests run without a
model.

## Keyword-only while the ingestion runs

`src/paperwatch/api/search_mode.py` answers one question: may this search use the vector branch?
`SearchModeGate.current()` returns a `ModeDecision` (`mode`, `reason`, `since`) in this order:

1. A `runs` row with `status = 'running'` whose `started_at` is within `MAIN_PASS_MAX_AGE` (three
   hours, the morning check's own stale-row limit) means the ingestion's main pass is running:
   `text_only`, reason `ingestion`, `since` that start.
2. `MemAvailable` below 800 MB (`SEARCH_MODE_MIN_AVAILABLE_MB`) means the machine is short:
   `text_only`, reason `memory`. A live model child is exempt from that count: its own `RssAnon` is
   added back before the comparison, because killing it is what would free it. Without that, the
   API's own child alone could put the machine under the threshold (measured on the deployment,
   2026-10-24: 1104 MB idle, 700 to 830 MB with the child loaded), and the gate would turn
   keyword-only and kill the child it just paid to load.
3. PSI `some avg10` above 5.0 (`SEARCH_MODE_MAX_PSI_SOME_AVG10`) means processes are stalling on
   memory: the same decision.
4. A database error means `text_only`, reason `database`: a search that cannot read the run table
   must not spend the model's memory either.
5. Otherwise `hybrid`, reason `ok`.

A missing `/proc` file is ignored, which is the Mac: there the gate sees no pressure and answers
hybrid. The decision is cached for two seconds (`SEARCH_MODE_CACHE_S`), because a tool call is a
few database queries and the run table does not need reading before each one. The cache is a
soft bound: a pass starting mid-answer is picked up by the next tool round, and the memory caps,
not this cache, are the hard guarantee.

The usage side is in `agent/tools.py`: `search_corpus` asks `mode()` before every search. When the
answer is `text_only`, or when `WorkerEmbedder.embed` raises `EmbedderUnavailable`, it searches
with `embedder=None` (`hybrid_search` then fuses the text branch alone) and prefixes the tool
result with `TEXT_ONLY_NOTE`, so the model knows exact names and distinctive terms work best and
does not apologise for a semantic search it never had. A `SearchLog` records that, and the answer
event carries `search_mode`. `run_question` also sends a `notice` event before the first token, so
the page can say why.

## The failure path, and what a reader sees

| Failure | What happens |
|---|---|
| The child is slow or hangs | The 8 s timeout kills it; that search runs by keyword; the next search spawns a fresh child. |
| The child crashes or the OOM killer takes it (`-9`) | The dead pipe raises `EmbedderUnavailable`; that search runs by keyword; the next one respawns. |
| The ingestion starts while a child is alive | The reaper sees `should_stop()` at its next two-second pass and kills the child; the next search is keyword-only. |
| Memory is low but no ingestion runs | The gate reads `MemAvailable` and PSI, and the answer runs keyword-only. |
| The database is unreachable | The gate answers `text_only/database`; the answer still runs from the model's side, though its tools will fail. |

None of these ends the answer: a reduced answer with a notice is the design, not an error.

## The caps around it

The child is bounded by the API unit (`MemoryMax=700M`, `MemorySwapMax=0`), which is bounded by
the shared `paperwatch.slice` (`MemoryMax=1100M`) that the ingestion joins with one drop-in line.
The two unit files and the numbers, with the measurements behind them, are `docs/deploy/vps.md` section 15 and
ADR 0013. `docs/learn/15` covers the credential in front of all of this.

## What a test can see

`tests/unit/test_worker_embedder.py` runs the real protocol against the fake child, including the
OOM probe (Linux only) and the idle and stop kills with an injected clock.
`tests/unit/test_search_mode.py` covers each rule and the cache with fake readers. The integration
tests drive `hybrid_search(..., None)`, a keyword-only `search_corpus`, the `/status` route, the
`notice` event and the prewarm ordering with a scripted model. The real model is only in
`tests/model`, opt-in with `-m model`.
