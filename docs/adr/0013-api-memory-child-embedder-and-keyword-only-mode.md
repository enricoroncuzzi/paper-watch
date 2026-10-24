# 13. The API holds no model, and answers by keyword while the ingestion needs the memory

Date: 2026-10-24

## Status

Accepted

## Context

With the embedding model resident, the API held 380 to 520 MB all day on a machine with about
1.3 GB available, next to an ingestion that peaks at 706.8 MB. Freeing the model inside the
process (dropping the reference and collecting) did not return the pages in the measurements, so an
in-process cache cannot promise the ingestion its memory back. The two cannot both fit.

The measurements also ruled out two systemd relationships. `Conflicts=` between the API and the ingestion
stops the ingestion, which is the one outcome the deployment cannot accept; `MemoryHigh=` on the
API without swap stalls it instead of killing it. What works is one shared ceiling with no swap:
under pressure the cgroup OOM killer picks a process, and with the API side's higher OOM score it
picks the API. The design target is that the API gives way, never the run.

The API must keep answering during the 06:00 pass. The retrieval eval measures recall@5 0.97 for
hybrid search and 0.82 for the text branch alone, so keyword-only is a real but usable reduction,
provided the reader is told.

## Decision

- **The model lives in a child process.** `src/paperwatch/llm/embed_worker.py` loads the local
  model, writes `1000` to its own `/proc/self/oom_score_adj`, and answers one JSON request line at
  a time. `WorkerEmbedder` starts it on demand, prewarms it while the chat model thinks, clips a
  query to 600 bytes, waits 8 s for an answer and kills a hung or dead child; the next search
  respawns it. The child exits after ten idle minutes, counted from the end of the last embed and
  never while an answer is open, and the reaper kills it at its next two-second pass when the
  search mode turns keyword-only. An exit is the one release the kernel always honours.
- **The mode comes from the run record and the kernel.** `src/paperwatch/api/search_mode.py`
  answers `text_only` when a `runs` row is `running` and younger than three hours (reason
  `ingestion`, with `since`), when `MemAvailable` is below 800 MB or PSI `some avg10` is above 5.0
  (reason `memory`), or when the run table cannot be read (reason `database`); otherwise
  `hybrid`. While a model child is alive, the gate adds that child's own `RssAnon` to
  `MemAvailable` first, because that is the memory its exit gives back: measured on 2026-10-24, the
  idle machine read 1104 MB available with the API at 130 MB RSS and its child at 213 to 219 MB
  `RssAnon` (46 MB `RssFile`, 276 MB peak), which left 700 to 830 MB with the child loaded, at or
  below the threshold, so an idle API would otherwise turn keyword-only and kill its own child.
  The decision is cached for two seconds. A missing `/proc` is ignored, which is the
  Mac. Each `search_corpus` asks again, so an ingestion starting mid-answer switches the next
  round; a child that fails makes that one search keyword-only, and the answer says so with a
  `notice` event and `search_mode`.
- **Each tool call has its own short read-only session.** With `AgentDeps.read_session_factory`
  set, `make_tools` opens a session, runs `SET TRANSACTION READ ONLY` as its first statement,
  closes it when the tool returns, and the answer holds no connection while it waits for the
  model. The CLI, the eval and the tests keep one read-only session. The API's read pool is four
  connections (`READ_POOL_SIZE`).
- **Caps are nested, and the ingestion is untouched.** `paperwatch-api.service` has
  `MemoryMax=700M`, `MemorySwapMax=0`, `OOMScoreAdjust=900`; `paperwatch.slice` has
  `MemoryMax=1100M`, `MemorySwapMax=0`; the ingestion joins it through one drop-in line
  (`Slice=paperwatch.slice`) and its own unit is not edited. There is no `Conflicts=`, `Requires=`,
  `BindsTo=`, `PartOf=` or `After=` between the units, and no `MemoryHigh=` on the API. The tunnel
  agent has its own 150M outside the slice.
- **Cold start is hidden, and measured.** Prewarm runs before the first model call, so the 0.8 s
  spawn-to-first-vector measured on the Mac (warm cache, 2026-10-24) sits under the chat model's
  first token. The live acceptance run records the same number.

## Consequences

- During a main pass, a question is answered from full-text search alone and labelled; recall is
  the text branch's 0.82, not 0.97, which is the price of never stopping the ingestion.
- The API's process holds no model, so its idle memory is the FastAPI/uvicorn footprint; the model's
  memory exists only while a child is alive. A crash of the child costs one search its semantic
  half, not the answer.
- The shared slice is 1100M: if the measured peak of either side grows past it, one of them is
  OOM-killed rather than the machine swapping. That is deliberate and visible in the journal, and
  the server measurement and the overlap rehearsal are the check.
- The API needs `/proc` for the memory signals; on a machine without it the gate reports hybrid,
  and the memory caps are the only guard.
