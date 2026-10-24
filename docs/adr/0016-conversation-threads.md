# 16. Conversation memory is our own two tables, with one registry per thread

Date: 2026-11-04

## Status

Accepted

Supersedes ADR 0007's "single-turn" paragraph: the graph is still built for one run, but a run now
carries a thread's earlier turns.

## Context

The page's Ask sheet can send a follow-up. Design section 10 asks for `POST /ask` to gain an
optional `thread_id`, with threads persisted in Postgres "a LangGraph checkpointer or an
equivalent table", and for a finished answer to be re-fetchable by its thread. Two constraints make
this more than a message list:

- **Citations must survive the turn.** An answer cites `S1`, `S2` by the aliases its tools issued,
  and `check_answer` trusts only the registry of the answer being checked. If a follow-up started
  from an empty registry, a citation to a source found in turn 1 would become `NO_SUPPORT`. Design
  section 15 (Q5) fixes the answer: one source registry per thread, persisted with it, so `S1` from
  turn 1 is citable and checkable in turn 5.
- **The API role writes only what it is granted.** The migration grants `paperwatch_api` INSERT on
  both tables, UPDATE on `threads` and USAGE on the turn sequence; a checkpointer would create its
  own tables and need DDL rights the API role must never have. The API's answer session is also opened read-only, so a turn cannot be written
  on it.

Section 15 (Q1) also settles the client behaviour: a disconnect still cancels the answer, so only
finished turns are stored, and the page re-fetches the thread's finished answers.

## Decision

**Two tables, no checkpointer.** `threads` holds the id, the title (the first question, at most 80
characters), the persisted registry and the timestamps. `thread_turns` holds one row per finished
turn: its number, the reader's question, the checked `AnswerData` payload, the search mode and the
time, with a unique `(thread_id, n)`. The grant follows the same helper as the other M3 tables
(`store/grants.py`), so the captain has no new admin step.

**Only finished turns are stored.** The write happens after the `answer` event and only for a
successful answer, in one transaction on a writable session factory (`AgentDeps.write_session_factory`),
never on the answer's read-only session. A failure is logged as `thread_save_failed` with the error
type only and the stream ends without a `thread` event: the reader keeps the checked answer
either way. An aborted or failed answer stores nothing, so a re-fetch never shows a turn that was
not checked.

**The registry is rewritten with every turn, and never trimmed.** Each saved turn carries the
thread's current aliases, so a follow-up that adds sources makes them citable in the turn after it
too. History is bounded instead: the last six turns and at most 6,000 characters of their answers,
oldest dropped first, because the per-answer token cap bounds one answer and not a conversation.
The graph's first-call guard counts from `turn_start`, the index of this turn's question, so a
follow-up is still recognisable as the first model call of its own run and a reused graph is still
refused.

**The stored question is the reader's own.** `run_question` builds the model prompt separately
(`prompt`), so the item block "Ask about it" adds is never what a turn stores; `item_id` is
pre-registered as `S1` only on a new thread, where a second `S1` cannot point at two papers.

**A follow-up's history is question and checked answer only.** An earlier `<source>` block is not
replayed: `HISTORY_NOTE` tells the model that earlier `[S#]` markers remain citable and that a
detail needs `get_item` before it is relied on, so the earlier sources are re-read rather than
trusted from memory.

**An unknown thread is a 404 before the stream.** The route reads the thread and refuses a missing
id with a 404, so no model call is spent answering into nothing; `run_question` loads the thread
again for the registry and history it needs.

## Consequences

- `GET /threads` (newest first, with a turn count) and `GET /threads/{id}` (every finished turn)
  need no model and no limiter, so the page can re-fetch after a dropped stream for free.
- The stream gains a `thread` event, `{"id", "turn"}`, as its last event on success; the contract
  version moves to 0.6.
- A turn is immutable, like a draft: regenerating a follow-up is asking again, which makes it a new
  turn, so a stored turn's check result can never describe different text.
- Storing the checked `AnswerData` means a re-fetch shows what the reader saw, without reading the
  corpus again. A field the payload adds later is a contract change, not a migration.
