# 7. The agent is a small LangGraph loop with a capped round count and a code check

Date: 2026-10-20

## Status

Accepted

## Context

M2 turns the corpus into a place to ask questions (spec section 3.4): "what's new on agent
harness design?", "compare these two papers". The answer must be grounded, because a model asked
to cite sources cites fluently whether or not the source exists: it can name an item it never
retrieved, invent an id, or write a confident sentence with no source at all. The product also has
to run on a free chat tier (ADR 0006), so a question has hard limits: a per-answer token budget, a
daily request cap, and a provider that can fail at any moment.

The tools exist already (Task 14, which has no ADR of its own): `search_corpus`, `get_item` and
`compare_items`, all read-only, all returning `<source>` blocks with escaped markup and
program-issued aliases (`S1`, `S2`, ...). The checker exists too (Task 15): `check_answer(text,
registry)` compares a finished answer against the sources this answer's tools actually returned.
What was missing was the part that decides when to search, when to open an item, and when to
answer.

## Decision

The agent is a LangGraph graph in `src/paperwatch/agent/graph.py`, single-turn: one question, one
answer, no conversation memory and no checkpointer. Follow-up questions need a thread id and
persisted state, and they arrive with M3's page, not before.

The graph has three nodes (`agent`, `tools`, `check`) and one conditional edge, whose routing
function is `route`. `agent` calls the model with the system prompt from `prompts.py`; `route`
reads the last message and sends it to `tools` when the model asked for a tool, or to `check` when
it answered; `tools` is LangGraph's `ToolNode` over the Task 14 tools; `check` runs `check_answer`
and stores the `CheckedAnswer` in the state. `check` is last and deterministic, so an answer with
no valid citation leaves the graph as the fixed `NO_SUPPORT` message, never as uncited prose.

A graph holds one run: its `SourceRegistry` and token ledger are bound at build time, so a second
run on the same graph is refused with a `RuntimeError` whatever the first one retrieved, a retry
after a provider failure included (the ledger is already spent).

State is typed and plain (`messages`, `rounds`, `answer`), which is what makes the trajectory
tests cheap: a scripted model replays an exact sequence of tool calls, and the test asserts which
tools ran, in which order, how many model calls were made, and how many `chat_calls` rows were
written, with no key and no network.

Four limits sit around the loop:

- **Tool rounds are capped** at `MAX_TOOL_ROUNDS = 4`. A round is one model message, however many
  tool calls it carries. The fifth model call is the answer round: it carries `FINAL_ROUND_NOTE`
  and the tool definitions with `tool_choice="none"` (a request that carries tool history must
  also carry `tools`, and OpenRouter and Groq both document `"none"` as "do not call a tool").
  That round counts as spent, so the cap cannot be escaped: `route` refuses to run a tool call
  after it, and the node drops any call the reply still contains, which keeps it out of the
  events and the tool list too. A model that ignores the note therefore gets its answer checked
  (`NO_SUPPORT` when it cited nothing) instead of another turn.
- **Every attempt is guarded** by the Task 13 `CallGuard`: `before_call` checks the daily cap,
  `after_call` records the real usage and charges the per-answer ledger, `after_failure` records an
  attempt that raised. The fallback chain is an explicit loop rather than
  `Runnable.with_fallbacks()`, because `.with_fallbacks()` hides each attempt from the guard: the
  guard must see every request, and it must know which provider answered. `DailyCapReached` and
  `BudgetExceeded` stop the answer instead of moving to the next model, since another provider
  would spend the same budget or bypass the same cap: they are decisions about this answer, not
  provider trouble. Around `after_failure` the loop swallows a store error into a logged warning,
  so a database problem while recording a 429 cannot replace the 429 the caller needs to see.
- **Parallel tool calls are disabled** with `parallel_tool_calls=False` on the bound tools. Both
  providers accept the field (it is in OpenRouter's chat request schema and in Groq's chat
  completions schema) and langchain-openai sends it with every request. This is defence in depth:
  the tools already serialize themselves with a lock (Task 14), because `ToolNode` would otherwise
  run one message's tool calls concurrently over a shared session and registry.
- **The model's text is not trusted as an answer.** `stream_answer` emits tokens as the model
  writes them, but the last event is always one `AnswerEvent` carrying the checked answer, and a
  client replaces the streamed text with it. A flag is shown beside the sentence it names; a
  `NO_SUPPORT` answer hides its flags, because they describe text the reader will never see.

Prompts live in `src/paperwatch/agent/prompts.py` (versioned in the repository, never built from
configuration), and one of them is a product rule: cite every sentence, topic and summary
sentences included, because real answers from the Task 15 review left exactly those uncited.

The data rule stands (captain decision 2026-10-16): a question sends the question, CC0 arXiv
abstracts, and the stored text of lab-blog posts to the free chat provider, only to answer the
captain privately. Every tool result carries the source's tier, title and URL, so a lab post is
always attributed and visibly `lab_blog`; M2 never republishes that text.

## Alternatives considered

- **A fixed retrieve-then-answer pipeline** (one search, one model call): cheaper and simpler, but
  it cannot open an item for detail and cannot compare two, which are two of the spec's example
  questions. It is also not an agent, and the milestone exists to show the difference.
- **LangGraph's prebuilt `create_react_agent`**: less code, but it hides the three things this
  project wants to show and test: the guard on every attempt, the round cap with its final note,
  and the deterministic check node. It also owns the loop, so a fallback chain that the guard sees
  would have to be bolted on outside it.
- **`Runnable.with_fallbacks()`** for the provider chain: the obvious LangChain answer, rejected
  because it makes its own attempts invisibly to `CallGuard`, and the daily cap would undercount
  them.
- **Passing database ids or arXiv ids to the model**: ids are strings a model can invent, and a
  citation to an invented id looks true. Per-answer aliases from `SourceRegistry` make the only
  citable things the ones a tool actually returned.
- **An LLM judge for citations**: another model call, non-deterministic and unmeasurable in CI.
  The checker is code, and its limits are listed in lesson 11.
- **Conversation memory now**: a checkpointer and a thread id are real M3 needs (the page shows a
  conversation), and nothing in M2's CLI or eval uses them. When history arrives, the first-run
  check in `call_agent` (`len(state["messages"]) == 1`) stops identifying the first model call,
  because the initial state will already carry messages, and must be replaced.

## Consequences

- The agent is offline-testable end to end: trajectory, cap, fallback, budget and streaming tests
  run against Postgres and a scripted model, so default CI never calls a provider.
- The worst case for one answer is five model calls, each bounded by
  `chat_max_output_tokens`, so one question cannot silently cost more than the per-answer budget
  plus one call.
- A provider outage costs one failed, counted attempt per configured model, then `ModelUnavailable`
  or a fallback answer; it never hangs and never falls through to an unchecked answer. A dead
  primary costs one failed request per model call of the answer (the primary is tried first every
  time), which the per-answer bound of five calls keeps small.
- The round cap is a hard stop even when the model ignores the note, so a question that genuinely
  needs five searches gets a thinner answer. The alternative, an uncapped loop, is a free-tier
  quota spent by one bad question.
- The citation check is a heuristic (lesson 11): it guarantees aliases, evidence and tier labels,
  not truth. Its known limits are documented rather than fixed one round at a time.
- `stream_answer` may emit tokens the checked answer then discards: an uncited answer becomes
  `NO_SUPPORT`, text can precede a tool call, and a stream that dies mid-answer can be followed by
  the fallback model's answer. The last event is authoritative, so a client replaces the streamed
  text with it, never appends; showing the flags instead would require rendering the model's
  ungrounded prose.
