# LangGraph agents

A pipeline runs fixed steps in a fixed order; an agent lets the model choose the next step. LangGraph expresses that as a graph: nodes are functions over a shared state, edges say what runs next, and a conditional edge reads the state (here: "did the model ask for a tool?") to route. The loop model -> tools -> model is the classic tool-calling agent; what makes it production-shaped is everything around the loop: a cap on rounds so it cannot spin forever, a guard on every model call for cost and quota, a fallback model, and a deterministic final node that checks the answer before anyone sees it. State is plain typed data, so a trajectory test can assert exactly which tools ran in which order without a live model.

## The graph

`src/paperwatch/agent/graph.py` builds it in `build_agent(models, tools, registry, guard)`. There
are three nodes (`agent`, `tools`, `check`) and one conditional edge whose routing function is
`route`:

- `agent` prepends the system prompt from `prompts.py` and calls the model chain.
- `route` reads the last message: a tool call goes to `tools`, anything else goes to `check`.
- `tools` is LangGraph's `ToolNode` over the three read-only tools from Task 14, with
  `handle_tool_errors=True` so a bad argument or a `ToolException` reaches the model as an error
  message it can act on instead of ending the answer.
- `check` runs `check_answer(text, registry)` and stores the `CheckedAnswer`.

The state is a `TypedDict`: `messages` (appended through `add_messages`, so a node returns only the
new messages), `rounds` (how many model messages have asked for tools) and `answer` (the checked
result). The system prompt is added at every call and never stored, so a trajectory test sees
exactly what the model saw and a client's transcript holds only the question, the tool calls and
the answer.

One graph answers one question: `build_agent` binds that answer's `SourceRegistry` and token
ledger, so the first model call of a second question on the same graph raises a `RuntimeError`.
Without it, a second question could cite `S1` although no tool returned it for that question.

LangGraph's message streaming works because LangChain chat models detect a streaming callback
handler on the run and call `_stream` instead of `_generate` even from `.invoke()`. That is why
`ScriptedChatModel` implements both: the same scripted answer drives `answer_question` and
`stream_answer`, and the tests cannot accidentally pass in one path only.

## The loop and the cap

`MAX_TOOL_ROUNDS = 4`, and a round is one model message however many tool calls it carries. While
`rounds < 4`, the model is called with the tools bound, and a message that calls a tool spends one
round. The fifth call is the answer round: a `HumanMessage` carries `FINAL_ROUND_NOTE` ("You have
used all tool calls for this question. Answer now from the sources you already have.") and the
tools are still bound, with `tool_choice="none"`.

Keeping the definitions is not cosmetic. A request whose history holds tool calls and tool results
must still carry `tools`; OpenRouter's tool-calling guide says so, and a strict upstream would
answer the capped round with a 400, turning almost every capped question into `ModelUnavailable`.
`"none"` forbids the call instead (OpenRouter and Groq both document that value), and that round
counts as spent. Because a lenient provider can still return a call, the cap has two
provider-independent halves: the node drops any call the answer round contains, and `route`
refuses to send a tool call from a round beyond the cap. A model that ignores the note therefore
ends with its answer checked (`NO_SUPPORT` when it cited nothing) instead of another turn. Without
the cap, one question that keeps searching spends the free tier's daily quota.

A round is counted from the message the model returned, not from anything the tools said: if the
model answers with no tool call, `rounds` does not move, and an answer with no evidence still runs,
which is how the empty-corpus case reaches `check` and becomes the fixed no-support message.

## Every attempt is guarded

The provider chain is primary then optional fallback (ADR 0006), and the loop over it is written
out in `call_models`, not delegated to `Runnable.with_fallbacks()`. The reason is accountability:
every attempt is a request against a free tier, so `CallGuard.before_call` must run before each
one, `after_failure` must record the ones that raised, and `after_call` must record the real usage
of the one that answered and stamp it with the provider that produced it. A LangChain fallback
would make its own attempts where the guard cannot see them.

Two errors stop the chain instead of moving on. `DailyCapReached` and `BudgetExceeded` are
decisions about this answer, not provider trouble; another provider would spend from the same
budget or bypass the same cap, so they propagate. `before_call` and `after_call` sit outside the
try on purpose: a cap or budget decision must never be mistaken for a provider failure. Any other
exception is logged by its shape (exception class and HTTP status), recorded as a zero-token
`"failed"` row, and the next model is tried; when none is left the graph raises
`ModelUnavailable`. Neither the log nor that message carries the provider's text: an SDK error's
string is the whole response body, and OpenRouter's moderation errors echo the flagged question.
The exception itself stays chained to the error for a debugger. Recording the failure is itself
inside a `try`: a store error while writing that row is logged and dropped, because the 429 that
caused it is the error worth reporting.

Tools are bound with `parallel_tool_calls=False` (both OpenRouter and Groq accept the field, and
langchain-openai sends it with every request). That is defence in depth rather than the only
defence: `ToolNode` runs one message's parallel tool calls on a thread pool, and the tools share
one `Session` and one `SourceRegistry`, so `make_tools` already holds a lock around every tool
body (lesson 10). A dead primary is tried first on every model call, so it costs one failed,
counted request per call; the five-call bound keeps that small.

## The check node

The graph does not trust the model's text. Whatever the last message says, `check` runs
`check_answer` (lesson 11) against the registry this answer's tools filled, and stores a
`CheckedAnswer`: the text (with unknown aliases masked as `?`), the deduplicated citations, the
flags, and `supported`. An answer with no evidence leaves the graph as
`NO_SUPPORT = "I could not find anything in the corpus that answers this question."`, so the
worst case of a bad model is an honest refusal, not an invented citation.

## What a caller sees

`answer_question(graph, question)` runs the graph and returns the `CheckedAnswer` plus the tool
names in call order; it is what the CLI and the tests use. `stream_answer` is a generator of
`ToolEvent(name, args)` when the model decides to call a tool, `TokenEvent(text)` as the answer is
written, and exactly one final `AnswerEvent(answer)` (on success; a cap or budget error ends the
stream without one). The tokens are previews, and the streamed text can differ from the checked text
in more than one way: an unknown alias is masked, an uncited answer becomes `NO_SUPPORT`, text
written beside a tool call is streamed before a later model call starts, and a stream that dies
mid-answer can be followed by the fallback model's answer. So a client replaces what it streamed
with the `AnswerEvent`'s text when it arrives, never appends it. Lesson 13 covers how those events
become server-sent events.

## What the tests pin

`tests/integration/test_agent.py` drives the real graph, the real tools and a real Postgres against
`ScriptedChatModel`, so no test needs a key or the network. It covers a search then an answer (with
the cited item's id and two `chat_calls` rows), a search then `get_item` (and that the item's text
reached the model), recovery from an alias the registry never issued, exactly four tool rounds with
the fifth call bound to the tools with `tool_choice="none"` and carrying the final-round note, one
message with two calls counting as one round, a model that ignores the final note (no fifth tool
runs, no sixth model call, and the stream echoes no `ToolEvent` for the ignored call), the route
refusing a tool call after the answer round, an invented alias flagged in the answer, the no-support
answer for an empty corpus, a second question on the same graph refused, the daily cap refusing
before any model call and again before a fallback attempt, the budget stopping mid-answer after two
recorded calls, the fallback answering after the primary raised, a failed-usage write that does not
hide the provider error, provider error text kept out of the log and `ModelUnavailable`, a wrong
tool argument recovered from through an error message, every call bound with
`parallel_tool_calls=False`, the event order and tool names of `stream_answer` (one `ToolEvent` per
call, one `AnswerEvent` last, the streamed text equal to the final text), and the ledger charged
from the aggregated message when the answer is streamed rather than from the individual chunks.
