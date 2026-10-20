# Streaming APIs

A model writes its answer token by token, and the first words can arrive in a second while the whole answer takes ten. Streaming sends each piece as soon as it exists. Server-sent events are the simplest way over HTTP: one long response of `event:` and `data:` lines that a browser reads natively, one direction only, which is all an answer needs. The stream carries different event types, so the page can show "searching the corpus..." while tools run, then the text, then the checked citations at the end, because citations can only be checked once the whole answer exists. A stream is a resource: when the client disconnects, the server must stop and release its database session.

## The same answer, three ways

`src/paperwatch/api/app.py` is the one place an agent failure becomes an `ErrorEvent` and the one
place the database session is released, and both surfaces use it:

- `POST /ask` (`create_app`) formats `run_question`'s events as server-sent events.
- `paperwatch ask "QUESTION" --live` (`src/paperwatch/cli.py`) prints the same events to a
  terminal, and prints tool calls and errors on stderr so a redirected answer stays clean text.
- `paperwatch serve --live` runs the API with uvicorn. It binds to `127.0.0.1` by default because
  the API has no sign-in until M3.

The graph, the tools, the guard and the checker do not know which surface is consuming them.

## The event stream

`run_question(deps, question, filters)` is a generator. It opens one session, marks it
`SET TRANSACTION READ ONLY` (the tools only read, so a bug cannot write to the corpus), builds a
fresh `SourceRegistry`, token ledger, `CallGuard` and tool set for this answer, and yields
`stream_answer`'s events. A new `SourceRegistry` per answer is what makes `S1` mean "the first
source this answer retrieved": a second question cannot cite the first one's aliases.

Four events reach a client, and each one is a Pydantic model in `api/app.py` (`ToolData`,
`TokenData`, `AnswerData` with `CitationData` and `FlagData`, `ErrorData`). The models build the
payloads and are the committed OpenAPI contract, so M3's generated client gets typed events and
the code cannot drift from the document. The `text/event-stream` media type is an `itemSchema`
for one parsed event (the `event` name plus a JSON `data` string with the payload's schema), not
the shape of the response body: an SSE client parses one message at a time, and an envelope
holding `event` and `data` keys together does not exist on the wire.

| Event | Data | Meaning |
|---|---|---|
| `tool` | `name`, `args` | The model called a corpus tool. `args` is model-authored. |
| `token` | `text` | A piece of the model's answer as it is produced. |
| `answer` | `text`, `supported`, `citations`, `flags` | The checked answer. Always last, unless an error follows. |
| `error` | `code`, `message` | `daily_cap`, `budget`, `model_unavailable` or `internal`. |

The `answer` event is authoritative, and that is the most important rule for a client. Streamed
tokens are what the model wrote before the check; the checked text can differ, because an alias no
tool returned is masked to `[?]` and an answer with no evidence is replaced by the fixed
`NO_SUPPORT` message. M2 has no per-model-call boundary on `token` events (a final round, or a
fallback model, can produce a second run of words), so a client replaces its buffer with
`answer.text` rather than appending, and a new page can revisit the boundary in M3.

`run_question` also turns exceptions into error events instead of letting a traceback escape a
stream that has already started (the status code is 200; the error arrives in the body). The
`model_unavailable` event names providers and exception classes only. The provider's own text stays
on `ModelUnavailable.__cause__`, for a debugger, and is never logged or serialised: an SDK error's
body can echo the question the provider refused. The `internal` branch logs the exception type,
the cause's type and the code site (`file:line:function`), never a message, a variable value or a
traceback, for the same reason.

## Why SSE and not WebSockets

An answer is one-directional and ends. Server-sent events are plain HTTP: no handshake, no
protocol upgrade, a native `EventSource` in the browser, and each event is a line a shell can
read with `curl -N`. WebSockets would add a bidirectional channel this feature never uses.

Each event is one `event:` line plus one JSON `data:` line, and `json.dumps` escapes control
characters, so neither a newline nor an ESC byte in model output can break the framing. The
response carries `Cache-Control: no-cache` and `X-Accel-Buffering: no`, because a cached or
proxy-buffered answer would arrive all at once. A client still renders every string as text, never
as HTML, and takes every link from `citations[].url` (`docs/api/README.md`).

## The session is a resource

A streamed response outlives the function that started it: Starlette runs a sync generator in a
thread pool and writes each chunk as it appears. `run_question` releases its session in a
`finally`. The session is read-only, so there is nothing to commit; the usage rows (`chat_calls`)
were already committed by `ChatCallLog`, which opens its own short session per row precisely so a
failed or abandoned request cannot hide calls it already spent against the free tier
(`docs/learn/09-chat-models-and-openai-compatible-apis.md`). `ChatCallLog` also has its own small
engine, so a usage write never queues behind an answer that holds a connection while it waits for
the model.

A disconnect is where the framework matters. Starlette 1.9 never closes the generator a
`StreamingResponse` was given: when the client hangs up it cancels the response task and leaves
the generator suspended at a `yield`. A later garbage-collection close would run on whatever
thread happens to trigger the pass, and closing the LangGraph stream joins the graph's worker
thread: on the event loop that freezes the server, and on the agent's own thread it deadlocks.
So the response itself closes the generator: `_ClosingStream.stream_response` does it in a
`finally`, on a worker thread, inside a shielded cancel scope, so the close is deterministic and
never races the `next()` still in flight.

Closing is also the abort signal. The response owns a `threading.Event` and sets it the moment
its disconnect listener fires, before the generator is closed, so the flag is set while the answer
is still running. A callback handler on the graph raises `AnswerAborted` (a `BaseException`, so
the fallback loop never mistakes it for provider trouble) at the model's next token, and
`call_models` checks the flag before it starts a request at all.

How quickly each phase stops depends on what it is doing, and the difference is worth knowing:

| Phase | Stops |
|---|---|
| The model is streaming words | Within a token: the callback aborts the call. |
| A tool is running | When the tool returns: a database query cannot be interrupted, but the round after it is refused before any request is sent. |
| The model is between tools and its first token | When the first token arrives, or when the request returns. |

`call_models` still records the attempt that reached the provider, so the daily cap counts what
was spent; an abort before the request starts records nothing. Ctrl-C in `paperwatch ask` uses the
same flag: the first press prints `stopping; press Ctrl-C again to quit now` and the command exits
130 once the current phase ends; because a model call is a thread Python joins at exit, a second
press flushes and leaves immediately (`os._exit(130)`). `paperwatch serve` passes uvicorn a
five-second graceful-shutdown timeout, so stopping the server cancels an open answer instead of
waiting for the model.

## Health and readiness

`/healthz` is `async def` and never touches the database, so a liveness probe cannot queue behind a
stalled one. `/readyz` runs one `SELECT 1` in a worker thread, under a two-slot
`anyio.CapacityLimiter` and a two-second `anyio.fail_after`, and answers 503 when no slot is free
or the bound passes. The slot is released in a `finally`, so a timed-out check cannot leave the
probe answering 503 forever.

The limiter bounds how many checks are started, not how many are still running: a check abandoned
at the timeout keeps its thread until the driver returns. What bounds that pile-up is the
three-second libpq connect timeout and the connection pool the check queues on. That is why
readiness queries through the small usage engine rather than the answer pool: with eight
connections held by live answers, a healthy database would otherwise answer 503, which an
orchestrator reads as "restart the process".

## What a test can see without a live model

`tests/integration/test_api.py` drives the real app with a scripted chat model, so the SSE wire
format, the event order, the error events and the OpenAPI contract are all checked offline. The
disconnect tests speak raw ASGI, because a `TestClient` reads the whole stream and never hangs up:
one hangs up under ASGI 2.3 (where Starlette listens for `http.disconnect`) and one under 2.4
(where it notices on a failed send), and both assert the session was released without a
`gc.collect()`. A third hangs up mid-answer and asserts the model stopped early and its usage row
was still recorded. Two more cover the phases that produce no token: a hang-up during a tool, and
one during a tool-call round, must not start another provider request. The committed
`docs/api/openapi.json` is compared with the app's own document, so an API change that skips the
contract fails the suite; `docs/api/README.md` holds the one-liner that rewrites it.
