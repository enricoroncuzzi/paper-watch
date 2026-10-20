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

Four events reach a client:

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
body can echo the question the provider refused.

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
thread pool and writes each chunk as it appears. `run_question` therefore releases its session in
a `finally`, so a normal end, an error and a generator close (what a framework does when the
client disconnects) all take the same path. The session is read-only, so there is nothing to
commit; the usage rows (`chat_calls`) were already committed by `ChatCallLog`, which opens its own
short session per row precisely so a failed or abandoned request cannot hide calls it already
spent against the free tier (`docs/learn/09-chat-models-and-openai-compatible-apis.md`).

## What a test can see without a live model

`tests/integration/test_api.py` drives the real app through `TestClient` with a scripted chat
model, so the SSE wire format, the event order, the error events and the OpenAPI contract are all
checked offline. The committed `docs/api/openapi.json` is compared with the app's own document, so
an API change that skips the contract fails the suite; `docs/api/README.md` holds the one-liner
that rewrites it.
