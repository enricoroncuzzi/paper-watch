# API

`paperwatch serve --live` runs the FastAPI app from `src/paperwatch/api/app.py` on
`http://127.0.0.1:8000` by default, behind the credential gate in `src/paperwatch/api/gate.py`.
Every route but `GET /healthz` needs `Authorization: Bearer <token>`: a Vercel OIDC token when
`PW_API_AUTH=oidc`, or one minted by `paperwatch api-token --live` when it is `hmac`. A request
whose `Host` is not loopback or `PW_API_PUBLIC_HOST` gets 400, any request carrying an `Origin`
gets 403, and a body over 16 KB gets 413. The served app publishes no `/docs`, `/redoc` or
`/openapi.json`; the committed contract is the client's reference. `docs/learn/15` explains the
gate and what each status code means.

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness: the process is up. Never touches the database. |
| `GET /readyz` | Readiness: one `SELECT 1` against Postgres succeeds, else 503. |
| `POST /ask` | One question, streamed as server-sent events. |

`POST /ask` takes `{"question": "...", "tiers": [...], "profiles": [...], "since": "YYYY-MM-DD"}`
(the last three optional) and answers with `text/event-stream`. Every event is one
`event:` line and one `data:` line holding a JSON object. The event payloads are Pydantic models
(`ToolData`, `TokenData`, `AnswerData`, `ErrorData`), and the committed contract's
`text/event-stream` media type is an `itemSchema` for one parsed event: the `event` name (one of
the four below) and `data`, a JSON string whose `contentSchema` is the `anyOf` of those payloads.
A client parses the `data:` line alone, which is what it will find:

- `tool` with `{"name", "args"}`, emitted when the model calls one of the read-only corpus tools;
- `token` with `{"text"}`, the model's words as it writes them;
- `answer` with `{"text", "supported", "citations", "flags"}`: the checked answer, its sources
  (`{"id", "item_id", "title", "url", "tier", "basis"}`) and any flags
  (`{"kind", "sentence", "detail"}`). It is the last event unless an `error` follows. `id` is the
  per-answer alias (`S1`, `S2`, ...) the answer cites;
- `error` with `{"code", "message"}`, where `code` is `daily_cap`, `budget`, `model_unavailable`
  or `internal`. It ends the stream, and the HTTP status is still 200: the stream had started.

## Rules for a client

- Render `text`, `title`, `sentence` and `detail` as text, never as HTML, and take every link
  from `citations[].url`. Nothing the model wrote is a link.
- Replace the streamed text with `answer.text` when the `answer` event arrives; never append it.
  The streamed words can differ from the checked answer: an unknown alias is masked to `[?]`, and
  an answer with no supporting source is replaced with the fixed no-support message. M2 has no
  per-model-call boundary on `token` events, so the `answer` event, not the accumulated tokens,
  is the signature of what was checked.
- `supported` is true when every visible claim points at a retrieved source. That is not a
  verification: sentences under six words and numbers are not checked
  (`docs/learn/11-citations-and-grounding.md`).
- On a `NO_SUPPORT` answer the `flags` list is empty and the flags are logged server-side: they
  describe model text the reader never saw.

A request without a valid credential answers 401 before anything is read; `docs/learn/15` lists
the gate's other refusals. A request the gate lets through but that fails validation before the
stream starts answers 422 in the shape the contract documents for a validation error: `{"detail": [{"type", "loc", "msg"}]}`. That covers both the pydantic
field checks and the two route-level refusals (an unknown profile name, a `since` in the future).

## The committed contract

`docs/api/openapi.json` is generated from the app and compared with the app in
`tests/integration/test_api.py`. Rewrite it only after an intended API change:

```bash
uv run python -c "import json; from datetime import UTC, datetime; from paperwatch.api.app import AgentDeps, create_app; from paperwatch.config import Settings; from paperwatch.llm.embedder import FakeEmbedder; from paperwatch.api.auth import HmacVerifier; deps = AgentDeps(lambda: None, FakeEmbedder(), [], Settings(), lambda: datetime.now(UTC), type('S', (), {'record_chat_call': lambda *a, **k: None, 'chat_calls_since': lambda *a: 0})(), frozenset(), HmacVerifier('x' * 32, lambda: 0.0), frozenset({'localhost'})); open('docs/api/openapi.json', 'w').write(json.dumps(create_app(deps).openapi(), indent=2) + '\n')"
```

The dependencies are stubs because the OpenAPI document depends only on the routes and their
models, so rewriting it needs no database and no API key.

`API_VERSION` in `api/app.py` is the contract's own version, independent of the package version.
Bump it, and rewrite `openapi.json` in the same change, when a route, a request field or an event
changes in a way an existing client would notice. A release that only bumps the package must leave
both alone, which is why the two versions are separate.
