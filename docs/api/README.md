# API

`paperwatch serve --live` runs the FastAPI app from `src/paperwatch/api/app.py` on
`http://127.0.0.1:8000` by default. It has no sign-in yet (Entra ID arrives with M3), so it is
bound to the loopback interface unless `--host` says otherwise.

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness: the process is up. Never touches the database. |
| `GET /readyz` | Readiness: one `SELECT 1` against Postgres succeeds, else 503. |
| `POST /ask` | One question, streamed as server-sent events. |

`POST /ask` takes `{"question": "...", "tiers": [...], "profiles": [...], "since": "YYYY-MM-DD"}`
(the last three optional) and answers with `text/event-stream`. Every event is one
`event:` line and one `data:` line holding a JSON object:

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

## The committed contract

`docs/api/openapi.json` is generated from the app and compared with the app in
`tests/integration/test_api.py`. Rewrite it only after an intended API change:

```bash
uv run python -c "import json; from datetime import UTC, datetime; from paperwatch.api.app import AgentDeps, create_app; from paperwatch.config import Settings; from paperwatch.llm.embedder import FakeEmbedder; deps = AgentDeps(lambda: None, FakeEmbedder(), [], Settings(), lambda: datetime.now(UTC)); open('docs/api/openapi.json', 'w').write(json.dumps(create_app(deps).openapi(), indent=2) + '\n')"
```

The dependencies are stubs because the OpenAPI document depends only on the routes and their
models, so rewriting it needs no database and no API key. The streamed payloads are described in
the `200` response's description, because server-sent events have no JSON body to model.
