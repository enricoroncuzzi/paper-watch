# API

`paperwatch serve --live` runs the FastAPI app from `src/paperwatch/api/app.py` on
`http://127.0.0.1:8000` by default, behind the credential gate in `src/paperwatch/api/gate.py`.
Every route but `GET /healthz` needs `Authorization: Bearer <token>`: a Vercel OIDC token when
`PW_API_AUTH=oidc`, or one minted by `paperwatch api-token --live` when it is `hmac`. A request
whose `Host` is not loopback or `PW_API_PUBLIC_HOST` gets 400, any request carrying an `Origin`
gets 403, and a body over 16 KB gets 413. The served app also serves `/docs`, `/redoc` and
`/openapi.json`, all behind the same credential, so anyone without a token gets 401. Swagger UI
cannot attach the bearer token from a browser, so the committed contract is the client's
reference. `docs/learn/15` explains the gate and what each status code means.

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness: the process is up. Never touches the database. The one route without a token. |
| `GET /readyz` | Readiness: one `SELECT 1` against Postgres succeeds, else 503. |
| `GET /status` | The search mode: `{"mode", "reason", "since"}`, where mode is `hybrid` or `text_only` and reason is `ok`, `ingestion`, `memory` or `database`. |
| `GET /today` | The today strip: `{"run": RunSummary or null, "search": {"mode", "reason", "since"}}`, the newest run by `started_at` and the search mode in the same call. |
| `GET /runs?limit=10` | `{"runs": [RunSummary]}`, newest `started_at` first; `limit` is 1 to 50. |
| `GET /runs/{run_id}` | One `RunDetail` (a summary plus `window_start`, `window_end` and `dead_letters`), or 404. |
| `GET /suggestions` | The picks and the Rising shelf: `{generated_at, list_version, picks, rising, rising_note}`. |
| `POST /suggestions/{item_id}/events` | `{"event": "opened" \| "dismissed"}`, 204, or 404 for an item that is not admitted. |
| `POST /ask` | One question, streamed as server-sent events. |

## Runs and the today strip

`RunSummary` is `{id, started_at, finished_at, status, healthy, problems, warnings, counts, tokens,
followup}`. `counts` always carries the same nine keys (`fetched`, `fetched_arxiv`, `stored`,
`new`, `irrelevant`, `failed`, `admitted_late`, `followup_passes`, `s2_refused`), a key the run
never counted being `0`. `healthy` is `null` when a run closed without a verdict, which is what
the next run's `fail_open_runs` leaves behind for a crashed one: the page shows that as unknown,
not as unhealthy. `followup` is `{pass, of, next_at, pending_s2}` while the run waits between
follow-up passes and `null` otherwise (including once the loop has ended). `RunDetail` adds the
window and `dead_letters: [{source, ref, error, attempts, created_at, resolved_at}]`, with `ref`
and `error` clipped to 500 characters. All timestamps are UTC ISO 8601.

These three routes open one short read-only session per request and are safe to poll: they never
touch a model and never count against the answer limits.

## Suggestions

`GET /suggestions` returns up to five `picks` (the first is the lead) and up to three `rising`
cards, computed on request and cached in the API process for five minutes. `rising_note` is the
printed caveat, "Rising, not yet independently vetted.", so the label travels with the data it is
about. A card is `{item_id, title, url, source, tier, basis, basis_label, venue, citation_count,
hf_upvotes, github_repo, github_stars, project_page, org, published_at, profile, relevance,
momentum_label, also, summary}`:

- `basis_label` is the sentence a reader checks: "peer-reviewed via venue: ICLR", "vetted preprint
  via 36 Hugging Face upvotes", "vetted preprint via 12 citations", "preprint from a recognised
  lab's authors, not yet independently vetted", or "lab post: the lab's own claim";
- `momentum_label` is the gain since the ranking's baseline ("+31 upvotes in 3 days"), or null;
- `also` holds the near-duplicates merged into this card (`{title, url, source}`), with the paper
  first;
- `summary` is `{status, what, result, why, catch, excerpt}` where status is `pending`, `checked` or
  `excerpt`. Each of the four fields is `{text, quote, interpretation}` or null. `pending` and
  `excerpt` show `excerpt`, the abstract's own first sentences; `checked` fields carry the
  verbatim `quote` the code verified, and `interpretation` is true for `why` and `catch`, which the
  page labels as interpretation rather than as the authors' words.

A dismissal or a draft removes the item from every later list. The API records a `shown` event for
each card at most once an hour, with the rank, the score and the six score components it ranked by.

## Limits

`POST /ask` spends a free-tier model call, so it is limited per client: 6 requests a minute and 100
a UTC day, and the server runs 2 answers at once across all clients. A refused request answers
`429` with `{"detail": "rate_limited" | "busy", "retry_after_s": N}` and the same value in a
`Retry-After` header. `rate_limited` means the minute or the day is spent, and `retry_after_s` counts to when it
frees (the next UTC midnight for the day). `busy` means both answer slots are open; a slot frees
when its stream ends, however it ends.

An answer that runs for 120 seconds ends with an `error` event whose code is `deadline`; the server
also cancels the model call, so the answer stops rather than finishing invisibly.

While the daily ingestion runs, or while the machine is short of memory, `GET /status` reports
`text_only`: each answer starts with a `notice`, its searches run on the full-text branch alone,
and its `answer.search_mode` is `text_only`. The page shows the notice and treats citations the
same way; nothing else about the stream changes.

## The event stream

`POST /ask` takes `{"question": "...", "tiers": [...], "profiles": [...], "since": "YYYY-MM-DD",
"item_id": "..."}` (all optional except the question) and answers with `text/event-stream`. Every event is one `event:` line
and one `data:` line holding a JSON object. The event payloads are Pydantic models (`ToolData`,
`StepData`, `SourceData`, `TokenData`, `NoticeData`, `AnswerData`, `ErrorData`), and the committed
contract's `text/event-stream` media type is an `itemSchema` for one parsed event: the `event` name
(one of the seven below) and `data`, a JSON string whose `contentSchema` is the `anyOf` of those
payloads. A client parses the `data:` line alone, which is what it will find:

- `tool` with `{"name", "args"}`, emitted when the model calls one of the read-only corpus tools;
- `step` with `{"round", "ms", "new_sources"}`, emitted when a tool round finishes: its number, how
  long it took and how many sources it added;
- `source` with `{"id", "item_id", "title", "url", "tier", "basis"}`, one per alias the first time
  a tool returns it, in alias order. It carries the same facts as a citation, so a card can render
  before the answer exists;
- `token` with `{"text"}`, the model's words as it writes them;
- `notice` with `{"code", "message"}`, sent once before the answer when the server runs in a
  reduced mode (today `text_only`, while the ingestion needs the memory);
- `answer` with `{"text", "search_mode", "supported", "citations", "flags"}`: the checked answer,
  the mode that produced it, its sources (`{"id", "item_id", "title", "url", "tier", "basis"}`) and
  any flags (`{"kind", "sentence", "detail"}`). It is the last event unless an `error` follows.
  `id` is the per-answer alias (`S1`, `S2`, ...) the answer cites;
- `error` with `{"code", "message", "retry_at"}`, where `code` is `daily_cap`, `budget`,
  `model_unavailable`, `internal` or `deadline`, and `retry_at` is the reset time for `daily_cap`
  (else null). It ends the stream, and the HTTP status is still 200: the stream had started.

A stream that has nothing to say for 15 seconds receives a server-sent comment line, `: ping`,
followed by a blank line. It has no `event:` or `data:` line. A client resets its watchdog on any
received bytes, so the comment is enough to tell thinking from a dead connection.

## Rules for a client

- Render `text`, `title`, `sentence` and `detail` as text, never as HTML, and take every link
  from a stored `url` (a `source` or `citations[]` entry). Nothing the model wrote is a link.
- Never render model markdown: an answer is paragraphs and citation superscripts, and a `*` or a
  `<b>` in it is a character, not formatting.
- Replace the streamed text with `answer.text` when the `answer` event arrives; never append it.
  The streamed words can differ from the checked answer: an unknown alias is masked to `[?]`, and
  an answer with no supporting source is replaced with the fixed no-support message. There is no
  per-model-call boundary on `token` events, so the `answer` event, not the accumulated tokens,
  is the signature of what was checked.
- `supported` is true when every visible claim points at a retrieved source. That is not a
  verification: sentences under six words and numbers are not checked
  (`docs/learn/11-citations-and-grounding.md`).
- On a `NO_SUPPORT` answer the `flags` list is empty and the flags are logged server-side: they
  describe model text the reader never saw.
- A dropped stream is not an error in the answer: the server cancels it. A client shows a
  connection lost state after 45 seconds without bytes, and asks again.

A request without a valid credential answers 401 before anything is read; `docs/learn/15` lists
the gate's other refusals. A request the gate lets through but that fails validation before the
stream starts answers 422 in the shape the contract documents for a validation error:
`{"detail": [{"type", "loc", "msg"}]}`. That covers both the pydantic field checks and the two
route-level refusals (an unknown profile name, a `since` in the future, an `item_id` that is not an
admitted item). An admitted `item_id` is registered as `S1` before the graph starts, so "Ask about
it" from a card produces an answer that can cite that item.

## The committed contract

`docs/api/openapi.json` is generated from the app and compared with the app in
`tests/integration/test_api.py`. Rewrite it only after an intended API change:

```bash
uv run python -m paperwatch.api.contract
```

The dependencies in `api/contract.py` are stubs because the OpenAPI document depends only on the
routes and their models, so rewriting it needs no database and no API key.

`API_VERSION` in `api/app.py` is the contract's own version, independent of the package version.
Bump it, and rewrite `openapi.json` in the same change, when a route, a request field or an event
changes in a way an existing client would notice. A release that only bumps the package must leave
both alone, which is why the two versions are separate.
