# 6. Chat runs on a free OpenAI-compatible API, guarded by a daily call cap

Date: 2026-10-19

## Status

Accepted

## Context

M2 adds a question-answering agent, so the application needs a chat model it can call many times
a day, for free, from a small VPS. The cloud budget is zero (the project's standing constraint),
and the spec's Decision 6 rules out using a coding subscription as the application's model API: a
Claude Pro, Cursor Pro or OpenCode Go subscription authenticates a human at a terminal, not a
server answering HTTP requests, and routing the product through one would break the moment the
vendor noticed, or the captain stopped paying.

Two properties of free tiers decide the design. First, free model ids are withdrawn and
rate-limited without notice, so a model id is a configuration value with a documented default,
not a constant baked into a request. Second, free access is limited by requests per day, not by
tokens: OpenRouter allows 1,000 free-model requests a day with the captain's $10 credit, and a
burst that exceeds it fails for the rest of the day. Almost every provider now speaks the request
shape of OpenAI's chat completions API, so one client (the OpenAI SDK, wrapped by LangChain's
`ChatOpenAI`) reaches OpenRouter, Groq or Azure OpenAI by changing a base URL, a key and a model
id. LangChain is used here only for that model and tool interface, not as an orchestrator.

## Decision

Chat runs on OpenRouter (`https://openrouter.ai/api/v1`) with a free model,
`nvidia/nemotron-3-ultra-550b-a55b:free` as the default, chosen by measurement on 2026-10-21
(the table below), not by the catalogue alone. Groq (`https://api.groq.com/openai/v1`) is an
optional fallback, enabled only when both `PW_GROQ_API_KEY` and `PW_GROQ_CHAT_MODEL` are set,
because Groq's free model list changes too often to pin a default. `PW_OPENROUTER_API_KEY` is the
one required secret; without it the feature refuses to start rather than failing inside an HTTP
request.

Two limits wrap every call, in `src/paperwatch/llm/chat.py`. `CallGuard.before_call` counts the
`chat_calls` rows since UTC midnight and refuses at `chat_max_calls_per_day = 800`, a fifth under
OpenRouter's 1,000. That gap is headroom for questions asked at the same time, not for retries:
every attempt that reaches the provider is counted, so the count is never a floor.
`CallGuard.after_call` records the provider's real `usage_metadata` (zero, with a warning, when a
provider reports none) and then charges the per-answer token ledger, the same `Ledger` and
`BudgetExceeded` that M1's ingestion runs use, capped at `chat_max_tokens_per_answer = 60_000`.
`CallGuard.after_failure` records an attempt that reached the provider and raised (429, timeout,
5xx, withdrawn model) as a row with zero tokens and model `"failed"`, because a failure is exactly
when the next request is most likely to be refused. The usage row is written before the charge,
so a call that spent tokens is on record even when the ledger then refuses the answer.

Rows are written through `ChatCallLog`, which opens its own short session and commits each row
before returning, not through the request's session. The request's session is rolled back when an
answer fails or the client hangs up, and the requests already spent must stay counted; a
committed row is also visible to a second question running at the same time. The cap is counted
from Postgres, not from memory, so a restart cannot forget calls already made today. Each model
is built with a request timeout (`chat_timeout_s = 60`) and `max_retries = 0`, so a stalled free
model becomes a fallback instead of a hung request, with `stream_usage=True` so a streamed
answer still reports the tokens it spent, and with `max_tokens = chat_max_output_tokens = 4096`,
which bounds one response and with it a stream that never stalls; `chat_timeout_s` applies per
read, so the worst case for one answer is `(MAX_TOOL_ROUNDS + 1) x (input + 4096)` tokens.
Retries are explicit: the agent's fallback loop makes
them, one guarded attempt at a time, because an SDK retry inside a call is a provider request the
guard would never see. Keys are `SecretStr` in `Settings`, so a settings object in a log or a
traceback shows `**********`, not the key.

## Measured model choice (2026-10-21)

The model id above was chosen by measurement, not by reading the catalogue: three models were
recorded on the 20-question QA set (`paperwatch qa-eval --live --model <id>`), each answer once,
with the same offline retrieval (`FakeEmbedder`) so only the chat model differs. A recording stores
the model's own final text (not the checker's rewrite) and a `prompt_sha`, so the scores below are
the recorded answers replayed through the current checker and a prompt change makes a recording
stale instead of silently comparable. The rules were: pick by `support_recall`, then
`citation_precision`, then `abstention`, then the lower `mean_tokens`.

| Model | Status | Requests | support_recall | citation_precision | clean_rate | abstention | mean_tools | mean_tokens |
|---|---|---|---|---|---|---|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b:free` | recorded | 79 | 1.000 | 0.826 | 0.500 | 1.000 | 3.20 | 7702 |
| `nvidia/nemotron-3.5-lightning:free` | recorded | 71 | 0.813 | 0.842 | 0.250 | 1.000 | 2.55 | 7138 |
| `minimax/minimax-m3.1-flash:free` | recorded | 98 | 1.000 | 0.186 | 0.188 | 0.000 | 8.75 | 17188 |
| `google/gemma-4-31b-it:free` | unavailable | 20 | - | - | - | - | - | - |
| `nvidia/nemotron-3-super-120b-a12b:free` | unavailable | 20 | - | - | - | - | - | - |
| `nvidia/nemotron-3.5-super-vl:free` | unavailable | 50 | - | - | - | - | - | - |

The first version of this table (same date, before the raw-text re-record) stored the checked text
and so reported ultra at 1.000/0.826/0.688/1.000; the raw-text numbers above supersede it, because
only they let a replay re-run the checker on what the model actually wrote. The request counts are
the implementer's tally from each run's HTTP log, which is not committed; the unavailable models
left no recording.

`nvidia/nemotron-3-ultra-550b-a55b:free` is the default: it ties minimax on `support_recall`
(1.000) and wins on `citation_precision` (0.826 against 0.186). Minimax answers every question,
including the four unanswerable ones, and cites far more sources than the supporting set, so it
scores 0.000 on `abstention`, and it spent 17188 tokens a question. Lightning is cheaper per
answer but misses supporting evidence on three answerable questions. Lightning's
`citation_precision` (0.842) is actually higher than ultra's (0.826); the choice rests on
`support_recall` being ranked first, and lightning is three answerable questions short there.

The three named candidates from 2026-10-16 did not all survive the day. Gemma answered 429
(upstream rate limit) on all 20 questions; nemotron-3-super answered an error on all 20 (429 in
the recording, and a direct probe showed its upstream answering 404 "Function not found for
account"), so it was never a rate-limit retry. The plan's rule is
to take the next most recent free model with `tools` instead, so minimax (the newest) and
lightning were recorded; `nvidia/nemotron-3.5-super-vl:free` started fine but failed 10 of 20
questions with upstream errors and was reported unavailable rather than retried. No measurement
retried a provider failure to get a better number, and the unavailable rows carry no metrics.

The recording files are committed under `evals/data/qa_recorded/`, so the comparison can be
replayed and re-scored; `evals/qa_baseline.json` holds only the default model's four gated
metrics, and `paperwatch qa-eval` fails on any drop because the replay has no model noise.

## Consequences

- No card, no per-token price, and no vendor lock-in at the client: switching provider is a base
  URL, a key and a model id in `config.py`.
- Free models change without notice. A withdrawn or degraded model surfaces as a failed call, the
  configured fallback answers when it exists, and Task 18's recorded QA eval is what detects a
  replacement that answers worse.
- The daily cap is a hard refusal, not a warning: on a busy day the 801st answer is refused with
  `DailyCapReached` rather than spending the free tier's last requests.
- The per-answer token cap bounds one answer, not the day; the request cap is what bounds the day,
  because a free tier meters requests.
- The 800 figure is tied to the captain's current $10 OpenRouter credit. A different credit, or a
  paid tier, is a settings change plus an edit here, never a silent default.
- A test never needs a key or the network: `ScriptedChatModel` replays recorded answers and
  raises recorded errors, so the whole agent graph runs offline in CI.
- LangSmith tracing is off, and must stay off. `langsmith` arrives as a transitive dependency of
  LangChain, and setting `LANGSMITH_TRACING` (or `LANGCHAIN_TRACING_V2`) would ship every prompt,
  which contains stored paper text, to a third party.
- The OpenAI SDK and LangChain also read `OPENAI_ORG_ID` (or its alias `OPENAI_ORGANIZATION`),
  `OPENAI_PROJECT_ID`, `OPENAI_PROXY` and `OPENAI_CUSTOM_HEADERS` (an `Authorization` line there
  replaces the API key passed here), so an OpenAI setup in the same shell would reach these
  providers too. `OPENAI_BASE_URL`, `OPENAI_API_BASE` and `OPENAI_API_KEY` do not apply, because
  this code always passes `base_url` and `api_key`. Accepted: nothing here sets them; a stray
  credential or header would go to the provider.
