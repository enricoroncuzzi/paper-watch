# Chat models and OpenAI-compatible APIs

Most model providers now accept the request shape of OpenAI's chat completions API, so one client (the OpenAI SDK, wrapped here by LangChain's `ChatOpenAI`) talks to OpenRouter, Groq or Azure OpenAI by changing a base URL, a key and a model id. That makes the provider a configuration value, which matters on free tiers: free models are withdrawn or rate-limited without notice, and a fallback provider keeps the feature alive. A chat call is paid in tokens (input plus output) and limited in requests per day, so every call goes through a guard that checks the daily request cap before the call and records the real token usage after it, the same ledger idea as M1's embedding budget. LangChain's role is small and explicit: a common chat-model interface with tool binding and streaming, not an orchestrator.

## What the code does

`src/paperwatch/llm/chat.py` holds three things.

`make_chat_models(settings)` builds the provider chain, primary first. OpenRouter is always
required; Groq is appended only when both its key and its model id are set. Each model is a
`ChatOpenAI` with a base URL, a key, a model id, a request timeout and a retry count, plus
`stream_usage=True`. That last flag matters more than it looks: a streamed response from an
OpenAI-compatible provider carries token usage only when the request asked for it, and without it
the ledger would charge nothing for the answers the agent actually streams.

`CallGuard` wraps one answer's calls. `before_call` counts today's rows in `chat_calls` from UTC
midnight and raises `DailyCapReached` at the cap, before the request goes out, because a free tier
meters requests. `after_call` reads `usage_metadata` from the response, falls back to zero tokens
with a logged warning when a provider omits it, takes the model id the provider reports (or
`unknown`), writes the `chat_calls` row and only then charges the `Ledger`. Writing first is the
point: a call that spent tokens must be on record even when the ledger's `BudgetExceeded` refuses
the answer it paid for.

`ScriptedChatModel` is a `BaseChatModel` that replays a list of `AIMessage`s and exceptions. It
records every message list it was asked (`seen`) and every set of tool names it was bound with
(`bound_tools`), so an agent test can assert the exact trajectory offline. Two details of its
implementation are lessons about LangChain rather than about our code. Pydantic validates a
`list[AIMessage | Exception]` in declaration order and `AIMessage`'s own validator raises
`AttributeError` on a non-dict instead of declining the value, so the union must be written
`Exception | AIMessage`. And LangChain appends its own empty final chunk marked
`chunk_position="last"` when a model's stream does not produce one, so a stream that must end with
its usage chunk has to mark that chunk as last itself.

## Why the cap lives in Postgres

An in-process counter is enough for M1's per-run token budget, because a run is one process. A
daily request cap is not: the API server and the CLI are different processes, and either can
restart. Counting rows in `chat_calls` makes the cap a property of the data, so two processes
share it and a restart does not reset it. The `ix_chat_calls_at` index keeps the count a range
scan rather than a growing table scan.

## The test double

`tests/unit/test_chat.py` covers the settings defaults, the missing-key refusal, the provider
order, the timeout and retry wiring, the guard's three charging paths (real usage, budget
exceeded, missing usage) and the cap boundary. `tests/integration/test_chat_calls.py` covers the
table itself: the count's boundary, the cap blocking before a call, and the cap resetting at UTC
midnight. No test calls a model or the network.

The decision and its consequences are recorded in `docs/adr/0006-free-chat-provider.md`.
