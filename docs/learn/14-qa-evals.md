# QA evals

A retrieval eval asks "did the right items come back?"; a QA eval asks "did the answer use them honestly?". It needs questions whose supporting evidence is known in advance, plus questions the corpus cannot answer, because a system that always answers looks perfect until you ask it something it does not know. Live model calls are slow, rate-limited and not reproducible, so the answers are recorded once per model and the scoring is replayed from the recording in CI: CI then guards the checker and the scoring exactly, and a new recording is the deliberate act of measuring a new model. The same frozen set later compares two agent frameworks on equal terms.

## The set

`evals/data/qa_questions.jsonl` holds 20 questions, each one JSON object:

```json
{"id": "q14", "question": "How do sparse attention and state space models each cut the cost of attention?", "supporting": ["ml-01", "ml-02"], "kind": "answerable"}
```

Sixteen are `answerable`, written from the 41 items of `evals/data/corpus.jsonl`, and four are
`unanswerable` (training energy, robot-control RL, GPU counts, data-centre carbon) with
`"supporting": []`. Three answerable questions need two items (the sparse-attention/state-space,
mixture-of-experts/RoPE and text/image-watermark pairs), and five are paraphrases of the title:
they share no content word with it (they do share stems with the item's body, which is the
lexical overlap the fake embedder needs). A test pins the counts, the unique ids, that every
supporting id exists in the corpus, and that the unanswerable set has no supporting item.

`QA_SET_VERSION` (currently 1) is stored in every recorded row. Changing the file means bumping
the version and re-recording: a recording made against another version of the set is not
comparable, and the tests refuse a mixed recording.

## Record once, score forever

`record(session, models, questions, corpus)` seeds the shared eval corpus with
`retrieval_eval.seed_corpus` and the offline `FakeEmbedder`, then answers every question through
the real graph (`build_agent`, `make_tools`, `answer_question`). Every model therefore meets the
same search results, and only the chat model differs. Each question gets a fresh
`SourceRegistry`, `Ledger` and `CallGuard`, because aliases, spending and the call cap belong to
one answer.

The recording keeps what the model produced and what the tools gave it, not the checker's
verdict. `answer_question` returns the final message's own text, and that is what is stored, so an
alias the checker would have masked to `[?]` is still there for a later checker to judge.

```json
{"question_id": "q01", "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "set_version": 1,
 "answer_text": "The model's own words, ... [S1]", "sources": [{"alias": "S1", "external_id": "ag-01", "tier": "vetted_preprint"}],
 "tools": ["search_corpus"], "tokens": 6123, "provider_error": false, "prompt_sha": "e85da7e3..."}
```

`prompt_sha` is a fingerprint of `SYSTEM_PROMPT` and `FINAL_ROUND_NOTE`. A prompt change changes
model behaviour, so `check_complete` refuses a recording made under another prompt, names the file
and prints the re-record command. The fingerprint does not cover the tool names, descriptions and
argument schemas, or `MAX_TOOL_ROUNDS`: those shape the model's behaviour too, but changing them
does not make a recording stale. The same function, used only by the `qa-eval` replay (the plan's single-row tests
call `score` directly and stay lenient), also refuses a recording that misses a question, carries
another `QA_SET_VERSION`, or names another model than the file it lives in. A partial recording
would otherwise score `abstention` 1.0 by dropping the unanswerable rows.

`provider_error` marks a question the provider never answered. The row stays in the file, so the
recording says what happened instead of silently shrinking the set, and the score counts it as a
failure: not a clean answer, and not an abstention (the system did not choose to say so). A live
recording that fails more than 2 of the 20 questions on provider errors is reported as
unavailable and is not written at all (`QA_MAX_PROVIDER_ERRORS` in `cli.py`).

`score(recorded, questions)` rebuilds each answer's registry from `sources` (aliases S1..Sn, with
a synthetic deterministic item id per external id) and runs the current `check_answer` on the
stored text. That is the point of the replay: a change to the citation checker is measured against
the model's own words, not remembered from an older verdict.

| Metric | Definition | Gated |
|---|---|---|
| `support_recall` | share of answerable questions whose valid citations include at least one supporting item | yes |
| `citation_precision` | valid citations on answerable questions that are supporting items, over all valid citations there (1.0 when there are none) | yes |
| `clean_rate` | share of answerable answers with no checker flag | yes |
| `abstention` | share of unanswerable questions whose checked text is `NO_SUPPORT` | yes |
| `mean_tools` | tool calls per answer | reported |
| `mean_tokens` | tokens per answer, from the per-answer ledger | reported |
| `provider_errors` | questions the provider never answered | reported |

"A valid citation" means an alias the registry issued for that answer; `S9` when only `S1` was
returned is masked by the checker and never reaches the metric.

## What the measurement chose

On 2026-10-21 three models were recorded, each answer once, with at most 98 requests apiece.
The table is from the raw-text recordings (an earlier run of the same day, which stored the
checker's rewritten text, reported ultra at 1.000/0.826/0.688/1.000):

| Model | Requests | support_recall | citation_precision | clean_rate | abstention | mean_tools | mean_tokens |
|---|---|---|---|---|---|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 79 | 1.000 | 0.826 | 0.500 | 1.000 | 3.20 | 7702 |
| `nvidia/nemotron-3.5-lightning:free` | 71 | 0.813 | 0.842 | 0.250 | 1.000 | 2.55 | 7138 |
| `minimax/minimax-m3.1-flash:free` | 98 | 1.000 | 0.186 | 0.188 | 0.000 | 8.75 | 17188 |

`google/gemma-4-31b-it:free` answered 429 on all 20 questions and
`nvidia/nemotron-3-super-120b-a12b:free` errored on all 20 (its upstream returned 404), so both
are unavailable; `nvidia/nemotron-3.5-super-vl:free` failed 10 of 20 and was also reported
unavailable rather than retried. The chosen default is the ultra model: it ties minimax on
`support_recall` and wins on `citation_precision`, while minimax answers the unanswerable
questions and cites far beyond the supporting set. The full table and the rules are in
[ADR 0006](../adr/0006-free-chat-provider.md).

The ready-made trap is visible in minimax's row: a model that always answers scores 1.000 on
`support_recall` and 0.000 on `abstention`. That is why the four unanswerable questions and
`clean_rate` exist.

## The gate

`evals/qa_baseline.json` holds the four gated metrics of the default model's recording. Scoring
is deterministic, so `gate(metrics, baseline)` uses a tolerance of 0: any drop fails. `paperwatch
qa-eval` replays the committed recording, prints every metric and exits 1 with a `REGRESSION:`
line for each gated metric below its baseline. The CI `eval` job runs it next to `paperwatch
eval`, and it needs no model, no network and no database.

The gate bites: with `score` returning an empty registry to the checker (so every answer becomes
`NO_SUPPORT`), `paperwatch qa-eval` exits 1 with `support_recall dropped from 1.000 to 0.000` and
`clean_rate dropped from 0.500 to 0.000`; restoring the file exits 0. The plan's suggested probe, a
checker that accepts unknown aliases, still changes nothing on the recording, because none of the
three models cited an alias its tools did not return. The raw text makes the class visible when it
does happen: `test_a_mixed_marker_keeps_its_unknown_source_flag_on_replay` pins that a recorded
`[S1, S9]` is still flagged, where a stored `[S1, ?]` would have lost the flag.

A recording is a snapshot, not a live measurement: it says how one model answered on one day. A
model that is withdrawn, rate-limited or silently changed is caught by recording again, not by CI
going red on its own. The recording command is `paperwatch qa-eval --live --model <id>` (it needs
`PW_OPENROUTER_API_KEY` and a database it can create), and `--write-baseline` is a deliberate,
separate act for the default model.

## What the number cannot say

The score replays the checker, so it measures retrieval support and citation discipline as the
checker defines them, not whether a sentence is true: a confidently wrong sentence with a valid
citation still counts as supported, which is lesson 11's known limit. The set is small (20
questions) and the corpus is synthetic, so the numbers compare models against each other on this
set, not models against the world. And `mean_tokens` mixes prompt tokens with completion tokens:
it is a cost signal, not an answer-length measure (the model calls include each round's growing
history).

**`abstention` counts the checker's `NO_SUPPORT`, not honesty.** A decline that cites nothing and
a hallucination that cites nothing both become `NO_SUPPORT`, so both count as abstention; only an
answer that points at a retrieved source is not an abstention. In ultra's recording q17 to q20
are honest declines ("The sources retrieved from the corpus do not contain information about the
energy consumption ..."), stored as raw text, and the checker turns them into `NO_SUPPORT` because
they cite nothing. The metric says "did not cite", not "knew it did not know". A model that
invented a confident uncited answer would score the same on this metric; a model that invented an
answer *with* a valid citation would not.

**`clean_rate` 0.5 is mostly paragraph-final citations, not half-wrong answers.** Ultra's eight
unclean answerable answers (q02, q04, q07, q11, q12, q13, q15, q16) carry `uncited_claim` flags
only; no other flag kind appears. q02 is the pattern: "According to the ToolBench benchmark paper,
ToolBench measures argument correctness and call order when a model calls a function." is the first
sentence and has no marker, while the next sentence ends in `[S1]`. The checker splits at the full
stop, so the first sentence is an uncited claim. q15 has four such sentences across two paragraphs.
`clean_rate` therefore measures where citations sit in sentences, which is the prompt's job, not
whether the claims are true. A model that puts one citation at the end of each paragraph will look
half-clean however good its answer is.

**Malformed tool calls are not directly visible.** The HTTP logs are request status lines with no
response bodies, so they cannot show a reply. The recordings store the model's final text. A reply
made only of malformed tool calls has no valid `tool_calls`, so the graph's `route` sends it to the
check node and the run ends with empty model text; a non-error row with an empty `answer_text`
would therefore be that reply's signature. None of the three recordings has one: every row has
text. So no final reply in this measurement consisted only of malformed tool calls. The evidence
cannot rule out malformed calls alongside a valid call or alongside answer text, because those run
or answer normally and the recording keeps only the final text, not the trajectory; the HTTP logs
cannot settle those cases either.


The recording does not store the round count, so the 4-round cap is visible only indirectly, in
the tool counts: a question with four or more tool calls in a model that makes one call per round
went through the capped answer round, which is the evidence that the chosen model accepts
`tool_choice: "none"` with the tool definitions attached. An all-malformed reply, if it happened,
would be the empty final text described above; none appears in these recordings. If one shows up
on a real model, the next step is to count it as a round and answer it with an error `ToolMessage`
(Task 16's note).

The gate only fails on a drop. A checker that becomes more lenient can raise `clean_rate` (fewer
flags) and pass; the replay sees the change, but the gate does not treat an improvement as a
signal. The tests are what pin the checker's hard guarantees (`docs/learn/11`), and a deliberate
re-record is what notices a model change.

The recording guard runs against the throwaway database, so the day's `chat_calls` count does not
include the recording's requests: the three re-recordings spent 248 requests against the
OpenRouter key's 1,000 a day while the application's own table saw none of them. That is fine for
a manual measurement made by an operator with their own key, and it is why the request counts in
the tables are the implementer's tally from each run's HTTP log rather than a database query.

The implementation is `evals/qa_eval.py` with the set and recordings under `evals/data/`, the
committed baseline in `evals/qa_baseline.json`, the command in `src/paperwatch/cli.py`, and the
tests in `tests/unit/test_qa_eval.py`, `tests/integration/test_qa_record.py` and the `qa-eval`
section of `tests/unit/test_cli.py`.
