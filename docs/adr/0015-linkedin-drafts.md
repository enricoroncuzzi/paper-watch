# 15. LinkedIn drafts are one checked call, with the point of view stored and never logged

Date: 2026-10-24

## Status

Accepted

## Context

The page's picks are read, not shared. The owner wants to turn one into a LinkedIn post in his own
voice without vouching for a claim the paper never made. Two failures matter. A draft can state a
number, quote or link the article does not contain, or claim he built or tested something he did
not. And the only personal text in the product, his point of view, must not reach a provider that
trains on inputs or any log line at all.

Design section 8 fixes the flow: one call per click, a fixed reply structure, and a check shown as
numbered notes. Section 15 (Q6) and section 16.3 (captain) fix the privacy: the per-request
`provider.data_collection: "deny"`, no fallback, and nothing personal logged or committed; the
owner later dropped the denial (below). This records the choices the design left to the builder.

## Decision

**One call, one structure, no retry.** The model returns JSON (`hook`, `body`, `hashtags`,
`source_link`) and code reads it leniently (a fence is stripped, the first balanced object wins). A
reply with no usable structure is a 503 `unusable` and saves nothing: a second call would spend
the free tier on a reply the check can already judge, and a half-written draft in Drafts is worse
than an error. `DRAFT_PROMPT_VERSION` is stored with every draft.

**OpenRouter only, one model, and no fallback.** `make_chat_models(settings, purpose="draft")`
builds one OpenRouter model, like the answer chain's primary, and never Groq. The point of view
goes to the one provider the owner chose, so a silent second provider would break the promise; a
failing call is the honest outcome. The provider's own error text can echo the prompt, so it is
reduced to its type in the log and to one fixed, cause-neutral sentence in the response ("The model
did not answer, so the draft was not generated. Try again later.").

**The owner accepted that the free provider may train on the draft prompt.** This supersedes the
first version's per-request `provider.data_collection: "deny"`. On 0.11.0 the default `:free` chat
model had no endpoint that satisfied the denial, so OpenRouter answered 404 and every draft ended
in `model_unavailable`. The owner chose to run drafts on the free chat model without the opt-out:
the provider may train on and retain the whole prompt, point of view included. Nothing personal is
logged or committed, as before. `settings.draft_model` (`PW_DRAFT_MODEL`) stays as an optional
override of the draft model, unset means `settings.chat_model`, and the answer chain never reads
it.

**The check is code over the saved text, and it never fixes anything.** Every number must be in the
title, the stored text, a checked summary quote or the point of view; every quoted span of at
least three words must be in the stored text or the point of view; every URL must be the item, its
repository or project page, or one the owner wrote, and `source_link` must be the item's own link.
A first-person sentence is allowed only when it is one of the owner's own sentences, word for
word: the model may copy them unchanged, as separate sentences, but may not trim one or blend it
with other text. A verbatim quote of the item is the authors' voice, not the owner's claim. A lab
post must name the lab next to an attribution verb; a non-peer-reviewed item must attribute every
sentence that carries a number; an early item must say it is early or not yet reviewed; a run of
twelve copied words outside a quote is flagged. Each failure is a numbered note on the draft, so
the reader sees which claim to fix. The check recognises the English first-person pronouns only:
a draft written in another language, a claim that drops its pronoun ("Tested it on 300 traces"),
`Let's` and a capitalised `Us` are not seen, and a trimmed or blended owner sentence is flagged by
design. Detection is loose in the other direction too: `I/O`, a `Stage I` label and a `We-` team
name read as first person, while a pronoun split by a homoglyph, a zero-width or soft-hyphen
character, or underscores is not seen. A sentence ends at a full stop followed by a space, not
before a lower-case letter and not after a single capital initial, so `Dr.` and `No.` can still
end one.

**Generating saves, and the item leaves the picks.** There is no separate save step, so a draft the
model wrote cannot be lost to a second click; the same request records the `drafted` suggestion
event and drops the ranking cache, which is what takes the item out of the list. Regenerate makes a
new draft with `regenerated_from` set and keeps the old one, so the previous post and its check
remain inspectable.

**The point of view is stored, returned only by the detail route, and never logged.** It is the
reader's own text: the list leaves it out so a page load does not carry it, the detail shows it so
"Open" can say what the draft was built from, and the logs carry the item id, the model, the flag
counts and the timing only. A test captures structlog at debug and the stdlib loggers uvicorn uses,
across success, a provider refusal and an unusable reply, and asserts the phrase appears nowhere.

**A draft from a Rising item is early by construction.** `early` reuses the ranking's own
`is_pick_grade`, so the dialog's notice, the draft's early sentence and the Rising shelf cannot
disagree. The prompt asks for the sentence; the checker flags its absence.

## Consequences

- The free tier's cap covers drafts through the same `CallGuard`, and the same `RequestLimiter`
  bounds a loop: a draft request costs one provider call and one limiter slot.
- A stored draft is immutable; regenerating adds a row. There is no edit, which keeps the check
  result meaningful: the text it describes cannot change under it.
- The point of view is personal data at rest. It lives in `drafts.pov` with the same access as the
  corpus, and it exists nowhere else: not in an event, not in a log, not in a commit. The CLI
  takes it from a file (`--pov-file PATH`) or stdin (`--pov -` or `--pov-file -`), so it does not
  reach `ps` or the shell history either; a NUL or a lone surrogate is refused before the paid
  call, because the draft row that would carry it could not be stored afterwards.
- The check's normalisation is written in `drafts/check.py` rather than shared with the summary
  check, so the two rule sets can evolve without one silently changing the other.
