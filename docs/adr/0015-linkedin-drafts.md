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
`provider.data_collection: "deny"`, no fallback, and nothing personal logged or committed. This
records the choices the design left to the builder.

## Decision

**One call, one structure, no retry.** The model returns JSON (`hook`, `body`, `hashtags`,
`source_link`) and code reads it leniently (a fence is stripped, the first balanced object wins). A
reply with no usable structure is a 503 `unusable` and saves nothing: a second call would spend
the free tier on a reply the check can already judge, and a half-written draft in Drafts is worse
than an error. `DRAFT_PROMPT_VERSION` is stored with every draft.

**OpenRouter only, with data collection denied, and no fallback.** `make_chat_models(settings,
purpose="draft")` builds one model with `extra_body={"provider": {"data_collection": "deny"}}`.
The point of view may only go to a provider that refuses training, so a silent second provider
would break the promise; a failing call is the honest outcome. The provider's own error text can
echo the prompt, so it is reduced to its type in the log and to one fixed sentence in the
response.

**The check is code over the saved text, and it never fixes anything.** Every number must be in the
title, the stored text, a checked summary quote or the point of view; every quoted span of at
least three words must be in the stored text or the point of view; every URL must be the item, its
repository or project page, or one the owner wrote, and `source_link` must be the item's own link.
A first-person sentence with an experience verb (built, tested, ran, shipped, used) is allowed only
when the point of view says the same thing. A lab post must name the lab next to an attribution
verb; a non-peer-reviewed item must attribute every sentence that carries a number; an early item
must say it is early or not yet reviewed; a run of twelve copied words outside a quote is flagged.
Each failure is a numbered note on the draft, so the reader sees which claim to fix. The check
is English-only and its verb list is closed: a point of view written in another language, or a
claim using a verb outside the list (created, prototyped, rebuilt, validated), is not matched, so
the claim is flagged rather than quietly passing.

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
  corpus, and it exists nowhere else: not in an event, not in a log, not in a commit.
- The check's normalisation is written in `drafts/check.py` rather than shared with the summary
  check, so the two rule sets can evolve without one silently changing the other.
