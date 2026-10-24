# 14. Suggestions are gated, scored, diversified and summarised in the API process

Date: 2026-10-24

## Status

Accepted

## Context

The page's "This week" section needs a short list of work the captain should look at: reliable,
cutting edge, and worth 30 seconds. Two properties fight each other. A list that ranks purely by
attention promotes whatever is loudest, which for a corpus of preprints means a viral lab preprint
with no independent evidence. A list that ranks purely by credibility is a static hall of fame,
which is not what "This week" means. And a summary a model writes reads as true whether or not it
is, so a card that states a number, a quote or a name the article does not contain is worse than no
card at all: the reader cannot tell it is wrong without opening the paper.

The design (section 15) settles the shape: 5 visible picks, a Rising shelf of up to 3, computed on
request, with checked 30-second summaries. This records the decisions that design left to the
builder.

## Decision

**Gate, then score, then diversify.** Reliability is a gate, never a weight. Only admitted items
inside the window with enough stored text are candidates. The picks belong to pick-grade items
(credibility at least `pick_min_credibility`, or a lab post from a lab a reader recognises); a
fast-rising preprint without independent evidence goes on the Rising shelf, which is labelled "not
yet independently vetted" in the response itself rather than only on the page. Attention can
reorder what passed the gate; it cannot promote past it.

**The weights live in `config/suggestions.yaml`, and every card records its components.** The score
is a weighted sum of six stored values, and each `shown` event keeps the six components beside the
score. A weight change is then a reviewed file edit whose effect can be replayed from the recorded
data, instead of a code change nobody can evaluate.

**Momentum is measured from a baseline at the window start, not from the item's lifetime.**
`signal_snapshots` is append-only and written only when a recorded number changes (ADR 0011), so a
paper with no recent rows is unchanged, not unknown. The baseline is therefore the upvote count the
last run that actually read the Hugging Face listing (`counts["hf_listing_stored"] == 1`) saw at or
before the window start, and the rate divides by the real gap to `now`. A paper that took 200
upvotes in two days and then went flat scores zero; a run that skipped the listing widens the
denominator instead of crediting the days nobody looked as speed. Only a paper first listed inside
the window falls back to its lifetime average, because there the lifetime is the window.

**The list is computed on request and cached for five minutes; the ranking is never stored.** The
cache lives in the API process and is dropped when a `dismissed` or `drafted` event changes the
excluded set, so a dismissal is visible on the next request. No stickiness beyond that: a list that
remembers what it showed becomes a second corpus nobody can inspect. "Shown" is recorded at most
once an hour per item, so the event measures what a reader saw rather than how often the tab
refreshed.

**Summaries are written in the API's own background thread, one at a time.** The list returns at
once with the abstract's excerpt and "summary on its way"; the queue fills it in and the page
re-fetches itself. A resident worker or a separate service would hold a database pool and a model
client all day for a few calls an hour, and the ingestion must not lose memory to it. The queue
pauses one item for thirty minutes after a provider outage and everything until UTC midnight after
the daily cap, because retrying before then cannot work.

**Every summary is checked by code, and the check has the last word.** `what` and `result` must
carry a verbatim quote from the stored text; every number and name-like token in any field must
appear in the title or the text; a superlative may only be used if the source uses it; each field
is at most 35 words and all four at most 110. `why` and `catch` are labelled interpretation and
need no quote, and "Not stated in the abstract." is a valid catch. One retry names the failed
checks, then the card falls back to the abstract's own first two sentences: an honest excerpt beats
a fluent invention. The failed check names are stored with the summary, so the "at least 70 percent
checked" review after the first 30 real items (design section 15) is a query, not a guess.

## Consequences

- A summary costs one or two chat calls per item, in the API process, counted against the same
  daily cap as answers. A page load after a long absence can therefore spend calls; the queue is
  serial and the cap is the bound.
- The ranking reads every admitted item inside the window plus their snapshots on a cache miss.
  Bounded by 90 days (`MAX_CANDIDATE_DAYS`), it is a few thousand rows, not a scan of the corpus.
- The checks are a heuristic, not a parser: a name-like token heuristics over-eagerly flags some
  capitalised words, which costs a retry and sometimes an excerpt. That direction of error is the
  one that keeps the page honest.
- `paperwatch suggest --live --summaries` reports the checked share, so tuning the prompt or the
  checks is measured against the same target the review uses.
- A lab post can only reach a pick when its lab is on `recognisable_names`, which is a presentation
  choice in a config file: the captain can change it without a release.
