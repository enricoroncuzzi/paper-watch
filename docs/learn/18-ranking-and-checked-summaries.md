# Ranking and checked summaries

A suggestion list here is a gate, a score and a shuffle for variety. The gate is reliability: only
admitted items, and the top slots only for evidence a reader can check, so attention can reorder
trustworthy work but never promote a hunch; a fast-rising item without that evidence goes on a
labelled shelf. The score is a weighted sum of plain features with the weights in a file, so a
change is a reviewed edit and its effect can be replayed from the recorded components. Variety
comes from maximal marginal relevance: each next pick is the best item that is also least like the
ones already picked.

## The gate, then the score

`src/paperwatch/suggestions/ranking.py` is one pure function: no clock, no database, no randomness.
`now`, the config (`config/suggestions.yaml`) and the candidates come in as arguments, and a
`Ranking` of five `picks` and up to three `rising` items comes out. Because nothing is read inside
it, a card can be re-scored later from the features the API stored with it, which is how a weight
change is judged: replay yesterday's candidates under the new file and read the order.

Three gates run before any score. A dismissed or drafted item never comes back (`excluded`). An
item older than the window (`window_days`, 21 days for a paper and 14 for a lab post) is out, as is
anything with under `min_text_chars` of stored text: a title-only post cannot carry a checked
summary, and a card whose summary could only be invented is worse than no card.

The five components, each in 0..1:

- **Momentum.** Upvotes gained per day over `momentum_days`, ranked as a percentile inside the
  eligible paper pool, so 30 upvotes a day means something different in a quiet week than in a busy
  one. The gain is measured from a **baseline**: the upvote count the last run that actually read
  the Hugging Face listing saw at or before the window start. `signal_snapshots` gets a row only
  when a recorded number changes, so a missing row inside the window means "unchanged", not
  "unknown". A paper that took 200 upvotes in its first two days and then went flat scores 0, not
  the lifetime average of about 15 a day; and when a run skipped the listing, the denominator is the
  real gap between observing runs, so the days nobody looked are not credited as speed. Only a
  paper first listed inside the window falls back to `upvotes / (days since listing + 1)`, because
  there the whole life is the window. Lab posts get a flat 0.5: a lab announcing its own work is
  neither a hype signal nor a dead one, and the shelf they sit on is labelled.
- **Recency.** `0.5 ** (age / half_life)`: a paper halves every five days, a lab post every four,
  because an announcement is news for a shorter time than a result.
- **Shareability.** Code or a demo a reader can try (0.4), a lab they recognise (0.4), and stars on
  a saturating curve (0.2) so ten thousand is the top of the scale.
- **Topic fit.** How far the item cleared its profile's calibrated relevance threshold, scaled by
  the profile's editorial weight, so a strong secondary-topic match can still outrank a weak primary
  one.
- **Credibility**, from a small map: peer reviewed 1.0, vetted preprint by upvotes or citations 0.8,
  a lab's own post 0.7, a preprint vetted only by a lab author 0.6.

A one-off event is added after the weighted sum, not inside it, because it is news rather than a
rate: a promotion to peer review inside 14 days adds 0.10 and a repository appearing inside 7 days
adds 0.05.

## Lanes, duplicates and variety

The lanes are what keeps attention from promoting a hunch. An item is pick-grade when its
credibility clears `pick_min_credibility` (0.8) or when it is a lab post from a lab on the
`recognisable_names` list, which is a presentation choice the captain edits freely. Everything else
is Rising, which carries the printed note "Rising, not yet independently vetted." A viral preprint
from a lab team therefore lands on the shelf, never on the lead slot.

Near-duplicates are merged next, inside each lane: two items whose stored embeddings have a cosine
of at least 0.90 become one card whose primary is the paper (a lab's write-up about its own paper
is folded into it as "also reported by"). Then the picks are diversified with maximal marginal
relevance: the next pick maximises `lambda * score - (1 - lambda) * similarity to the picks already
chosen`, with `lambda` at 0.7, so a list of five variations on one idea becomes one pick and four
others. A per-profile quota of three stops one topic filling the page, and if no lab post survived
all of that, the highest-scoring pick-grade one replaces the lowest pick. Ties break by
`(source, external_id)`, never by database id, so a replay is reproducible.

## A summary a model writes reads as true whether or not it is

So the model must hand back, for the claims that matter, the exact sentence it took them from, and
code checks that sentence is in the stored text, that every number and name it used is there too,
and that it did not add a superlative. Opinions are allowed only in fields the page labels as
interpretation. When the checks fail twice, the card shows the abstract's own first sentences
instead: an honest excerpt beats a fluent invention.

`src/paperwatch/suggestions/prompts.py` holds the prompt and its version (`s1`), and the source goes
into the same escaped `<source>` block the agent's tools use (`agent/prompts.py` `source_block`), so
an abstract containing `</source>` and "ignore previous instructions" is data here too. The text is
clipped to `SUMMARY_TEXT_MAX_CHARS`, the same bound as `get_item`, because a feed stores a lab post
whole.

`check_summary` in `src/paperwatch/suggestions/summary.py` returns the names of the checks that
failed, and an empty list means the summary is checked. Every comparison normalises both sides
first: HTML entities unescaped (the model reads the source inside an escaped block, so it copies
`&amp;` for `&`), NFKC, curly quotes and dashes folded to ASCII, whitespace collapsed, case folded.
Then:

- **Shape.** All four fields, each with a non-empty `text`; `what` and `result` also carry a `quote`.
- **Quotes.** A quote must be a substring of the stored text, word for word, and at least 20
  characters long: a `what` or `result` quote of one word backs nothing. This is the check that
  makes the others meaningful: a model that cannot copy a sentence has not read the source.
- **Numbers.** Every number in every field (`41%`, `1,000`, `3.1`) must appear in the title or the
  text, compared without spaces or thousands separators. A number is the easiest thing for a model
  to drift on, and the most costly for a reader to trust wrongly.
- **Names.** Every name-like token must appear too: anything with a digit (`GPT-5`), a capital
  inside the word (`OpenAI`), a hyphen between letters, or a capital where a sentence does not
  require one. Each token must match a whole word in the source, not a substring of one (an
  invented `Bert` does not pass because the text says `Albert`). That is a heuristic, not a parser,
  and it is deliberately over-eager: a false
  positive costs one retry, a missed invented benchmark costs the reader's trust.
- **Superlatives.** `state-of-the-art`, `breakthrough`, `first` and four more may only be used when
  the source uses them; a claim of novelty is the sentence a reader would quote back at us.
- **Length.** One sentence of at most 35 words per field, at most 110 words in total: the card is a
  30-second read, and a summary that needs more is an excerpt with extra steps.

`why` and `catch` are the interpretation fields: they need no quote and are labelled as
interpretation wherever they are shown, and the catch may be exactly "Not stated in the abstract."
A failed check earns exactly one retry whose message names the failed checks, so the model learns
what code rejected instead of being told to "be careful". A second failure, or unparseable JSON
twice, stores `status="excerpt"` with the abstract's first two sentences (at most 60 words) and the
failed check names, so a reviewer can see the rate and the reason rather than a silent fallback. A
`DailyCapReached`, `BudgetExceeded` or `ModelUnavailable` is not a check failure: it propagates, and
the queue that asked for the summary decides what happens next (`api/summary_queue.py`). A budget
overrun will recur on every attempt, so the queue stores it as an excerpt with the failed check
`budget` and does not ask again until the text changes.
