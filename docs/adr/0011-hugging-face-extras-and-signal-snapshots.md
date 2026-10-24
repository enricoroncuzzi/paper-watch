# 11. Hugging Face extras live on the item, and signal snapshots are append-only

Date: 2026-10-23

## Status

Accepted

## Context

"Rising" needs velocity: how fast a paper is gaining attention, not how much attention it has
ever had. The store kept only a running maximum of Hugging Face upvotes (`Repository.merge_signals`
takes `max`), which answers "25 upvotes now" and cannot answer "how many it had yesterday". The
daily-papers call the ingestion already makes also returns the GitHub repository and its star
count, the project page, the comment count, the organisation and the listing time (the fixture
`tests/fixtures/hf_daily.json` carries all of them, with `numComments` on the entry and the rest on
`paper`). Keeping them costs no extra request, and history can only be built forward: every day
without snapshots is a day of velocity that cannot be recovered.

## Decision

`HfDailySource.papers()` replaces `upvotes()` and returns one `HfPaper` per arXiv base id with the
extras, merged across the pages and days of the call: counts take the maximum, the identifying
extras the first non-null value, the listing time the earliest. `HfPaper` lives in `domain.py` next
to `Signals`, so the store consumes it without importing the HTTP client. The paging, caps, health
flags and failure behaviour are unchanged, and a malformed extra becomes `None` instead of costing
the row its upvotes. Numbers the int4 columns cannot hold are dropped, the organisation is
clipped to the column width in the client and again in the store, and control or surrogate
characters are rejected at the client. A value the database still refuses costs only that step,
which is a warning, so one hostile value cannot fail the run.

- **Extras on `items`.** `hf_comments`, `github_repo`, `github_stars`, `project_page`, `hf_org`
  and `hf_listed_at` are new nullable columns. Upvotes keep their existing rule (the stored value
  never goes down, because a lower reading is a lag); the other numbers and links take the
  listing's newest non-null value, because they describe the paper now, and a null reading is a
  missing observation rather than a zero. That includes `github_stars`: the newest observed count
  replaces the stored one, and only a null leaves it alone.
- **`signal_snapshots` is append-only.** `Repository.apply_hf_listing()` appends one row with
  `observed_at` (the run's `now`), upvotes, comments, GitHub stars and citation count, but only
  when that tuple differs from the item's latest snapshot or the item has none. A long-listed
  paper therefore does not write a flat row every morning, and the history stays a list of
  changes.
- **One query per table.** The step reads the stored arXiv items with one `external_id IN (...)`
  select and their latest snapshots with one `DISTINCT ON (item_id)` read; it never queries per
  item.
- **A small failure is a warning.** The step runs after the main pass's admission, in its own
  savepoint; a failure rolls back only the snapshot writes, logs the error type, and appends
  "hugging face extras not stored this run" to the health warnings. Extras are presentation data,
  so they must never cost an item an admission attempt or make the run unhealthy.
- **Links, names, ids and listing times are filtered at the source.** Only an https GitHub
  repository URL of at most 200 characters and an https project page of at most 2,048 characters
  are kept; the page must name an https host with no whitespace, control character, surrogate or
  credentials, so a `javascript:` scheme, a look-alike host, a malformed URL or a name the
  database cannot store becomes no value at all. A listing key must be a real arXiv id (new style
  `2510.12345`, old style `hep-th/9901001`), and a listing time is normalised to UTC and kept only
  inside a plausible listing span (2020 to 2100). An instant at the edge of the calendar could not
  be read back under every session time zone, and a sentinel such as the Unix epoch would
  otherwise be stored forever as the earliest listing.
- **Every listed stored arXiv item gets a snapshot when it first appears and whenever a number
  changes, not only re-check targets.** The step covers every stored arXiv item whose id is in the
  listing window, including peer-reviewed ones that `recheck_targets` never revisits, so the
  Rising shelf can be built over the whole corpus.
- **Ignored rows.** An item classified `irrelevant` and a listing id with no stored item are left
  alone. Follow-up passes never re-read the listing, so they write no snapshots.

## Consequences

- Momentum becomes a difference between two snapshots. The ranking can use a rate once at least
  two snapshots exist; until then it falls back to the current numbers.
- The table grows by at most one row per listed arXiv item per day, and in practice less: only
  changes are recorded. `ix_signal_snapshots_item_observed` keeps one item's history and the
  cascade from `items` cheap.
- The extras are now stored before the page that reads them exists, which is the point of shipping
  this slice first: the history starts accumulating while the rest of M3 is built.
- A source that stops returning an extra leaves the stored value in place. That is deliberate for
  a repo link or a project page, and it means a repo removed from the listing is not erased. The
  snapshot's star count does stop moving, which is visible as a flat line rather than a zero.
- Snapshots are written only on a run that reads the listing. The read sits inside the phase-3
  re-check target check, so a run with no arXiv target, or one that stops before phase 3 (for
  example a budget exhausted during classification), writes none and leaves a gap in that day's
  history. Moving the read out of that gate is a separate change if gaps turn out to matter; the
  plan deliberately left it where it was.
- A run whose listing was degraded still stores what it read but does not count as observing
  (`hf_listing_stored` stays `0`), so papers listed on a failed day are not observed by that run.
- `HfPaper` is a plain frozen dataclass in `domain.py`, so the store and the source share one type
  for the listing without the store importing the HTTP client.
