# 8. Semantic Scholar refusals are retried by follow-up passes inside the same run

Date: 2026-10-23

## Status

Accepted

## Context

The daily run on the shared VPS (`deploy/vps/`, `docs/deploy/vps.md`) gets its venue and citation
evidence from Semantic Scholar's keyless batch endpoint. That endpoint refuses a whole chunk with
429 or 503 under congestion, in bursts of unknown length: on 2026-10-23 it refused every attempt at
06:03 and again at 09:52, 145 to 152 ids each time, and no API key can be obtained. The papers S2
refused are the ones whose evidence would have admitted them, and they stay due, so today the cost
of a burst is one day of lag per paper plus a run that reads `partial`, unhealthy, with the unit
failed and the morning check red (Task 23's scout report, kept privately).

The corpus keeps no data either way, and `recheck_targets` already re-asks a refused paper the next
day. What was missing was a way to finish the work while the burst clears, on a server the captain
described as having spare capacity: "i think we increase a lot the number of retries ... reties
every 30 min till 22". The alternatives were more patience inside the S2 client (every admission in
the run waits behind the sleeps, and the spacing is invisible), a separate timer or `Restart=` chain
(new units, and the run's exit code stops describing the day), and doing nothing structural (the
paper waits a day; the verdict needs a day-level definition).

## Decision

A scheduled run keeps retrying only the papers S2 refused, in follow-up passes inside the same
process, the same advisory lock and the same run row, behind `--followup`, which only the unit
passes. A manual `paperwatch ingest --live` never waits.

- **Trigger.** After the phase-3 admission loop, before the final verdict, only when `--followup`
  was given and `s2.refused_ids` is non-empty. The flag is off by default precisely so a debugging
  run cannot sit in a 16-hour wait.
- **What a pass does.** Asks S2 for the refused ids and nothing else, with the existing client,
  circuit breaker and pacing; merges the answers; runs the existing admission step for just those
  targets, reloaded fresh from the store; stamps `s2_checked_at` with the run's `now` (never the
  pass time), so the daily cadence is unchanged; and lets the source's own outage letter follow its
  state. A pass never re-fetches arXiv, the lab feeds or Hugging Face, never classifies, never
  retries a dead letter, and never retries an item whose admission failed in the main pass.
- **Bounds.** Waits before the passes are 5, 10, 20 and 40 minutes, then 30 minutes each, one scale
  up from the client's own 5/10/20/40 s backoff; at most 40 passes; and no pass starts at or after
  22:00 Europe/Rome, so a late catch-up never retries into the night on the shared machine. The
  pass count, the interval and the deadline hour are `Settings` fields with constant defaults, so a
  drop-in can shorten the window without a release.
- **Memory.** The run commits before every wait and releases the classifier and embedder, so the
  idle process holds only the interpreter rather than the model's ~520 MB; a pass rebuilds the pair
  lazily, and only when it must embed a first admission.
- **Verdict.** The row is finished provisionally after the main pass (`health.followup` with the
  pass number, the cap, `next_at` and the pending count) and overwritten after every pass, so it is
  never `running` during the window and the 25 h freshness test keeps working. The final record is
  `{"passes": n, "stopped": "resolved" | "attempts" | "time_of_day" | "budget_exhausted"}`, counts
  gain `followup_passes`, and the process exit code is the final verdict's: a recovered day exits
  0 and leaves no failed unit. The run's degraded flag became a set of causes so a recovered S2
  contributes nothing to the verdict.
- **Severity.** S2 still refusing after the last pass is the first day of an outage, like one
  missed lab-feed day: a health warning, with its `s2/signals` letter left open. When the open
  letter was opened by an earlier run (keyed on the letter's `run_id`, never on its existence), it
  is the second day in a row and a failure. The same rule now decides a lab feed's first miss.
- **The morning check.** `check.sh` prints `WAIT` while `next_at` is in the future (counting no
  failure), `FAIL follow-up did not complete (killed?)` once it has passed, and
  `ok follow-ups: n used, <stopped>` after a finished loop. The unit's `TimeoutStartSec` rises to
  17 h for the 06:00 to 22:00 window; the stuck-`running` limit stays at 3 h, because a waiting row
  is finished, not `running`.

## Consequences

- A burst of minutes to hours no longer costs a day of evidence, and the morning reads truthfully:
  recovered days are healthy and exit 0, while a whole-day refusal is one warning the first day.
- A process may stay alive for up to 16 hours, from 06:00 to 22:00 Rome time. A fully refused day
  costs about 33 S2 requests, one every 30 minutes, and no extra fetches, classification or
  embeddings, with no model resident between passes.
- A kill inside the window (`kill -9`, an OOM kill, a stop) loses the rest of that window and
  leaves a `next_at` in the past, which the check reports. The next morning's run is the backstop,
  because every refused paper is still due.
- Feeds and the arXiv remainder are deliberately out of scope: they keep today's next-day retry
  until the refusal data shows the loop is where the value is.
- If S2's refusals turn out to last past 22:00, the window (a Settings value) or the design (a
  timer, D2 in the scout report) is what changes; both are small, because a pass is a function of
  database state.
