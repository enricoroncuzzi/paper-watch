# Citations and grounding

A model asked to cite sources will sometimes cite the wrong one, cite one it never read, or write a sentence with no source at all, and it does so fluently. Grounding is therefore checked by code after generation, not requested in the prompt and trusted. This checker uses facts the program already holds: which sources the tools returned during this answer, and each source's tier from the database. It cannot judge whether a sentence is true; it judges whether each claim points at real evidence the reader can open, and it refuses to let an answer with no evidence pass as an answer. Flags are shown, not hidden, because a visible warning teaches the reader where to double-check.

## What the checker sees

`src/paperwatch/agent/citations.py` holds `check_answer(text, registry)`. The registry is the one Task 14 created for this answer, so the only aliases that can resolve are the ones a tool registered during this conversation. `check_answer` does no I/O, calls no model and reads no database: it is pure text plus that in-memory registry, which makes it cheap to run on every answer and easy to test.

The answer is split into sentences on newlines and after `.`, `!` or `?` when a space follows, bullets (`-`, `*`, `1.`) are stripped, and a fragment that is only a marker is attached to the sentence before it, because models write both `"...bugs [S1]."` and `"...bugs. [S1]"`. A sentence is a claim when, with its markers removed, it has at least `CLAIM_MIN_WORDS` (6) words and does not end in `?`: short connectives ("In short:") and questions carry no factual claim worth a citation.

## The three flags

- `unknown_source`: a marker names an alias the registry never issued. The alias is masked in the returned text, so it can never reach the reader as a real source: `[S9]` becomes `[?]`, and a mixed marker keeps only what is known, `[S1, S9]` becoming `[S1, ?]`. Masking happens before any rendering, so the invented alias is not merely flagged somewhere else.
- `uncited_claim`: a claim sentence with no marker at all. A sentence whose only marker is unknown is reported as `unknown_source` alone, because the model did point at something and the useful repair is "that source does not exist"; flagging it twice would say the same thing twice.
- `tier_mislabel`: a sentence containing "peer-reviewed", "accepted at" or "published at" whose cited sources are all below `peer_reviewed`. A sentence may cite several sources and the label holds if any one of them supports it, so `[S1, S2]` with S2 peer-reviewed passes, and a source already cited earlier in the answer still counts, so repeating `[S1]` does not hide a preprint behind its first citation. Without a citation the sentence is already an `uncited_claim`, and there is no data to contradict.

Flags name the sentence they came from, with markers and a trailing full stop removed, so the reader can find it in the answer instead of hunting.

Citations come back in first-seen order and once each, so the source list reads in the order the answer used them. `supported` is true exactly when there is at least one citation and no flag; when nothing valid is left, the returned text is the fixed `NO_SUPPORT` message rather than the model's uncited prose, because an answer with no evidence is not an answer.

A `Citation` carries the source's title and URL for rendering, unescaped, exactly as `SourceRef` does. Whoever renders them into a prompt or a stream must escape them: a stored title is written by strangers and is data, the same reason Task 14 escapes every `<source>` block.

## What the tests pin

`tests/unit/test_citations.py` covers distinct citations in first-seen order, a grouped `[S1, S2]` marker, a marker placed after the full stop, an unknown alias flagged and masked with the remaining valid citation kept, an uncited claim, a short connective and a question left unflagged, bullets and numbered items read as sentences, a preprint called peer-reviewed flagged while the same sentence citing a peer-reviewed source passes, a repeated citation of a preprint still flagged, and the no-evidence answer for both an uncited and a wholly unknown-cited text. The plan's mutation check accepts every alias the model invents and makes the unknown-alias and no-evidence tests fail, which is what keeps the registry lookup from turning into an honour system.
