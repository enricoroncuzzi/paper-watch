# Citations and grounding

A model asked to cite sources will sometimes cite the wrong one, cite one it never read, or write a sentence with no source at all, and it does so fluently. Grounding is therefore checked by code after generation, not requested in the prompt and trusted. This checker uses facts the program already holds: which sources the tools returned during this answer, and each source's tier from the database. It cannot judge whether a sentence is true; it judges whether each claim points at real evidence the reader can open, and it refuses to let an answer with no evidence pass as an answer. Flags are shown, not hidden, because a visible warning teaches the reader where to double-check.

## A heuristic safety net

The checker is a heuristic safety net, not a parser of English. Three guarantees are hard:

1. An alias no tool returned never becomes a `Citation` and, outside code spans, closed fences and URLs, is always visible as `?` in the text.
2. An answer with no cited text is the fixed `NO_SUPPORT` message.
3. A sentence that calls its cited source or sources peer-reviewed, refereed or venue-accepted when none of them is is flagged.

Everything else is a known limit listed at the end of this lesson. When a new phrasing slips through, it becomes another line in that list unless it breaks one of the three guarantees or makes an ordinary, correctly cited answer fail; otherwise the rule count grows without converging.

## What the checker sees

`src/paperwatch/agent/citations.py` holds `check_answer(text, registry)`. The registry is the one Task 14 created for this answer, so the only aliases that can resolve are the ones a tool registered during this conversation. `check_answer` does no I/O, calls no model and reads no database: it is pure text plus that in-memory registry, which makes it cheap to run on every answer and easy to test.

Code spans, a closed code fence and URLs are content, never citations. Everything else is scanned in one pass with one combined pattern, which finds the protected spans and the citation groups together; because a protected span comes first in the alternation, a group can never start inside one, and detection, stripping and rewriting agree. That pass is linear, so a model stuck in a repetition loop (thousands of `[S1] http://a.b `) costs milliseconds rather than seconds.

A citation group is recognised by two rules:

- A square bracket that holds an alias token within 200 characters of its opening bracket, on one line, is a citation group, whatever else it holds: `[S1, S2, and S9]`, `[S1 S9]`, `[S1-S9]`, `[S9, p. 4]`, `[S1 AND S9]`. Every alias in it is checked, and an unknown one is flagged and shown as `?` in place while the rest of the wording is kept, so `[S1, S2, and S9]` becomes `[S1, S2, and ?]`. A token with a word character on either side (`pages 4`, `Williams 2023`) is not an alias.
- A paren group counts only when it holds aliases and separators alone (`(S9)`, `(S1, S2)`, `(S1; S2)`), and is then rewritten to the canonical `[S1, S2]` form. `(S3)` in ordinary prose means object storage, not a source, and is a known limit below.

Aliases are normalised to `S<digits>` whatever their case or spacing, zero-width characters between `S` and the digits are ignored rather than allowed to hide an alias, and non-ASCII digits are kept, so `[S٣]` is an alias nobody issued instead of an invisible one. A bare `S9` in prose is not a citation; only a bracket or paren group is.

The answer is split into sentences on newlines and after `.`, `!` or `?`, including when closing quotes, parens, or bold or italic marks follow the stop (`_italic._ Next`, `.)Next`), and after a `]` that is followed by whitespace and a capital letter (`"... tasks [S1] Most agents..."`). A full stop inside a bracket group (`[S9, p. 4]`) and after a listed abbreviation does not end a sentence; `etc.` ends one before a capital, because that is how it is used in a real sentence. Bullets (`-`, `*`, `1.`, `1)`, `•`, `·`) are stripped, and a fragment that is only a marker is attached to the sentence before it, because models write both `"...bugs [S1]."` and `"...bugs. [S1]"`.

A sentence is a claim when it has at least `CLAIM_MIN_WORDS` (6) words, is not a question, and is not layout: a heading, a table row, a line ending in a colon or a wholly bold line. A wholly bold line with a citation is a real sentence and is judged. An uncited line that is layout stays unflagged whatever it says, except that a tier phrase in ordinary prose with no citation ("It was accepted at ICLR.") is flagged `uncited_claim` at any length.

## The three flags

- `unknown_source`: a marker names an alias the registry never issued. The alias is masked in the returned text, so it can never reach the reader as a real source: `[S9]` becomes `[?]`, and a mixed marker keeps the invented alias visible in place, `[S1, S9]` becoming `[S1, ?]`. Masking happens before any rendering, so the invented alias is not merely flagged somewhere else.
- `uncited_claim`: a claim sentence with no citation at all, whether it is ordinary prose or a tier phrase. A sentence whose only marker is unknown is reported as `unknown_source` alone, because the model did point at something and the useful repair is "that source does not exist". Identical flags are reported once.
- `tier_mislabel`: a cited sentence whose sources are all below `peer_reviewed` while it claims otherwise, whatever its layout, so a "Venue" column in a table and a heading are checked too. The phrases are `peer review` in any hyphenation or spacing (including the non-breaking hyphen), `refereed`, `accepted at/to/for` and `published at`. A sentence may cite several sources and the label holds if any one of them supports it, so `[S1, S2]` with S2 peer-reviewed passes, and a source already cited earlier in the answer still counts, so repeating `[S1]` does not hide a preprint behind its first citation.

Negations are honoured only for the listed forms: `not`, `non`, `n't`, `never`, `no`, `un`, `without`, `awaiting`, `pending`, `yet to be`, `under` and `submitted for`, within three words before the phrase. That is what keeps the honest label the prompt asks for ("a preprint, not yet peer-reviewed", "hasn't been peer-reviewed", "under peer review") from being punished. `not only` and `not just` assert the opposite and are not negations. Anything outside the list ("was none too peer-reviewed") is a known limit.

Flags name the sentence they came from, with citations and a trailing full stop removed, so the reader can find it in the answer instead of hunting. `Flag.detail` clips any alias it echoes at 32 characters, since a model can write one tens of thousands of digits long.

Citations come back in first-seen order and once each, so the source list reads in the order the answer used them. `supported` is true exactly when there is at least one evidence citation and no flag. Evidence means a known citation beside at least `EVIDENCE_MIN_WORDS` (3) words, so a terse table row or bullet (`| Reflexion | verbal feedback loops [S1] |`) counts while `[S1]`, `ok [S1]`, `See [S1].` and `[S1][S2]` are the model stalling. Without evidence the returned text is the fixed `NO_SUPPORT` message rather than the model's uncited prose, and the flags are kept for the log but not shown: Tasks 16 and 17 do not render flags beside that message, because they describe sentences the reader will never see.

## Escaping and links

Everything `check_answer` returns is untrusted in one of two ways, and all of it must be escaped wherever it is rendered into a prompt, an SSE event or a terminal. `CheckedAnswer.text`, `Flag.sentence` and `Flag.detail` are model-authored; `Citation.title` and `Citation.url` are stored third-party data, from a paper or a blog post. Titles and URLs stay raw in `Citation` exactly as `SourceRef` holds them, so the renderer applies one escaping rule.

No markup the model wrote is rendered as a link, and every clickable link comes from `Citation.url`. A bare URL in the text stays visible as plain text and is not verified against the corpus; a terminal or chat client may still make it clickable, so the CLI either defangs URLs in the text or accepts that. This is how the spec's "every link must match the stored item's URL" is kept without parsing the answer, and there is no fourth flag kind for links.

## Known limits

- Sentences under six words are not checked for a citation, and numbers are not checked against the cited source anywhere in M2, so `supported` means "every visible claim points at a retrieved source", not "verified".
- A bare `S9` in prose is not read as a citation; only bracket and paren groups are. A bracket group holding unrelated text is read only through the alias tokens in it.
- `(S3)` for object storage and `(s1, s2)` for RL states in ordinary prose are read as citations when the alias exists, because a paren group of aliases alone is indistinguishable from a source marker.
- A heading, table row, wholly bold line, colon-ended line or fenced block can carry an uncited claim and stays unflagged as layout. An unclosed or inline triple-backtick line is ordinary text, so it is checked.
- Dotted abbreviations outside the listed set (`al`, `e.g`, `i.e`, `vs`, `Fig`, `Dr`, `Inc`, `Corp`, `Ltd`, `No`, `Prof`, `Sec`, `Eq`, `approx`, dotted initialisms such as `U.S.`) can split a sentence and produce a false flag.
- Any `published in` (a date or a venue) is not a tier claim, by decision: dating a source is ordinary and constant, and `published in Nature` goes unchecked too. Only `published at <venue>` is checked, and other venue words (`appeared at ICLR`, `a NeurIPS paper`, `presented at`) are not phrases at all.
- A general remark about peer review with no citation ("Treat these preprints with caution until peer review completes.") is flagged `uncited_claim`, because an uncited tier phrase is treated as a claim at any length.
- Negations outside the listed forms are not recognised, and a negator from an earlier clause within about three words can hide a real mislabel.
- The tier check is per sentence: a phrase about another paper than the one cited ("Reflexion is cited by Voyager, a peer-reviewed paper, ... [S1]") is flagged.
- Guarantee 2 counts cited text with at least three words, so a cited question or a cited heading (`## Results from the preprint [S1]`) is evidence, while `[S1]`, `[Source 1]` and `[1]` alone fail closed to the no-support message.
- Other bracket styles and notations (fullwidth `［S99］`, `{S99}`, `<S99>`, `^S99`, `(see S99)`, `(S99, p. 4)`) are not read as citations beside a valid marker.
- A sentence that starts, after a marker, with a digit, markup or a non-ASCII capital (`[S1] 41% of ...`, `[S1] **Most ...**`, `[S1] Über ...`) stays inside the cited sentence.
- A `> ` block quote is not stripped: the quoted line is an uncited claim and its flag keeps the `>`.
- A full stop directly before a capital splits a name (`Node.JS`, `Memory.Json`), like any unlisted abbreviation.

## What the tests pin

`tests/unit/test_citations.py` covers distinct citations in first-seen order, a grouped marker, a marker placed after the full stop, markers after bold, quote, paren and italic, a marker that ends a sentence without a stop, an unknown alias flagged and masked with the remaining valid citation kept, the in-place `[S1, ?]`, the wide square-bracket forms (`[S1, S2, and S9]`, `[S1 S9]`, `[S1-S9]`, `[S9, p. 4]`, `[S1 AND S9]`) and the paren forms (`(S1; S2)`, `(S1, S2)`, `(S1 and S2)`, `(S1, S2, and S2)`), a locator with a trailing letter (`[S2, pages 4-5]`, `[S1, Williams 2023]`) left untouched, zero-width characters, an uncited claim, a short connective, a heading, a bold heading, a table row, a code fence, a colon lead-in and an unclosed or inline fence, bullets and numbered items read as sentences, each listed abbreviation keeping its full stop, `etc.` before a capital ending the sentence and before a closer not ending it, a preprint called peer-reviewed flagged while the same sentence citing a peer-reviewed source passes, a repeated citation of a preprint still flagged, the tier phrases in prose, a table cell and a heading, the listed negations plus a curly apostrophe, a capitalised negation and the four-words-away case, `not only`, a dated publication left alone, the six-word claim boundary, the three-word evidence boundary, a cited table and terse bullets counted as evidence, a bold sentence with a citation, the flag sentence and detail, deduplicated flags, aliases in code and URLs ignored, a closed fence returned byte-identical, an unbalanced backtick that still masks, the alias clip boundary and wording, flags kept beside the no-support message, the no-evidence answer for an uncited text, a wholly unknown-cited text, `[S1][S2]` and `See [S1].`, a 5,000-repeat input checked in well under a second, and a 40-alias unclosed paren list plus a 50,000-space paren run checked under a second each. The plan's mutation check accepts every alias the model invents and makes the unknown-alias and no-evidence tests fail, which is what keeps the registry lookup from turning into an honour system.
