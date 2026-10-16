# arXiv OAI-PMH fixtures

Recorded from the live arXiv OAI-PMH endpoint on 2026-10-16, one request every
few seconds on a single connection, for datestamp day 2026-10-06. The raw
responses are whitespace-trimmed: a trailing space on the `<OAI-PMH ...>`
declaration line is stripped so `pre-commit` stays clean, and every record
element is byte-identical to arXiv's.

Real captures (the record list is trimmed; for the two cs pages the
`resumptionToken` is edited; the stat.ML day's empty token is real):

- `statml_day.xml` - `verb=ListRecords&metadataPrefix=arXivRaw&set=stat:stat:ML&from=2026-10-06&until=2026-10-06`.
  Records: `1909.09849` (a cs.MA/cs.AI/cs.LG/stat.ML paper with four versions,
  also seen in the cs set) and `1506.04782` (cs.LG/stat.ML, two versions, no
  profile term).
- `cs_page1.xml` and `cs_page2.xml` - `verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from=2026-10-06&until=2026-10-06`,
  split by hand into a two-page pair. `cs_page1.xml` ends with a synthetic
  `cs-token-2`; `cs_page2.xml` ends with an empty token. Records: `1909.09849`
  (matching agent paper), `2503.21975` (cs.RO/cs.AI, "robot" but no profile
  term), `2301.07757` (cs.CG/cs.CC, outside every profile category),
  `2404.09982` and `2610.04539` (two more matching agent papers). In the
  real-fixture test both dropped records fail the term filter; the category
  rule is pinned separately by the synthetic
  `test_category_filter_drops_records_outside_every_profile`.

Hand-written (element shapes copied from the real responses):

- `noRecordsMatch.xml` - an empty datestamp day.
- `badResumptionToken.xml` - an expired resumption token.
- `deleted.xml` - a deleted header followed by a valid record. The deleted
  `<header status="deleted">` shape follows the OAI-PMH spec and was not
  observed on arXiv; the valid record is a hand-edited copy of `2404.09982`.

To re-record, ask arXiv for that datestamp day, one request every few seconds on
one connection:

    curl 'https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from=2026-10-06&until=2026-10-06'

The raw captures are not committed; they were kept outside the repository while
trimming.
