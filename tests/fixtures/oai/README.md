# arXiv OAI-PMH fixtures

Recorded from the live arXiv OAI-PMH endpoint on 2026-10-16, one request every
few seconds on a single connection, for datestamp day 2026-10-06.

Real captures (records kept exactly as arXiv returned them; the record list is
trimmed and, for the two cs pages, the `resumptionToken` is edited):

- `statml_day.xml` - `verb=ListRecords&metadataPrefix=arXivRaw&set=stat:stat:ML&from=2026-10-06&until=2026-10-06`.
  Records: `1909.09849` (a cs.MA/cs.AI/cs.LG/stat.ML paper with four versions,
  also seen in the cs set) and `1506.04782` (cs.LG/stat.ML, two versions, no
  profile term). The empty `resumptionToken` is real: the day fits one page.
- `cs_page1.xml` and `cs_page2.xml` - `verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from=2026-10-06&until=2026-10-06`,
  split by hand into a two-page pair. `cs_page1.xml` ends with a synthetic
  `cs-token-2`; `cs_page2.xml` ends with an empty token. Records: `1909.09849`
  (matching agent paper), `2503.21975` (cs.RO/cs.AI, "robot" but no profile
  term), `2301.07757` (cs.CG/cs.CC, outside every profile category),
  `2404.09982` and `2610.04539` (two more matching agent papers).

Hand-written (element shapes copied from the real responses):

- `noRecordsMatch.xml` - an empty datestamp day.
- `badResumptionToken.xml` - an expired resumption token.
- `deleted.xml` - a deleted record followed by a valid one.

Recorded with the URL and the trimmed response kept in
`data/awakening-m1/` outside the repository; the raw captures are not committed.
