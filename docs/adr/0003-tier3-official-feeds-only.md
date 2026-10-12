# 3. Tier-3 lab content comes only from official RSS/Atom feeds

Date: 2026-10-12

## Status

Accepted

## Context

Tier 3 is the design's "lab engineering blog" reliability tier (design section 3.1). Its items are
less formal than papers, so the value is the lab's own first-party writing, not something
re-published or scraped third-hand. Scraped HTML is brittle (markup changes break the parser),
carries unclear reuse terms, and would make the corpus depend on page structure nobody promised to
keep stable. The design's decision 3 names candidate labs, including several Chinese labs, and
requires each to publish an official RSS/Atom feed or documented API listing.

## Decision

Paper Watch ingests a lab's blog only through that lab's own official RSS or Atom feed. A feed
counts as official when it is linked from the lab's own site (a `<link rel="alternate">` tag or a
documented `/rss` or `/feed` path) or listed in the lab's documented API. Each feed is verified once
at implementation time with a HEAD request that returns 200 and an XML content type, and the
verified list lives in `config/feeds.yaml` as data, not code. The client never scrapes HTML for
content. A lab without an official feed is not ingested through tier 3; its papers still arrive
through the tier 1/2 affiliation signals if they exist. Tier-3 text is stored for private search
only, with the licence recorded as "copyright (private search only)".

## Lab list

The full list of included and excluded labs is being researched separately by a dedicated scout
step, because feed URLs change and each candidate needs its own verification. `config/feeds.yaml`
therefore currently contains one verified feed, the Hugging Face blog feed
(`https://huggingface.co/blog/feed.xml`). The researched list, with the reason for every lab that
was checked and excluded, will be filled into `config/feeds.yaml` and this section in a follow-up
commit. This ADR's decision does not depend on which labs that research admits.

## Consequences

- A feed outage is a reported failure, not a crash: the source records the feed URL and error and
  keeps the other feeds flowing.
- Adding or removing a lab is a one-line change to `config/feeds.yaml`, reviewed like any other
  configuration change.
- Tier 3 cannot reach a lab that publishes only through an SPA or a PDF; that is an accepted loss
  of coverage, not something to work around with scraping.
