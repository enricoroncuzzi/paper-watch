# 10. Every source request is bounded by a body cap, a wall-clock deadline and a redirect policy

Date: 2026-10-23

## Status

Accepted

## Context

The ingestion run shares one HTTP client across arXiv, the lab feeds, Hugging Face and Semantic
Scholar, and the client's pool is one connection because arXiv's terms allow one. Everything the
run fetches is parsed in memory straight after it is read, and the run has a hard ceiling: the
systemd unit runs with `MemoryMax=950M` and no swap, so a run that exceeds it is OOM-killed by the
cgroup and the morning is lost (ADR 0009 records the 706.8 MB measured peak).

Three shapes of the problem were open. A response body is held and parsed whole, so a broken or
hostile answer is a memory event, not just a slow one: measured through `send_with_retries` into
the arXiv parser, a 49 MiB arXivRaw page of 21 360 records grew RSS by about 380 MiB. httpx's
30 s read timeout restarts on every byte, so it bounds one read and not one request: a server that
drips a header line every 20 s is never timed out, and h11 allows 16 KiB of headers, which is days
of a drip. And httpx followed redirects by itself, so a hop could downgrade the connection to plain
http or take a configured lab feed off its own site without anything noticing, while Semantic
Scholar's batch endpoint wants redirects refused outright.

## Decision

`src/paperwatch/sources/base.py` owns three limits, applied to every request:

- **A body cap per source**, counted on the bytes as they arrive and abandoned at the first byte
  over it: arXiv 16 MiB, Hugging Face and Semantic Scholar 8 MiB, lab feeds 10 MiB. Each is about
  5x to 10x the real page (an arXiv day page of a few MB, a 500-id batch answer or a 100-row
  listing under 1 MB), which keeps the parse growth under about 130 MiB, well inside the roughly
  243 MB of headroom the measured peak leaves under `MemoryMax=950M`. Requests ask for
  `Accept-Encoding: identity` so a compressed body cannot expand past the cap on decode. Over the
  cap is a fetch failure for that source, dead-lettered like any other; it is never retried, since
  the same body would arrive again.
- **One wall-clock deadline per attempt, covering the whole request**: connect, response headers
  and body, 120 s, shared by the hops of one redirect chain. It is enforced from outside the
  request, in a worker thread: the caller waits the budget, then shuts the connection's socket
  down (`SHUT_RDWR`, taken from httpx's `trace` extension as soon as the connection exists) and
  closes the response, both so the request actually stops and so the client's single pooled
  connection comes back for the next source. Closing alone does not wake a thread blocked in
  `recv`, which is why the shutdown is there. httpx per-request timeouts could not do this: they
  bound each phase, not the request, so they cannot bound a drip. The cost is one short-lived
  thread per request, which is what makes the bound hold for a server that answers nothing.
- **A redirect policy**: redirects are followed by hand, never by the client, and every hop is
  checked before the request goes out. https to http is always refused; a feed must stay on the
  registrable domain of the URL configured in `config/feeds.yaml`; more than five hops is refused;
  a non-http(s) target is refused. A feed's registrable domain is its eTLD+1 from the public suffix
  list (`publicsuffixlist`, a small pure-Python package with the list bundled, so no run does a DNS
  lookup or a download to make this decision). eTLD+1 is what makes "same site" mean anything:
  comparing the last two labels would let `evil.co.uk` pass for `lab.co.uk`, and comparing hosts
  exactly would refuse the `blog.example.co.uk` -> `www.example.co.uk` moves the rule exists to
  allow. Semantic Scholar asks with redirects refused, so a moved batch endpoint is an answer this
  client will not take rather than a silent detour.

Retryable statuses are 429, 500, 502, 503 and 504 under the one tenacity policy; every other
answer, a 4xx in particular, is deterministic and never retried. A chunk Semantic Scholar turns
away with a retryable status stays due, so a follow-up pass asks again (ADR 0008).

## Consequences

- A body over its cap, a deadline that expires and a refused redirect all arrive as ordinary fetch
  failures for that source: the feed is dead-lettered with its cursor pinned, arXiv degrades the
  run, Hugging Face and Semantic Scholar record a degraded source. None of them can take the run
  down, and none of them can allocate without bound.
- The caps bound bytes, not elements: a hostile page of millions of tiny elements can still grow
  the parse by more than its size, and a server that ignores `Accept-Encoding: identity` gets one
  decoded chunk past the cap before the read stops. Both need a hostile or badly broken official
  host, which the deadline does not help with; they are recorded here rather than fixed.
- The deadline is real time, so the tests that exercise it use sub-second budgets and loopback
  sockets, not a mocked clock: the point is that a blocked socket read wakes. The loopback drip
  test (`test_an_abandoned_request_frees_the_shared_connection`: a header drip and a body drip, each
  on a fresh and on a reused pooled connection) is the guard for the httpcore internals the abort
  depends on, which no httpx API promises: the trace event names `connection.connect_tcp.complete`
  and `connection.start_tls.complete` that hand over the connection, and `get_extra_info("socket")`
  returning the socket to shut down. The `test` CI job fails if any of its four cases is skipped or
  not collected, and a failure to start its loopback server fails the test rather than skipping it.
- A feed that legitimately moves to a different registrable domain now fails until
  `config/feeds.yaml` names the new address, which is a visible morning FAIL rather than a silent
  change of subject.
- The corpus gains a third-party dependency (MPL-2.0) and a thread per HTTP request. Both are
  small; the alternative to the first is a rule that is wrong about real domains, and the
  alternative to the second is a bound that does not hold.
