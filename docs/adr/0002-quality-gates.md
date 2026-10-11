# 2. Quality gates

Date: 2026-10-11

## Status

Accepted

## Context

A change can break Paper Watch in ways a single test run does not catch: formatting and import
drift, type errors, a committed secret, a dependency with a known vulnerability, a dangerous code
pattern, or a version number updated by hand and forgotten. Each of these has a cheap automated
check, but only if the checks run on every change and in an order that surfaces the most
informative failure first.

## Decision

Every pull request and every push to `main` runs the same ordered gate in `.github/workflows/`:

1. `lint` (ruff check and format) - seconds, catches the cheapest mistakes first.
2. `typecheck` (mypy strict) - catches interface drift before tests run.
3. `test` (pytest) - the behavioral gate, including integration tests against Postgres + pgvector.
4. `security` - pip-audit for known-vulnerable dependencies and gitleaks for committed secrets.
5. `codeql` - static analysis for dangerous code patterns, plus a weekly scheduled scan so new
   queries reach the repository without a code change.

Dependabot keeps the `uv` lockfile and the pinned GitHub Actions current with weekly pull requests,
so the gate stays meaningful over time. Releases are derived from history: release-please reads
conventional commits on `main` and opens a release pull request with the next semantic version and
a generated changelog. `lint`, `typecheck`, `test`, and `security` become required status checks on
`main`.

## Consequences

- A change is trustworthy when the same gate every other change passed also passes it, not because
  an author says so.
- Ordering the gate cheapest-first means a typo fails in seconds, before the slower test and
  security jobs start.
- Security is layered (secrets, dependencies, code patterns) because no single tool covers all three.
- release-please makes version numbers a consequence of commit history, so versioning is not a
  manual step that can drift.
- The gate adds CI minutes per pull request and depends on the repository allowing GitHub Actions to
  create pull requests for the release job.
