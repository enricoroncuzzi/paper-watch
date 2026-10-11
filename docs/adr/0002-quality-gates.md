# 2. Quality gates

Date: 2026-10-11

## Status

Accepted

## Context

A change can break Paper Watch in ways a single test run does not catch: formatting and import
drift, type errors, a committed secret, a dependency with a known vulnerability, a dangerous code
pattern, or a version number updated by hand and forgotten. Each of these has a cheap automated
check, but only if the checks run on every change and the cheapest, most informative failure
reports first.

## Decision

Every pull request and every push to `main` runs the same gate in `.github/workflows/`. The jobs
run in parallel and are listed cheapest first, which is also the order in which they usually
report, because cheaper jobs finish sooner:

1. `lint` (ruff check and format) - finishes in seconds, so the cheapest mistakes report first.
2. `typecheck` (mypy strict) - catches interface drift, usually reporting before the slower test job
   finishes.
3. `test` (pytest) - the behavioral gate, including integration tests against Postgres + pgvector.
4. `security` - pip-audit for known-vulnerable dependencies and gitleaks for committed secrets.
5. `codeql` - static analysis for dangerous code patterns, plus a weekly scheduled scan so new
   queries reach the repository without a code change.

Once retrieval exists (M1), an `eval` job joins the gate after `test`. It runs offline with the
deterministic fake embedder against a committed metric baseline, so default CI still never calls a
model.

Dependabot keeps the `uv` lockfile and the pinned GitHub Actions current with weekly pull requests,
so the gate stays meaningful over time. Releases are derived from history: release-please reads
conventional commits on `main` and opens a release pull request with the next semantic version and
a generated changelog. `lint`, `typecheck`, `test`, and `security` become required status checks on
`main`.

## Consequences

- A change is trustworthy when the same gate every other change passed also passes it, not because
  an author says so.
- All jobs start together, so a typo still fails `lint` within seconds and reports before the
  slower test and security jobs finish. A failing job does not stop the others.
- Security is layered (secrets, dependencies, code patterns) because no single tool covers all three.
- release-please makes version numbers a consequence of commit history, so versioning is not a
  manual step that can drift.
- The gate adds CI minutes per pull request and depends on the repository allowing GitHub Actions to
  create pull requests for the release job.
- release-please opens its release pull request with the default `GITHUB_TOKEN`, and GitHub does
  not trigger `ci` or `codeql` for pull requests created with that token. Once the checks are
  required, the maintainer either closes and reopens the release pull request, which runs the
  checks under the maintainer's identity, or merges it with an admin bypass. Passing a repo-scoped
  GitHub App token or fine-grained PAT as the action's `token` input is the later upgrade path.
