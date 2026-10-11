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
6. `workflow-lint` - actionlint (workflow syntax, expressions, and shellcheck over `run:` blocks)
   and zizmor (workflow security: permissions, template injection, unpinned actions, persisted
   credentials), run from the same pinned pre-commit hooks a local commit uses.

Every job installs with `uv sync --locked`, which fails when `uv.lock` no longer matches
`pyproject.toml` instead of silently installing a stale lock. Checkouts do not persist the token,
every job has a short `timeout-minutes`, and every action is pinned to a full commit SHA with its
release in a trailing comment. On pull requests a newer push cancels the running check; on `main`
every commit keeps its own run, so two quick merges each get a verdict.

Once retrieval exists (M1), an `eval` job joins the gate after `test`. It runs offline with the
deterministic fake embedder against a committed metric baseline, so default CI still never calls a
model.

Dependabot keeps the `uv` lockfile, the SHA-pinned GitHub Actions, and the pre-commit hook revs
current with weekly pull requests, so the gate stays meaningful over time. Each ecosystem waits a
seven-day cooldown after a release and groups minor and patch bumps into one pull request. Releases
are derived from history: release-please reads conventional commits on `main` and opens a release
pull request with the next semantic version and a generated changelog. Before 1.0.0 a `feat:` bumps
the minor version rather than the major (`bump-minor-pre-major`). The release pull request also
bumps the project's own `paperwatch` version in `uv.lock` (a TOML `extra-files` entry), so the
lockfile still matches `pyproject.toml` and `uv sync --locked` keeps passing after a release. `lint`, `typecheck`, `test`, and `security` become required status checks on
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
- Dependabot's `uv` ecosystem raises the lower bounds in `pyproject.toml`, not only the locked
  versions. That is acceptable for an application, which nobody installs alongside other
  constraints; a published library would want to keep its lower bounds wide.
- Dependabot pull requests run without repository secrets, so default CI must stay offline: no job
  may need an API key or a model call to pass.
