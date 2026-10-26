# 17. The API runs under its own Unix account

Date: 2026-10-26

## Status

Accepted.

Supersedes the "the API is a `paperwatch` user unit" half of ADR 0012's and ADR 0013's unit layout:
the tunnel, the ingestion and the backup stay user units for `paperwatch`, and the API is a system
unit under `paperwatch-api`.

## Context

The API and the ingestion ran as the same Unix account, `paperwatch`. Peer authentication maps that
account to the `paperwatch` role, which owns the database, runs the migrations and can read every
table. The env file that holds the chat key, and the directory the nightly `pg_dump` writes into,
are in the same home.

So one remote code execution in the API, in a route, a dependency or a model-written tool call, is
the owner of the corpus: it can rewrite items, drop tables, read this week's backups, and read the
ingestion's env file. The API also reaches the internet by definition (it calls an LLM provider and
is published through a tunnel), while the ingestion reaches only four known sources, so the API is
the larger surface. Least privilege for its database role was already in place
(`paperwatch_api`, read everything and write only the rows the docs name); the Unix account was not.

## Decision

The API runs as its own Unix account, `paperwatch-api`, with its own home, its own checkout of the
release it serves, its own `uv` and venv, its own model cache and its own env file. Peer
authentication maps `paperwatch-api` to `paperwatch_api` and nothing else, and the `paperwatch`
account keeps its own lines for the owner role and the read-only role Beekeeper uses.

It becomes a **system** unit (`/etc/systemd/system/paperwatch-api.service`, `User=paperwatch-api`)
instead of a `paperwatch` user unit, because a system account needs no lingering user manager and a
unit that runs as another account is the system manager's to start. The tunnel, the ingestion and
the backup stay user units for `paperwatch`: nothing about them changes, and linger stays on for
that account.

The unit also carries the cheap sandboxing that fits the account: `ProtectSystem=strict` and
`ProtectHome=read-only` with only `/home/paperwatch-api/.cache` and the venv writable,
`PrivateTmp=yes`, `PrivateDevices=yes`, no capabilities, no new namespaces, no setuid bit, no
personality change.

## Consequences

- An API RCE can read every admitted item and write the rows the docs list, and it can read its own
  env file because it must. It cannot write the corpus, run the migrations, read the backups, or
  read the ingestion's env file, which holds the owner role.
- The model cache is downloaded twice, once per account, about 180 MB each. That is the price of not
  letting the API traverse `/home/paperwatch`, which holds the other account's secrets and backups;
  disk is not the scarce resource here.
- A release is fetched twice: the ingestion's checkout and the API's, each at the same tag, each
  with its own `uv sync --frozen --no-dev`. The migration (`alembic upgrade head`) stays with the
  ingestion's account, which owns the schema.
- `paperwatch ask` on the server now runs as the API's account
  (`sudo -u paperwatch-api -- env HOME=...`), because the chat key moved with the API.
  `docs/deploy/vps.md` section 14 has the exact command.
- **The shared `paperwatch.slice` no longer spans both.** A systemd slice cannot contain units from
  two managers, so the API left it and the ingestion keeps it to itself. The protection the slice
  gave on a 1.9 GB machine survives on two other settings: each unit has its own no-swap
  `MemoryMax` (API 700M, ingestion 950M), which bounds it absolutely, and the API keeps
  `OOMScoreAdjust=900` while the ingestion runs at `Nice=10`, so when the machine is short the
  kernel picks the API and the morning run finishes. What is lost is a single ceiling on their sum;
  what is gained is that neither account's unit can be starved by the other's accounting.
- `ProtectHome=tmpfs` with read-only binds was the alternative sandboxing when one account had to
  serve both roles. With two accounts it would need four read-only binds (the checkout, the venv,
  the `uv` binary and the env file) and one missed path stops a live unit, so the account boundary
  does that work instead.
- The migration for the running server is `docs/deploy/vps.md` section 18: an ordered procedure with
  a rollback. It is applied by the owner, separately from the release that changed the repository.
