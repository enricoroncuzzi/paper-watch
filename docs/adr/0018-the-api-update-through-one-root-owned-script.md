# 18. The routine API update runs through one root-owned script

Date: 2026-10-29

## Status

Accepted.

Extends [ADR 0017](0017-the-api-under-its-own-unix-account.md): the API keeps its own Unix account
and its own system unit, and the routine update gains one narrow way to install a release into that
account's checkout.

## Context

ADR 0017 moved the API to `paperwatch-api`, with its own checkout, venv and system unit. That is
the account boundary the API's blast radius depends on, and it is also why none of the API's files
belong to the routine update's login: the checkout is `paperwatch-api`'s and the unit and its
restart are root's. The routine update runs over ssh as `paperwatch`, which has no sudo, so every
release afterwards needed the owner at a root prompt for the API's checkout, venv and restart.

Two ways out were on the table: the owner runs the API's half of each update by hand with sudo, or
the server hands the `paperwatch` login one narrow root permission. The first keeps the permission
model exactly as it is and pays a manual step on every release; the second puts a root-owned
program on the update path, and it is the one chosen.

## Decision

One root-owned script, installed as `/usr/local/sbin/paperwatch-update-api` from
`deploy/vps/update-api.sh`, and one sudoers line from
`deploy/vps/sudoers-paperwatch-update-api`:

```
paperwatch ALL=(root) NOPASSWD: /usr/local/sbin/paperwatch-update-api
```

The login can run that command and nothing else. It takes exactly one release tag, validated inside
the script because a sudoers command entry permits any argument to the command it names, and it
takes no repository URL, path, account or timeout from its environment: the drop-in leaves sudo's
`env_reset` on and grants no `SETENV`.

Every repository step runs as `paperwatch-api` with a cleared environment: fetch the tags from the
clone's existing origin, check out the tag, `uv sync --frozen --no-dev`, restart `paperwatch-api`,
wait for `/healthz` with a bound, then run the release's `api-check.sh` as the API's account. A
failure after the checkout moves checks the previous release back out, syncs and restarts it, and
exits non-zero.

It never installs a unit. A release whose `deploy/vps/paperwatch-api.service` differs from the
installed one stops the script before the checkout moves: a unit change is a compatibility change
(a sandbox path, the `ExecStart` line, a memory cap), and installing it stays the owner's step.

The script is installed `install -o root -g root -m 0755` by the owner from a root-owned clone,
`/usr/local/src/paper-watch`, that root fetches and checks out at the release tag itself. Neither
`paperwatch`'s nor `paperwatch-api`'s checkout is an install source: whatever root installs from a
copy an account can edit, that account can turn into root, so a compromised API account must not
reach root through it either. The drop-in is compared with its one expected line, because
`visudo -cf` checks only that it parses, then validated with `visudo -cf` and placed at mode 0440.
A new release of the script itself is reinstalled the same way, by the owner: the command updates
the API, never the root-owned program that runs it.

## Consequences

- A release can be installed without the owner, and the permission it takes is one program rather
  than root. The `paperwatch` login can name a release tag and nothing else, and tags come from
  origin, which that account cannot write.
- The script is a second thing to keep current. Between its own releases, an API release that
  changes how updating works waits for the owner to reinstall it, which is the price of the script
  not being replaceable by the account it runs for.
- The API account's boundary is untouched: the checkout and venv remain `paperwatch-api`'s, the
  unit and its restart remain root's, and the ingestion's account is not involved at all. The
  migration stays on the ingestion's side, where the owner role lives.
- An ingestion run does not matter to this update: there is no ordering or relationship between the
  two units (ADR 0013, ADR 0017), and restarting the API neither stops nor delays a run.
- With `PW_API_AUTH=oidc` the acceptance check skips its authenticated lines, because a token cannot
  survive sudo's clean environment; the HMAC path mints its own tokens. The check says so in its
  output rather than passing silently.
- `docs/deploy/vps.md` section 11 now ends the routine update with the one command, and section 11.1
  is the one-time install. The install is the owner's, from root's own clone, and running or
  installing a copy from either account's checkout is not: that account could edit the copy.
