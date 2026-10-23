# 9. Ingestion runs on the shared VPS as systemd user units

Date: 2026-10-22

## Status

Accepted

## Context

The corpus only has value if it grows every day, and the project's cloud budget is zero, so the
daily ingestion had to run somewhere free that is already paid for. The server at hand is shared
with another production project: Debian 13, 2 CPU, 1.9 GB RAM of which about 1.3 GB is available,
2 GB of swap with swappiness 10, 48 GB of disk free, clock in Europe/Berlin (the same UTC offset
as Rome all year) and NTP on. The neighbour uses about 650 MB and has its own services and its own
Postgres. The server already runs Postgres 17.11 with `vector` 0.8.0, so a database there costs no
money and no memory of its own.

The alternatives were a managed vector store toward the same end (ADR 0005 already picked one
Postgres for items, vectors and full text), a managed Postgres on a free tier whose storage cap
the corpus would reach, and Docker Compose on the VPS as the design originally sketched. Docker
needs root to install and would add a daemon and a container image to a machine that already
carries another team's production, on 1.3 GB of free memory.

## Decision

Production ingestion runs on the shared VPS as two systemd user units of the `paperwatch` account,
against the server's Postgres over the local Unix socket with peer authentication (its own
database `paperwatch`, its own role, no password, no TCP):

- **No Docker and no root step.** `paperwatch-ingest.service` runs `uv run --frozen --no-dev
  paperwatch ingest --live --followup` from a checkout in `/home/paperwatch/paper-watch`, pinned to
  a release tag rather than to `main`. Linger is on for the account, so the user manager keeps the
  timers after logout. Everything the deployment needs lives in that home.
- **Daily run at 06:00 Europe/Rome**, `Persistent=true` so a missed trigger catches up. The unit
  is one `Type=oneshot` service with `MemoryMax=950M` and `MemorySwapMax=0`: with no swap the
  cgroup OOM killer stops a runaway run instead of letting it crawl in the swap the neighbour
  shares. `TimeoutStartSec=17h` sits above the 16 h follow-up window the run caps itself at (ADR
  0008), and `Nice=10` / best-effort I/O keeps a fitted model from starving the other project.
- **Nightly `pg_dump -Fc` at 05:15**, written under a temporary name and renamed only when
  complete, keeping seven nights.
- **A morning `check.sh`** the captain can read over SSH: last run status and verdict, problems and
  warnings, counts and duration, unit and timer state, admitted items, backup age and size, free
  disk, a `WAIT` line while a run is between follow-up passes, and a `FAIL` line per bad check with
  exit 1 instead of 0.
- **No change to networking, firewall, SSH, proxy or the Postgres server configuration.** Those
  are root steps on a shared machine and stay the captain's decision.

The guide is `docs/deploy/vps.md`.

## Consequences

- Deploying is a tag checkout plus `uv sync` (the unit itself runs `uv run --frozen --no-dev`),
  with no image to build, publish or pull; the repository keeps no container image for the
  application. A rollback is checking out the previous tag.
- The database is the server's own Postgres, so there is no managed backup, failover or point-in-
  time recovery: the nightly dump is the recovery plan, and its restore drill is still M4 work.
- The no-root boundary means no Postgres server tuning on this host, which would be a server
  configuration change on a machine another project depends on. Two CPU and 1.3 GB of free memory
  are the budget the run fits in.
- The measured ingestion peak is 706.8 MB with no swap (2026-10-22), which is what `MemoryMax=950M`
  is derived from. This supersedes the server facts stated in ADR 0004 (1.9 GB RAM, no swap, a
  neighbour using about 150 MB): the machine now has 2 GB of swap and a neighbour using about
  650 MB, and the run has been measured rather than estimated.
- The swap file, the role, the extension and linger were set up once by the captain before the
  guide starts, so a reinstall reproduces them only through those steps, not through the repo.
