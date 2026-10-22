# Running Paper Watch's ingestion on the shared VPS

The corpus has to accumulate every day, whatever the captain is building, so ingestion runs on
the shared VPS ahead of M3. Two systemd user units, both as the `paperwatch` account: the daily
ingestion at 06:00 Europe/Rome, and a nightly `pg_dump` backup at 05:15. The API, the weekly
shortlist, email alerts and the restore drill are not part of this deployment (M3 and M4).

Everything here is a step for the `paperwatch` user. There is no root step: the database, the
role, pgvector, the swap file and linger were set up by the captain before this guide starts.

## Server facts (verified 2026-10-22 over SSH as `paperwatch`)

- Debian 13, 2 CPU, 1.9 GB RAM (about 1.3 GB available), 2 GB swap (swappiness 10), 48 GB disk
  free, clock in Europe/Berlin (the same UTC offset as Rome all year), NTP on.
- Shared with jd-scraper, a production service using about 650 MB: three Python services and its
  own Postgres.
- Postgres 17.11. Database `paperwatch`, owned by role `paperwatch`, with `vector` 0.8.0 enabled.
  Peer authentication over the local Unix socket: no password, no TCP.
- Linger is on for `paperwatch`, so the user's systemd manager keeps running while nobody is
  logged in. `loginctl show-user paperwatch -p Linger` prints `Linger=yes`.
- Python 3.13 and git are installed. `uv` is not, and lives in the user's home (step 1).
- The SSH alias is `shared-vps`.

## Boundaries (binding on every change here)

- Everything lives in `/home/paperwatch/` and runs as `paperwatch`.
- Never read, stop, restart or modify anything under `/home/enrico/`, the `jds-*` services, their
  watchdog, or the `jds` database. The two deployments share a machine, not a schema.
- No changes to networking, firewall, SSH, VPN/proxy, system packages or the Postgres server
  configuration. A root step is the captain's decision, never a shortcut.
- No Docker here: the units run the checkout directly.

## 1. Install uv

```bash
ssh shared-vps
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs into ~/.local/bin, no root
~/.local/bin/uv --version
```

## 2. Clone the repository at a release tag

```bash
git clone https://github.com/enricoroncuzzi/paper-watch ~/paper-watch
cd ~/paper-watch
git checkout paperwatch-v0.3.0        # tags are paperwatch-vX.Y.Z
```

Pin a tag, not `main`: a checkout that changes under a running ingestion is a different program
halfway through. `deploy/vps/` ships the units and the two scripts, and git keeps them executable.

## 3. Create the environment

```bash
uv sync --frozen --no-dev
```

`--frozen` installs exactly the committed `uv.lock` and never rewrites it; `--no-dev` leaves out
the test, lint and type-check tools, which this server never runs.

## 4. Write the environment file

```bash
mkdir -p ~/.config/paperwatch
umask 077
cat > ~/.config/paperwatch/env <<'EOF'
PW_DATABASE_URL=postgresql+psycopg://paperwatch@/paperwatch?host=/var/run/postgresql
PW_USER_AGENT=paper-watch/0.3 (+https://github.com/enricoroncuzzi/paper-watch; you@example.com)
PW_LOCAL_EMBEDDING_CACHE_DIR=/home/paperwatch/.cache/paperwatch-models
EOF
chmod 600 ~/.config/paperwatch/env
```

- No host and no port: `?host=/var/run/postgresql` is the socket directory, so psycopg connects
  over the socket and peer authentication identifies the role by the Unix user. There is no
  password anywhere, and the URL cannot silently fall back to TCP.
- Replace the address in `PW_USER_AGENT`: arXiv asks bulk harvesters for a contact.
- The model cache stays outside the checkout, so switching to a new release does not throw away
  the 180 MB embedding model.
- No chat key: ingestion never calls a chat model.

Confirm the connection works without a password:

```bash
psql -d paperwatch -c 'select current_user, current_database()'
```

## 5. Create the schema

```bash
uv run --frozen --no-dev alembic upgrade head
psql -d paperwatch -c '\dt'      # items, tier_history, runs, dead_letters, source_cursors, chat_calls
```

The role owns the database, so no grant is needed. Every later release runs the same command
again; migrations are additive one version at a time.

## 6. Install and enable the units

```bash
mkdir -p ~/.config/systemd/user
cp ~/paper-watch/deploy/vps/paperwatch-ingest.service \
   ~/paper-watch/deploy/vps/paperwatch-ingest.timer \
   ~/paper-watch/deploy/vps/paperwatch-backup.service \
   ~/paper-watch/deploy/vps/paperwatch-backup.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now paperwatch-backup.timer paperwatch-ingest.timer
systemctl --user list-timers 'paperwatch-*'
```

`list-timers` shows the next trigger for each: the backup at 05:15, the ingestion at 06:00, both
in Rome time. `Persistent=true` means a trigger missed while the machine was down fires once when
it can, instead of being skipped for the day.

What the units do, and why:

- `paperwatch-ingest.service` runs
  `uv run --frozen --no-dev paperwatch ingest --live` from `/home/paperwatch/paper-watch` with
  the environment file above. It is capped at `MemoryHigh=900M` / `MemoryMax=1200M` so the
  embedding model can never push the scraper into swap, and it runs at low CPU and I/O priority
  (`Nice=10`, best-effort priority 7) so a morning run does not slow the scraper down.
  `TimeoutStartSec=3h` covers a slow first run.
- `paperwatch-backup.service` runs `deploy/vps/backup.sh`: `pg_dump -Fc` into
  `~/backups/paperwatch-<UTC stamp>.dump`, keeping seven nights, written under a temporary name
  and renamed, so an interrupted dump never looks like a good one.
- The backup is ordered before the ingestion, not required by it: a failed backup must not cost a
  day of corpus.

## 7. The first run, measured

Run it by hand under `time` first, so the memory cap is tuned from a measurement instead of a
guess:

```bash
cd ~/paper-watch
/usr/bin/time -v uv run --frozen --no-dev paperwatch ingest --live
```

Record the `Maximum resident set size`, the wall clock and the last line (`succeeded: {...}`).
The first run downloads the model (about 180 MB, once) and covers a wider window than a daily one,
so it is the worst case. The local model peaks near 520 MB; if the measured peak is within about
300 MB of `MemoryMax`, raise the caps in
`~/.config/systemd/user/paperwatch-ingest.service`, then `systemctl --user daemon-reload`.

Confirm the caps are really applied, which needs the memory controller delegated to the user
manager (Debian's `user@.service` does that):

```bash
systemctl --user show paperwatch-ingest.service -p MemoryHigh -p MemoryMax
```

Then start the timer's first run the same way the timer will:

```bash
systemctl --user start paperwatch-ingest.service
journalctl --user -u paperwatch-ingest -n 50 --no-pager
```

## 8. The morning check

```bash
~/paper-watch/deploy/vps/check.sh
```

It prints the last finished run's status, health verdict and problems, the run's counts and
duration, the total admitted items, the newest backup's age and size, and free disk. Exit code 0
means every check passed, 1 means at least one is bad, and the bad lines start with `FAIL`:

- no finished run, or a status other than `succeeded` or `partial`;
- an unhealthy or missing health verdict, or a non-empty list of problems;
- the last finished run older than 26 hours (the freshness SLO), so a missed morning is caught
  even when the previous run was perfect;
- no dump in `~/backups/`, or the newest older than 26 hours or empty;
- less than 2 GB free in the home.

Thresholds can be overridden for one invocation, for example while tuning:
`PW_CHECK_MAX_RUN_AGE_S`, `PW_CHECK_MAX_BACKUP_AGE_S`, `PW_CHECK_MIN_FREE_GB`,
`PW_CHECK_BACKUP_DIR`, `PW_CHECK_DB`. It needs only `psql` and coreutils, and reads only the
`paperwatch` database.

## 9. Exit codes and the journal

`paperwatch ingest --live` tells systemd exactly what happened:

| Exit | Meaning | What to do |
| --- | --- | --- |
| 0 | The run finished and its health verdict is healthy. | Nothing. |
| 3 | The run finished (`succeeded` or `partial`) but judged itself unhealthy: a source returned nothing, a cursor did not advance, a signal source was degraded, an item failed. The problems are printed to stderr. | Read the problems; the corpus is intact but a source needs a look. |
| 1 | The run did not finish its work (a failure, or the token budget was exhausted). | Read the journal; the next run resumes from the stored cursors. |
| 0 | Another ingestion held the single-run lock, so this invocation did nothing (`skipped`). | Nothing. |
| 2 | A usage error, for example a missing `--live`. | Fix the command. |

systemd records 1 and 3 as a failed unit, which is deliberate: both deserve a look, and
`systemctl --user list-units --failed` is a second place to notice. The problems themselves are in
`journalctl --user -u paperwatch-ingest`.

## 10. Updating to a new release

```bash
cd ~/paper-watch
git fetch --tags
git checkout paperwatch-vX.Y.Z
uv sync --frozen --no-dev
uv run --frozen --no-dev alembic upgrade head
```

The timers keep pointing at the same path, so the next run uses the new code and nothing needs
re-enabling. Run `systemctl --user daemon-reload` only when a unit file itself changed (copy the
new `.service`/`.timer` first). Never edit files in the checkout: the next `git checkout` would
refuse to overwrite them.

## 11. Browsing the database from the Mac

The Postgres server is not reachable from a network, and that stays true. An SSH tunnel forwards
a local port to the server's Unix socket instead:

```bash
ssh -N -L 5433:/var/run/postgresql/.s.PGSQL.5432 shared-vps
```

`-N` opens no shell, and the tunnel exists only while that command runs. Then, in Beekeeper Studio
Community Edition (open source, the captain's chosen client):

1. New connection, PostgreSQL.
2. Host `localhost`, port `5433`, user `paperwatch`, database `paperwatch`, no password.
3. Tick the read-only mode: the role owns the data, so the client is the only guard against an
   accidental edit.

Useful starting points: `select status, health, counts, started_at from runs order by started_at
desc limit 5;` for recent runs, and `select tier, count(*) from items where status = 'admitted'
group by tier;` for the corpus. Nothing here is exposed beyond the tunnel, so closing it closes the
access.

## 12. Restoring a backup

The drill is M4, but the shape is fixed. A restore goes into a new, empty database, and creating a
`vector` database is a privilege this role deliberately does not have (see below), so the captain
creates it and the restore itself runs as `paperwatch`:

```bash
pg_restore --dbname=paperwatch_restore ~/backups/paperwatch-20261022T031500Z.dump
```

The custom format carries the schema, so the tables and indexes come back with the data. Never
restore into `paperwatch` itself while an ingestion is running.

## What must not run here

- `paperwatch eval`: it creates and drops its own throwaway database and needs the `vector`
  extension in it, which means `CREATEDB` and superuser. The role has neither, by design.
- `paperwatch serve` and `paperwatch ask`: the API is M3 and runs elsewhere. Ingestion needs no
  chat key, so none is installed here.
- Docker, and anything outside `/home/paperwatch/`.
