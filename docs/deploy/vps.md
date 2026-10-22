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
- Shared with another production project using about 650 MB, with its own services and Postgres.
- Postgres 17.11. Database `paperwatch`, owned by role `paperwatch`, with `vector` 0.8.0 enabled.
  Peer authentication over the local Unix socket: no password, no TCP.
- Linger is on for `paperwatch`, so the user's systemd manager keeps running while nobody is
  logged in. `loginctl show-user paperwatch -p Linger` prints `Linger=yes`.
- Python 3.13 and git are installed. `uv` is not, and lives in the user's home (step 1). The
  project pins `>=3.12,<3.13`, so the first `uv sync` downloads its own managed Python 3.12 into
  `~/.local/share/uv/python`: no root, nothing to install.
- The SSH alias is `shared-vps`.

## Boundaries (binding on every change here)

- Everything lives in `/home/paperwatch/` and runs as `paperwatch`.
- Never read, stop, restart or modify the other project's services, database or home directory.
  The two deployments share a machine and a Postgres server, not a schema.
- No changes to networking, firewall, SSH, VPN/proxy, system packages or the Postgres server
  configuration. A root step is the captain's decision, never a shortcut.
- No Docker here: the units run the checkout directly.

## 1. Install uv

```bash
ssh shared-vps
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs into ~/.local/bin, no root
~/.local/bin/uv --version
```

The installer puts `uv` in `~/.local/bin`, which a login shell adds to `PATH` from `~/.profile`.
Open a new session, or `. ~/.local/bin/env`, before step 3 relies on the bare `uv` name; the
units always use the absolute path.

## 2. Clone the repository at the release that contains `deploy/`

```bash
git clone https://github.com/enricoroncuzzi/paper-watch ~/paper-watch
cd ~/paper-watch
git checkout paperwatch-v0.4.0        # the first release with deploy/vps; tags are paperwatch-vX.Y.Z
```

Pin a tag, not `main`: a checkout that changes under a running ingestion is a different program
halfway through, and only a release tag is known to contain the units and both scripts. `deploy/vps/`
keeps the scripts executable through git, so a clone needs no `chmod`.

## 3. Create the environment

```bash
uv sync --frozen --no-dev
```

`--frozen` installs exactly the committed `uv.lock` and never rewrites it; `--no-dev` leaves out
the test, lint and type-check tools, which this server never runs. The first sync also downloads
the managed Python 3.12 described above.

## 4. Write the environment file

```bash
mkdir -p ~/.config/paperwatch
umask 077
cat > ~/.config/paperwatch/env <<'EOF'
PW_DATABASE_URL=postgresql+psycopg://paperwatch@/paperwatch?host=/var/run/postgresql
PW_USER_AGENT="paper-watch/0.4 (+https://github.com/enricoroncuzzi/paper-watch; you@example.com)"
PW_LOCAL_EMBEDDING_CACHE_DIR=/home/paperwatch/.cache/paperwatch-models
EOF
chmod 600 ~/.config/paperwatch/env
```

- No host and no port: `?host=/var/run/postgresql` is the socket directory, so psycopg connects
  over the socket and peer authentication identifies the role by the Unix user. There is no
  password anywhere, and the URL cannot silently fall back to TCP.
- The user agent is quoted on purpose: it holds spaces, parentheses and a semicolon, and without
  the quotes `uv run --env-file` stops reading at that line and silently drops every line after
  it. Quoted, systemd, bash, dash and `uv --env-file` all read the same three variables. Replace
  the address: arXiv asks bulk harvesters for a contact.
- The model cache stays outside the checkout, so switching to a new release does not throw away
  the 180 MB embedding model.
- No chat key: ingestion never calls a chat model.

Nothing reads this file implicitly: the units load it through `EnvironmentFile=`, and an
interactive command has to pass `--env-file ~/.config/paperwatch/env`. Check that it parses and
points at the socket:

```bash
uv run --frozen --no-dev --env-file ~/.config/paperwatch/env python -c \
  'from paperwatch.config import get_settings; s = get_settings(); print(s.database_url, s.user_agent)'
psql -d paperwatch -c 'select current_user, current_database()'
psql -d paperwatch -Atc 'show idle_in_transaction_session_timeout'    # must print 0
command -v psql pg_dump ~/.local/bin/uv
```

`idle_in_transaction_session_timeout` must be `0`: the run holds a transaction open around slow
source calls, and a server-side timeout would kill it mid-run. If it is not 0, stop and tell the
captain, because changing it is a root step.

## 5. Create the schema

```bash
uv run --frozen --no-dev --env-file ~/.config/paperwatch/env alembic upgrade head
psql -d paperwatch -c '\dt'      # items, tier_history, runs, dead_letters, source_cursors, chat_calls
```

The role owns the database, so no grant is needed. Every later release runs the same command
again; migrations are additive one version at a time.

## 6. Install the units

```bash
mkdir -p ~/.config/systemd/user
cp ~/paper-watch/deploy/vps/paperwatch-ingest.service \
   ~/paper-watch/deploy/vps/paperwatch-ingest.timer \
   ~/paper-watch/deploy/vps/paperwatch-backup.service \
   ~/paper-watch/deploy/vps/paperwatch-backup.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload
```

Nothing is enabled yet: the timers are switched on in step 9, after the measured first run and a
dry run of the morning check, so an untuned cap never meets a 06:00 trigger.

What the units do, and why:

- `paperwatch-ingest.service` runs
  `uv run --frozen --no-dev paperwatch ingest --live` from `/home/paperwatch/paper-watch` with
  the environment file above. `MemoryMax=950M` with `MemorySwapMax=0` keeps a runaway run from
  touching the machine: with no swap the cgroup OOM killer stops it, systemd marks the unit failed,
  the run row stays `running` until the next run closes it as failed, and the morning check reports
  both. That is deliberate: a visible failed morning, never a slow one that starves the other
  project in the shared swap. The cap needs the memory controller delegated to the user manager,
  which Debian's `user@.service` does. The run also has low CPU and I/O priority (`Nice=10`,
  best-effort priority 7) and a `TimeoutStartSec=3h` ceiling.
- `paperwatch-backup.service` runs `deploy/vps/backup.sh`: `pg_dump -Fc` into
  `~/backups/paperwatch-<UTC stamp>.dump`, keeping seven nights. The dump is written under a
  hidden fixed temporary name and renamed only once it is complete, so even a SIGKILL, which no
  trap can catch, cannot leave a short file that looks like a backup; TERM, INT and HUP are
  trapped too, so a stop or the 1 h timeout also removes the partial file. The script creates the
  directory 0700 and the files 0600 itself.
- The backup starts 45 minutes before the ingestion, so it usually finishes first; that is the
  clock, not a guarantee, and an overlap is harmless because `pg_dump` reads an MVCC snapshot.

## 7. The first run, measured

Start it outside the other project's busy hours, roughly 07:00 to 15:00 Rome time: before 06:30,
or after 15:00. The first run downloads the embedding model (about 180 MB) and covers a wider
window than a daily one, so it is both the heaviest and the only one whose peak you cannot guess.

There is no `/usr/bin/time` to rely on (the `time` package is not in a minimal Debian install and
installing it is a root step), so run the real command in a transient unit with a cap of 1200M
and no swap: the run that has never been measured is exactly the one that must not be allowed to
swap.

```bash
systemd-run --user --wait --unit=paperwatch-measure \
  --property=EnvironmentFile=/home/paperwatch/.config/paperwatch/env \
  --property=WorkingDirectory=/home/paperwatch/paper-watch \
  --property=MemoryMax=1200M --property=MemorySwapMax=0 \
  /home/paperwatch/.local/bin/uv run --frozen --no-dev paperwatch ingest --live
```

`--wait` blocks this session until the run ends, so run it inside `tmux` or accept a dropped
SSH: the transient unit is independent of the client and keeps running. When it ends,
`systemd-run` prints the numbers to copy, including `Service runtime:` and a `Memory peak:` line
with the swap peak; the journal has the run's own output and a `Consumed ... memory peak` line.

```bash
journalctl --user -u paperwatch-measure -n 40 --no-pager
```

If the run is killed instead (the line reads `oom-kill`), the peak is above what this machine can
give: stop and tell the captain, rather than raising the cap. A successful `--wait` run removes
the transient unit, so `systemctl show` on it afterwards prints nothing useful; a failed attempt
stays loaded, and the next try needs `systemctl --user reset-failed paperwatch-measure` first.

Then set the cap in a drop-in instead of editing the installed unit, so a later `git checkout` of
a new release does not silently reset it. Use the measured peak plus about a third, and never less
than the model's own 520 MB. The installed unit already carries `950M`, the repo's default from the
706.8M peak measured on this machine on 2026-10-22, so this step is only needed when your own peak
differs from that measurement, or you want the value pinned against a future default:

```bash
mkdir -p ~/.config/systemd/user/paperwatch-ingest.service.d
cat > ~/.config/systemd/user/paperwatch-ingest.service.d/memory.conf <<'EOF'
[Service]
MemoryMax=950M
EOF
systemctl --user daemon-reload
```

Write the measured number down with the deployment notes, so a later release moves the repo
default only from a real measurement. Then run once through the real unit:

```bash
systemctl --user start --no-block paperwatch-ingest.service    # --no-block: a oneshot start blocks
journalctl --user -fu paperwatch-ingest
systemctl --user show paperwatch-ingest.service -p MemoryMax -p MemorySwapMax -p Result
```

If that run exits 1 or 3, read the journal before going on: 3 means it finished and judged the
day unhealthy, 1 means it did not finish its work. Once you have judged the result acceptable,
run `systemctl --user reset-failed paperwatch-ingest.service`, because a failed unit is reported
by the morning check until the next run starts. A row left `running` by a kill is closed as
failed by the next run, and the check reports it once as such; until then the check reports it
as stuck after three hours, the unit's own start timeout.

Take one backup by hand now, so `pg_dump` is proven on this database before a timer relies on it
and the morning check has a dump to read. It needs no environment file: the defaults are this
home's `backups/` and the database `paperwatch`, reached over the local socket as the `paperwatch`
role.

```bash
~/paper-watch/deploy/vps/backup.sh
```

It creates `~/backups/` (0700) and one `paperwatch-<stamp>.dump` (0600), and prints the dump's
size.

## 8. The morning check

```bash
~/paper-watch/deploy/vps/check.sh
```

It prints the last finished run's status, health verdict and problems, the run's counts and
duration, whether a run is still in progress or left `running` (a kill or a hang), the unit and
timer state of the account's own user manager, the total admitted items, the newest backup's age
and size, and free disk. Exit code 0 means every check passed, 1 means at least one is bad, and the
bad lines start with `FAIL`:

- no finished run, or a status other than `succeeded` or `partial`;
- the last run was killed or crashed and left no health verdict, which is what a killed run's
  row looks like once the next run has closed it: one `FAIL` for the one event;
- an unhealthy or missing health verdict, or a non-empty list of problems;
- the last run started more than 25 hours ago, so a missed 06:00 morning is caught by 07:00;
- a run left `running` for more than three hours, matching the unit's `TimeoutStartSec`, so a live
  run is never called dead while systemd would still let it finish;
- `paperwatch-ingest.service` failed (a startup crash, an OOM kill, a timeout, exit 1 or 3), or
  `paperwatch-ingest.timer` not active, so no morning run is scheduled;
- no `~/backups/` directory, no dump in it, or the newest older than 26 hours or empty;
- less than 2 GB free in the home.

The unit and timer lines are skipped with a `WARN` where no user manager answers, so the same
script works from a cron job or a laptop without pretending it checked them. Thresholds can be
overridden for one invocation:
`PW_CHECK_MAX_RUN_AGE_S`, `PW_CHECK_MAX_RUNNING_S`, `PW_CHECK_MAX_BACKUP_AGE_S`,
`PW_CHECK_MIN_FREE_GB`, `PW_CHECK_BACKUP_DIR`, `PW_CHECK_DB`. It needs only `psql` and coreutils,
and reads only the `paperwatch` database, the backup directory and the account's own user manager.

A run's counts normally show `"s2_unchecked": 1`: one arXiv paper in the corpus is not in
Semantic Scholar's index at all, so it is left unanswered and asked again next run. That is
expected and does not make a run unhealthy; a much larger count comes with a degraded `s2` source
and its own `FAIL`.

At this point the timer is not enabled yet, so this dry run reports one `FAIL` for the timer.
If you skipped the backup in step 7 it also fails on the backup, and before the first backup that
is a missing directory: `no backup directory /home/paperwatch/backups`. With `~/backups/` there
but empty, or emptied later, it reads `no backup in /home/paperwatch/backups` instead. If the run
in step 7 exited 1 or 3 and you have not reset the unit, it reports that too. One `FAIL` for the
timer, and nothing else, is the expected result; step 9 removes it.

## 9. Enable the timers

```bash
systemctl --user enable --now paperwatch-backup.timer paperwatch-ingest.timer
systemctl --user list-timers 'paperwatch-*'
~/paper-watch/deploy/vps/check.sh        # check passed, once no failed unit is left
```

`list-timers` shows the next trigger for each: the backup at 05:15, the ingestion at 06:00, both
in Rome time. `Persistent=true` means a trigger missed while the machine was down fires once when
it can, instead of being skipped for the day.

## 10. Exit codes and the journal

`paperwatch ingest --live` tells systemd exactly what happened:

| Exit | Meaning | What to do |
| --- | --- | --- |
| 0 | The run finished and its health verdict is healthy. | Nothing. |
| 3 | The run finished (`succeeded` or `partial`) but judged itself unhealthy: a source returned nothing, a cursor did not advance, a signal source was degraded, an item failed. The problems are printed to stderr. | Read the problems; the corpus is intact but a source needs a look. |
| 1 | The run did not finish its work (a failure, or the token budget was exhausted). | Read the journal; the next run resumes from the stored cursors. |
| 0 | Another ingestion held the single-run lock, so this invocation did nothing (`skipped`). | Nothing. |
| 2 | A usage error, for example a missing `--live`. | Fix the command. |

systemd records 1 and 3 as a failed unit, which is deliberate: both deserve a look, and the morning
check reports the failed unit too. The problems themselves are in
`journalctl --user -u paperwatch-ingest`.

## 11. Updating to a new release

```bash
cd ~/paper-watch
git fetch --tags
git checkout paperwatch-vX.Y.Z
uv sync --frozen --no-dev
uv run --frozen --no-dev --env-file ~/.config/paperwatch/env alembic upgrade head
```

The timers keep pointing at the same path, so the next run uses the new code and nothing needs
re-enabling. Run `systemctl --user daemon-reload` only when a unit file itself changed (copy the
new `.service`/`.timer` first). Never edit files in the checkout: the next `git checkout` would
refuse to overwrite them. Do not migrate while an ingestion or a dump is in progress.

## 12. Browsing the database from the Mac

The Postgres server is not reachable from a network, and that stays true. An SSH tunnel forwards
a local port to the server's Unix socket instead:

```bash
ssh -N -L 5433:/var/run/postgresql/.s.PGSQL.5432 shared-vps
```

`-N` opens no shell, and the tunnel exists only while that command runs. Then, in Beekeeper Studio
Community Edition (open source, the captain's chosen client):

1. New connection, PostgreSQL.
2. Host `localhost`, port `5433`, user `paperwatch`, database `paperwatch`, no password.
3. Leave SSL off: the tunnel is already encrypted, and the server refuses TLS on a Unix socket.
4. If the connection form offers a read-only or safe mode, turn it on. The role owns the data, so
   nothing else stops an accidental edit; check it by trying `create table zz (i int)` and expecting
   a refusal.

Useful starting points: `select status, health, counts, started_at from runs order by started_at
desc limit 5;` for recent runs, and `select tier, count(*) from items where status = 'admitted'
group by tier;` for the corpus. Nothing here is exposed beyond the tunnel, so closing it closes the
access.

## 13. Restoring a backup

The drill is M4, but the shape is fixed. A restore goes into a new database that the captain
creates, owned by `paperwatch` and with the `vector` extension in it: creating a database or an
extension is a privilege this role deliberately does not have.

```bash
# the captain, as the admin:
createdb -O paperwatch paperwatch_restore
psql -d paperwatch_restore -c 'create extension vector'

# then as paperwatch:
pg_restore --no-comments --exit-on-error --dbname=paperwatch_restore \
  ~/backups/paperwatch-20261022T031501Z.dump
```

`--no-comments` because `COMMENT ON EXTENSION vector` is refused for a role that does not own the
extension, and that one error would make an otherwise good restore exit 1; `--exit-on-error`
because a restore that stops halfway must not look finished. Never restore into `paperwatch` itself
while an ingestion is running.

## 14. What must not run here

- `paperwatch eval`: it creates and drops its own throwaway database and needs the `vector`
  extension in it, which means `CREATEDB` and superuser. The role has neither, by design.
- `paperwatch serve` and `paperwatch ask`: the API is M3 and runs elsewhere. Ingestion needs no
  chat key, so none is installed here.
- Docker, and anything outside `/home/paperwatch/`.
