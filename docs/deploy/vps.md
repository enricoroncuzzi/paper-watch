# Running Paper Watch's ingestion on the shared VPS

The corpus has to accumulate every day, whatever the captain is building, so ingestion runs on
the shared VPS ahead of M3 ([ADR 0009](../adr/0009-ingestion-runs-on-the-shared-vps.md)). Two
systemd user services with their timers, both as the `paperwatch` account: the daily ingestion at
06:00 Europe/Rome, and a nightly `pg_dump` backup at 05:15. The API, the weekly
shortlist, email alerts and the restore drill are not part of this deployment (M3 and M4).

Everything here is a step for the `paperwatch` user. There is no root step: the database, the
role, pgvector, the swap file and linger were set up by the captain before this guide starts.

## Server facts (verified 2026-10-22 over SSH as `paperwatch`)

- Debian 13, 2 CPU, 1.9 GB RAM (about 1.3 GB available), 2 GB swap (swappiness 10), 48 GB disk
  free, clock in Europe/Berlin (the same UTC offset as Rome all year), NTP on.
- Shared with another production project using about 650 MB, with its own services and its own
  database on the same Postgres server.
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

## 2. Clone the repository at a release tag

```bash
git clone https://github.com/enricoroncuzzi/paper-watch ~/paper-watch
cd ~/paper-watch
git checkout paperwatch-v0.6.0        # a release tag; use the latest, tags are paperwatch-vX.Y.Z
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
PW_USER_AGENT="paper-watch (+https://github.com/enricoroncuzzi/paper-watch; you@example.com)"
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
psql -d paperwatch -Atc 'show idle_session_timeout'                   # must print 0
command -v psql pg_dump ~/.local/bin/uv
```

`idle_in_transaction_session_timeout` must be `0`: the run holds a transaction open around slow
source calls, and a server-side timeout would kill it mid-run. `idle_session_timeout` must be `0`
too: the single-run advisory lock lives on a connection that sits idle between follow-up passes,
and a server-side idle timeout would close it and silently drop the lock. If either is not 0,
stop and tell the captain, because changing them is a root step.

## 5. Create the schema

```bash
uv run --frozen --no-dev --env-file ~/.config/paperwatch/env alembic upgrade head
psql -d paperwatch -c '\dt'      # items, tier_history, signal_snapshots, runs, dead_letters,
                                  # source_cursors, chat_calls
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
  `uv run --frozen --no-dev paperwatch ingest --live --followup` from `/home/paperwatch/paper-watch`
  with the environment file above. `MemoryMax=950M` with `MemorySwapMax=0` keeps a runaway run from
  touching the machine: with no swap the cgroup OOM killer stops it, systemd marks the unit failed,
  the run row stays `running` until the next run closes it as failed, and the morning check reports
  both. That is deliberate: a visible failed morning, never a slow one that starves the other
  project in the shared swap. The cap needs the memory controller delegated to the user manager,
  which Debian's `user@.service` does. The run also has low CPU and I/O priority (`Nice=10`,
  best-effort priority 7) and a `TimeoutStartSec=17h` ceiling. `--followup` may keep the process
  alive from 06:00 until 22:00 Rome time retrying Semantic Scholar refusals, and the loop also
  caps its own window at 16 h from the start, so no wait can outlive the timeout whatever the start
  time. The timeout must stay above that 16 h cap: it is the last resort, never the normal end.
  Because it is 17 h, a hung main pass is no longer killed at 3 h: the morning check names it
  `hung (still running)` and the action is `systemctl --user stop paperwatch-ingest`.
- `paperwatch-backup.service` runs `deploy/vps/backup.sh`: `pg_dump -Fc` into
  `~/backups/paperwatch-<UTC stamp>.dump`, keeping seven nights. The dump is written under a
  hidden fixed temporary name and renamed only once it is complete, so even a SIGKILL, which no
  trap can catch, cannot leave a short file that looks like a backup; TERM, INT and HUP are
  trapped too, so a stop or the 1 h timeout also removes the partial file. The script creates the
  directory 0700 and the files 0600 itself. It also takes a non-blocking lock (`~/backups/.lock`)
  so a manual run during the timer's dump exits 1 with `another backup is running` instead of
  sharing the one temporary file, and it refuses a `PW_BACKUP_KEEP_DAYS` that is not a whole
  number from 1 to 9999.
- The backup starts 45 minutes before the ingestion, so it usually finishes first; that is the
  clock, not a guarantee, and an overlap is harmless because `pg_dump` reads an MVCC snapshot.

Both units also set `NoNewPrivileges=yes`, so nothing they run can gain privilege through a setuid
or setgid binary: the ingestion and `pg_dump` both work as the account and never need one. To see
where a unit stands, run `systemd-analyze --user security paperwatch-ingest` and read its
`Overall exposure level` line: lower is better, and the points left on it are the knobs this
deployment leaves at their defaults.

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
This measured run deliberately has no `--followup`: it must end with the main pass, so the peak
and the duration are the main pass's. The scheduled unit adds the flag (and waits for follow-ups
only when Semantic Scholar refused something).

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
as stuck after three hours (its own limit, `PW_CHECK_MAX_RUNNING_S`; the unit's start timeout is
17 h).

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

It prints the last finished run's status, health verdict, problems and warnings, the run's counts
and duration, whether a run is still in progress or left `running` (a kill or a hang), the unit and
timer state of the account's own user manager, the total admitted items, the newest backup's age
and size, and free disk. Exit code 0 means every check passed, 1 means at least one is bad, and the
bad lines start with `FAIL`:

- no finished run, or a status other than `succeeded` or `partial`;
- the last run was killed or crashed and left no health verdict, which is what a killed run's
  row looks like once the next run has closed it: one `FAIL` for the one event;
- an unhealthy or missing health verdict, or a non-empty list of problems;
- the last run started more than 25 hours ago, so a missed 06:00 morning is caught by 07:00;
- a run left `running` for more than three hours. A run waiting for a follow-up pass is finished,
  not `running`, so this limit does not need the 17 h timeout; a row older than three hours is a run
  killed or hung before its main pass ended. When the unit is still alive the check says
  `hung (still running)` and names `systemctl --user stop paperwatch-ingest`, because with a 17 h
  timeout systemd will no longer kill it; otherwise it says `killed or hung`. "Alive" is the unit's
  own `ActiveState`: a running oneshot shows as `activating`, not `active`;
- a follow-up pass that never came: while a run waits, the check prints
  `WAIT  next follow-up pass N of 40 at HH:MM (s2: M ids pending)` and counts no failure (the run
  is deliberately unhealthy until S2 answers). For the minutes a pass itself takes, its due moment
  has just passed and the row is not rewritten yet, so up to 20 minutes past it prints
  `WAIT  follow-up pass N of 40 running (s2: M ids)`, as long as systemd still holds the unit. Past
  that it prints one `FAIL` for the one event: `follow-up did not complete (hung: the unit is still
  running)` with the stop command when the process is alive, or `(killed?)` when it is gone, and
  the next morning retries the same papers. A row whose follow-up fields cannot be read at all is
  one named `FAIL`, not a psql error that hides the rest of the morning;
- `paperwatch-ingest.service` failed (a startup crash, an OOM kill, a timeout, exit 1 or 3), or
  `paperwatch-ingest.timer` not active, so no morning run is scheduled;
- no `~/backups/` directory, no dump in it, or the newest older than 26 hours or empty;
- less than 2 GB free in the home.

The unit and timer lines are skipped with a `WARN` where no user manager answers, so the same
script works from a cron job or a laptop without pretending it checked them. A warning from the
run's health verdict is also printed as a `WARN` and never as a `FAIL`: the first time a lab feed
fails, the cursor stays pinned to the missed window, so the next run retries the same days and
nothing is lost. Only the same feed failing again on the next run is a real outage, and that run is
degraded and fails the check. Semantic Scholar's refusals work the same way: a day whose
follow-up passes were all refused is one warning, and only a second day in a row is a failure.
Thresholds can be overridden for one invocation: `PW_CHECK_MAX_RUN_AGE_S`,
`PW_CHECK_MAX_RUNNING_S`, `PW_CHECK_FOLLOWUP_GRACE_S`, `PW_CHECK_MAX_BACKUP_AGE_S`,
`PW_CHECK_MIN_FREE_GB`, `PW_CHECK_BACKUP_DIR`, `PW_CHECK_DB`. It needs only `psql` and coreutils,
and reads only the `paperwatch` database, the backup directory and the account's own user manager.

### A morning with follow-ups

The unit runs `paperwatch ingest --live --followup`, so when Semantic Scholar refuses papers the
process stays alive instead of leaving them a day behind. The 06:00 run finishes its main pass and
writes its row, then waits 5, 10, 20 and 40 minutes and every 30 minutes after that, retrying only
the refused papers, until nothing is refused, 40 passes have run, the next pass would start at or
after 22:00 Rome time, or 16 hours have passed since the start (ADR 0008). Nothing but Semantic
Scholar is asked again: no arXiv harvest, no lab feeds, no Hugging Face, no classification.

What the captain sees:

- the journal gets one `followup_pass` line per pass (`pass_`, `pending`, `waited_s`, `rss_mb`) and
  one `followup_finished` line (`passes`, `stopped`, `pending`) at the end, next to the S2 client's
  own refusal lines. `rss_mb` is what the idle process really holds: the model is released before
  each wait, so an idle window costs little memory on the shared machine;
- `check.sh` prints the `WAIT` line above between passes, `ok    follow-ups: n used, resolved` (or
  `attempts`, `time_of_day`, `budget_exhausted`) once the loop is over, and the verdict as usual.
  A recovered morning exits 0 and leaves no failed unit;
- a manual `paperwatch ingest --live` run in that window prints
  `skipped: another ingestion run is active` and exits 0: the follow-up loop holds the single-run
  lock for as long as it waits, by design;
- a main pass that hangs is not killed by the timeout any more (it is 17 h, for the window). The
  check reports `hung (still running)` after three hours and the captain stops it with
  `systemctl --user stop paperwatch-ingest`: the stop is SIGTERM, which the command turns into an
  orderly exit, so the run closes its own row as failed (`exit 143`) instead of leaving it
  `running` for the next morning;
- the run row's `health.followup` has the same facts as the journal (`next_at` while waiting,
  `passes` and `stopped` after), and `counts.followup_passes` says how many passes it used.

A run's counts normally show `"s2_unchecked": 1` (observed 2026-10-23): one arXiv paper in the
corpus is not in Semantic Scholar's index at all, so it is left unanswered and asked again next
run. That is expected and does not make a run unhealthy; a much larger count comes with a
degraded `s2` source and its own `FAIL`. `"s2_refused"` is normally 0: a non-zero count means
Semantic Scholar refused (throttled) those papers, not that it does not know them, and with
`--followup` it reaches 0 as soon as a pass is answered.

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
| 0 | The run finished and its health verdict is healthy (a warning is printed and does not change the exit code). | Nothing. |
| 3 | The run finished (`succeeded` or `partial`) but judged itself unhealthy: a source returned nothing, a cursor did not advance, a signal source was degraded, an item failed. The problems are printed to stderr. | Read the problems; the corpus is intact but a source needs a look. |
| 1 | The run did not finish its work (a failure, or the token budget was exhausted). | Read the journal; the next run resumes from the stored cursors. |
| 0 | Another ingestion held the single-run lock, so this invocation did nothing (`skipped`). | Nothing. |
| 2 | A usage error, for example a missing `--live`. | Fix the command. |

The exit code is the final verdict's: a day that started unhealthy because Semantic Scholar refused
and recovered in a follow-up pass exits 0, and the window itself is not a failure (a run killed
inside it is caught by the next morning's check).

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
new `.service`/`.timer` first), and copy it when it changed: the release that added the follow-up
passes changed `paperwatch-ingest.service` (`ExecStart` gained `--followup`, `TimeoutStartSec` rose
from 3 h to 17 h), so updating to it means copying both files and running `daemon-reload` before
the next 06:00. Never edit files in the checkout: the next `git checkout` would refuse to
overwrite them. Do not migrate while an ingestion or a dump is in progress (a follow-up window
counts: `check.sh` says `WAIT` while one is open).

## 12. Browsing the database from the Mac

The Postgres server is not reachable from a network, and that stays true. An SSH tunnel forwards
a local port to the server's Unix socket instead:

```bash
ssh -N -L 127.0.0.1:5433:/var/run/postgresql/.s.PGSQL.5432 shared-vps
```

`-N` opens no shell, and the tunnel exists only while that command runs: the forward names
`127.0.0.1` explicitly, so the local port answers only on the Mac and not on whatever network it
is on. Close it when you are done (Ctrl-C in that terminal), and nothing outside that window can
reach the database. Then, in Beekeeper Studio Community Edition (open source, the captain's
chosen client):

1. New connection, PostgreSQL.
2. Host `localhost`, port `5433`, user `paperwatch`, database `paperwatch`, no password.
3. Leave SSL off: the tunnel is already encrypted, and the server refuses TLS on a Unix socket.
4. If the connection form offers a read-only or safe mode, turn it on. The role owns the data, so
   nothing else stops an accidental edit; check it by trying `create table zz (i int)` and expecting
   a refusal.

Once the server admin has created the read-only role `paperwatch_ro` (SELECT only), use it as the
user in step 2: it is the recommended connection for Beekeeper, and the safe mode above becomes a
second guard rather than the only one. The M3 API and agent are not read-only: the daily chat cap
records one row per call in `chat_calls`, so a SELECT-only role fails the first answer. Give them
their own role instead - `paperwatch_api` in the grants below, name it as you like - which reads
every table and writes only those rows. `paperwatch` owns the tables and runs the migrations, so
the default privileges granted `FOR ROLE paperwatch` are what keeps tables added by later
migrations readable:

```sql
-- as the admin, in database paperwatch
GRANT USAGE ON SCHEMA public TO paperwatch_ro, paperwatch_api;

-- Beekeeper: every table, no writes.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO paperwatch_ro;

-- The API and agent: every table, plus the one row per chat call the daily cap counts. The id
-- comes from the table's own sequence, created by the migration that adds `chat_calls`.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO paperwatch_api;
GRANT INSERT ON chat_calls TO paperwatch_api;
GRANT USAGE ON SEQUENCE chat_calls_id_seq TO paperwatch_api;

-- Tables later migrations create. A later table the API must write needs its own INSERT grant.
ALTER DEFAULT PRIVILEGES FOR ROLE paperwatch IN SCHEMA public
  GRANT SELECT ON TABLES TO paperwatch_ro, paperwatch_api;
```

The admin also has to let the three roles in without a password, and without loosening anything for
the other project's database. All three steps are the admin's, and both files live in the
cluster's data directory:

1. One dedicated `pg_hba.conf` line, for the `paperwatch` database only, placed above the generic
   `local all all peer` line: `pg_hba.conf` uses the first line that matches, so the order is what
   makes it apply. The role field lists all three roles because the second field is the database
   and the third is the role the client asks for, so a line naming only `paperwatch` would match
   the owner alone and send `paperwatch_ro` and `paperwatch_api` to the generic line, where they
   fail. Never put `map=` on `local all all peer`: that line also covers the other project's
   database.

   ```
   # TYPE   DATABASE    USER                                     METHOD
   local    paperwatch  paperwatch,paperwatch_ro,paperwatch_api  peer  map=paperwatch
   ```

2. The identity map in `pg_ident.conf`, three lines under the same name. `peer` compares the
   accessing Unix account with the requested role unless a map is given, and a map accepts only the
   system-user/role pairs it lists, so these lines are what restrict access to the Unix account
   `paperwatch`. Leaving out the first line locks out the ingestion itself, which connects as
   `paperwatch`:

   ```
   # MAPNAME     SYSTEM-USERNAME  PG-USERNAME
   paperwatch    paperwatch       paperwatch
   paperwatch    paperwatch       paperwatch_ro
   paperwatch    paperwatch       paperwatch_api
   ```

   A password for each role is the alternative, and then the Beekeeper connection stops being
   passwordless.

3. Reload Postgres so it re-reads both files (no restart needed), then check that the owner still
   gets in before you test the two new roles: a mistake in either file would otherwise lock the
   ingestion out of the database it owns.

   ```bash
   # as the admin
   psql -c 'select pg_reload_conf()'   # or: pg_ctl reload
   # as paperwatch, which must still see the database
   psql -d paperwatch -c 'select 1'
   ```

   Only once that prints `1`, try the two roles through the tunnel.

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
