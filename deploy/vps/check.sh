#!/bin/sh
# Morning check for Paper Watch on the shared VPS: what the last ingestion run did, whether it is
# still running or left a killed run behind, how big the corpus is, whether last night's backup
# landed, and how much disk is left. Exits 0 when every check passes and 1 when any of them is
# bad, so it works as an ssh one-liner, in a journal, or in a `&&` chain. See docs/deploy/vps.md.
#
# Needs only psql and coreutils on Debian 13, and reads only the `paperwatch` database, the backup
# directory and the account's own systemd user manager: never another project's database or units.
#
#   PW_CHECK_DB                  database to read (paperwatch)
#   PW_CHECK_BACKUP_DIR          where the nightly dumps go ($HOME/backups)
#   PW_CHECK_MAX_RUN_AGE_S       limit on the age of the last run, from its start (90000, 25 h)
#   PW_CHECK_MAX_RUNNING_S       limit on how long a run may stay 'running' (10800, 3 h)
#   PW_CHECK_MAX_BACKUP_AGE_S    limit on the age of the newest dump (93600, 26 h)
#   PW_CHECK_MIN_FREE_GB         free disk the home must keep (2)
#
# A run that is waiting for its next follow-up pass is a finished row, not a 'running' one, so the
# stuck-run limit below never has to cover the 17 h the ingestion window may last: the follow-up
# lines report that window instead, from the run's own health record.
set -u

db=${PW_CHECK_DB:-paperwatch}
backup_dir=${PW_CHECK_BACKUP_DIR:-$HOME/backups}
# A daily run plus an hour of slack: the design's freshness SLO is 26 hours, but a limit of 25 h is
# what makes a missed 06:00 run visible by 07:00. It is measured from the run's start, so a long
# run does not delay the alarm, and an hour of duration still fits.
max_run_age_s=${PW_CHECK_MAX_RUN_AGE_S:-90000}
# This is the ingest unit's own TimeoutStartSec: a run still working at two hours is alive, one
# past three hours has been killed by systemd (and the failed-unit line below says so). The row
# stays 'running' until the next run closes it, so without this a dead morning would read as the
# previous green run. The follow-up window (06:00 to 22:00) does not raise this limit: the run
# finishes its row before it waits and rewrites it after every pass, so a waiting run is never
# 'running', and a row older than three hours is still a run killed before its main pass ended.
max_running_s=${PW_CHECK_MAX_RUNNING_S:-10800}
max_backup_age_s=${PW_CHECK_MAX_BACKUP_AGE_S:-93600}
# Below this the next dump, or the next Postgres write, is the thing that fills the disk.
min_free_gb=${PW_CHECK_MIN_FREE_GB:-2}

failed=0
fail() {
    printf 'FAIL  %s\n' "$*"
    failed=$((failed + 1))
}

# A WARN is a fact worth seeing that does not make the morning bad, such as a check that cannot
# run here at all. It is loud on purpose: silence would read as a pass.
warn() {
    printf 'WARN  %s\n' "$*"
}

# Every threshold is a whole number of seconds or of GB, and a typo must be a clean exit 1 with a
# message: in dash a non-numeric value in arithmetic ends the script with exit 2 and no reason.
number() {
    case $2 in
        '' | *[!0-9]*)
            printf "FAIL  %s must be a whole number, got '%s'\n" "$1" "$2"
            exit 1
            ;;
    esac
}
number PW_CHECK_MAX_RUN_AGE_S "$max_run_age_s"
number PW_CHECK_MAX_RUNNING_S "$max_running_s"
number PW_CHECK_MAX_BACKUP_AGE_S "$max_backup_age_s"
number PW_CHECK_MIN_FREE_GB "$min_free_gb"
min_free_kb=$(( min_free_gb * 1048576 ))

# Seconds as "2h05m"; a missing or unreadable value is reported as unknown rather than crashing
# the arithmetic.
human() {
    case $1 in
        '' | *[!0-9]*)
            printf 'unknown'
            return 0
            ;;
    esac
    printf '%dh%02dm' "$(( $1 / 3600 ))" "$(( ($1 % 3600) / 60 ))"
}

# One query, one `key|value` line per fact, so a missing run cannot shift the columns around.
# The values come from the columns the code writes: runs.status, runs.health (JSON, written as
# {"healthy": bool, "problems": [...], "warnings": [...], "followup": {...}}), runs.counts
# (JSON), runs.started_at/finished_at and items.status. A health record from before warnings or
# follow-ups existed has no such key, so it reads as no warnings and no follow-up. `open_age_s`
# is the oldest row still 'running', which is how a run killed before it could finish stays
# visible. The follow-up keys are all '(none)' until the loop writes them: `next_at` with
# `pass`/`of`/`pending` while it waits, `passes` and `stopped` once it ended.
read_run() {
    psql --no-psqlrc --quiet --tuples-only --no-align --field-separator='|' \
        --set=ON_ERROR_STOP=1 --dbname="$db" <<'SQL'
WITH last_run AS (
    SELECT status, health, counts, started_at, finished_at
    FROM runs
    WHERE finished_at IS NOT NULL
    ORDER BY finished_at DESC
    LIMIT 1
)
SELECT 'status', coalesce((SELECT status FROM last_run), '(none)')
UNION ALL
SELECT 'healthy', coalesce((SELECT health->>'healthy' FROM last_run), '(unknown)')
UNION ALL
SELECT 'problems', coalesce((SELECT health->>'problems' FROM last_run), '(unknown)')
UNION ALL
SELECT 'finished', coalesce(
    (SELECT to_char(finished_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') FROM last_run),
    '(none)')
UNION ALL
SELECT 'age_s', coalesce(
    (SELECT (extract(epoch FROM now() - started_at))::bigint::text FROM last_run), '(unknown)')
UNION ALL
SELECT 'took_s', coalesce(
    (SELECT (extract(epoch FROM finished_at - started_at))::bigint::text FROM last_run),
    '(unknown)')
UNION ALL
SELECT 'counts', coalesce((SELECT counts::text FROM last_run), '{}')
UNION ALL
-- One row per warning, so a warning is printed without parsing the JSON array in the shell.
SELECT 'warning', w
FROM last_run, json_array_elements_text(coalesce(health -> 'warnings', '[]'::json)) AS w
UNION ALL
-- Seconds until the next follow-up pass: positive means the run is waiting, negative means the
-- wait ran out (the process died inside the window). Only a waiting record has a `next_at`.
SELECT 'followup_next_in_s', coalesce(
    (SELECT (extract(epoch FROM (health -> 'followup' ->> 'next_at')::timestamptz - now()))
            ::bigint::text
     FROM last_run WHERE health -> 'followup' ->> 'next_at' IS NOT NULL), '(none)')
UNION ALL
SELECT 'followup_next_rome', coalesce(
    (SELECT to_char((health -> 'followup' ->> 'next_at')::timestamptz
                    AT TIME ZONE 'Europe/Rome', 'HH24:MI')
     FROM last_run WHERE health -> 'followup' ->> 'next_at' IS NOT NULL), '(none)')
UNION ALL
SELECT 'followup_pass', coalesce(
    (SELECT health -> 'followup' ->> 'pass' FROM last_run), '(none)')
UNION ALL
SELECT 'followup_of', coalesce(
    (SELECT health -> 'followup' ->> 'of' FROM last_run), '(none)')
UNION ALL
SELECT 'followup_pending', coalesce(
    (SELECT health -> 'followup' -> 'pending' ->> 's2' FROM last_run), '(none)')
UNION ALL
SELECT 'followup_passes', coalesce(
    (SELECT health -> 'followup' ->> 'passes' FROM last_run), '(none)')
UNION ALL
SELECT 'followup_stopped', coalesce(
    (SELECT health -> 'followup' ->> 'stopped' FROM last_run), '(none)')
UNION ALL
SELECT 'open_age_s', coalesce(
    (SELECT (extract(epoch FROM now() - min(started_at)))::bigint::text
     FROM runs WHERE status = 'running' AND finished_at IS NULL), '(none)')
UNION ALL
SELECT 'admitted', (SELECT count(*)::text FROM items WHERE status = 'admitted');
SQL
}

if ! rows=$(read_run); then
    printf 'FAIL  cannot read the database %s with psql\n' "$db"
    exit 1
fi

status='(none)'
healthy='(unknown)'
problems='(unknown)'
finished='(none)'
age_s='(unknown)'
took_s='(unknown)'
counts='{}'
warnings=''
open_age_s='(none)'
admitted='(unknown)'
followup_next_in_s='(none)'
followup_next_rome='(none)'
followup_pass='(none)'
followup_of='(none)'
followup_pending='(none)'
followup_passes='(none)'
followup_stopped='(none)'
while IFS='|' read -r key value; do
    case $key in
        status) status=$value ;;
        healthy) healthy=$value ;;
        problems) problems=$value ;;
        finished) finished=$value ;;
        age_s) age_s=$value ;;
        took_s) took_s=$value ;;
        counts) counts=$value ;;
        # Collected, not printed here: psql's row order is not the order a reader wants.
        warning) warnings="$warnings$value
" ;;
        open_age_s) open_age_s=$value ;;
        admitted) admitted=$value ;;
        followup_next_in_s) followup_next_in_s=$value ;;
        followup_next_rome) followup_next_rome=$value ;;
        followup_pass) followup_pass=$value ;;
        followup_of) followup_of=$value ;;
        followup_pending) followup_pending=$value ;;
        followup_passes) followup_passes=$value ;;
        followup_stopped) followup_stopped=$value ;;
    esac
done <<EOF
$rows
EOF

# Which of the three follow-up states this row is in. A waiting run is provisional by design (its
# verdict is unhealthy while Semantic Scholar refuses) and must count no failure; one whose wait
# ran out was killed inside the window; a finished loop already said how it ended.
followup_state=none
case $followup_next_in_s in
    '' | '(none)') : ;;
    -*) followup_state=overdue ;;
    *[!0-9]*) followup_state=unknown ;;
    *) followup_state=waiting ;;
esac

# The follow-up block comes first, because it decides whether this row's verdict is worth judging
# at all: a run waiting for its next pass finished its main pass and is unhealthy on purpose,
# while one whose wait ran out died inside the window. One event, one line, as with a killed run.
freshness=0
if [ "$status" = '(none)' ]; then
    fail 'no finished ingestion run in the database'
elif [ "$followup_state" = 'unknown' ]; then
    fail "cannot read the next follow-up pass from the health record: $followup_next_in_s"
elif [ "$status" = 'failed' ] && [ "$healthy" = '(unknown)' ]; then
    # Killed (SIGKILL, the memory cap) and closed by the next run, or crashed out of the
    # pipeline: both leave `failed` with no health verdict, and the health and freshness lines
    # would only say the same thing again. One event, one line.
    fail 'last run was killed or crashed and left no health verdict (journalctl --user -u paperwatch-ingest)'
elif [ "$followup_state" = 'waiting' ]; then
    # Not a failure and not a pass either: the loop is between passes and nothing is wrong yet.
    printf 'WAIT  follow-up %s of %s pending (s2: %s ids), next pass %s\n' \
        "$followup_pass" "$followup_of" "$followup_pending" "$followup_next_rome"
    freshness=1
elif [ "$followup_state" = 'overdue' ]; then
    # The row still says a pass was coming, and that moment has passed: the process was killed
    # (an OOM kill, a stop, `kill -9`) inside the window. The next run resumes the refused papers.
    fail 'follow-up did not complete (killed?)'
    freshness=1
else
    freshness=1
    case $status in
        succeeded | partial)
            # `partial` is a finished run: the verdict below, not the status, decides.
            printf 'ok    last run %s, finished %s, took %s (started %s ago)\n' \
                "$status" "$finished" "$(human "$took_s")" "$(human "$age_s")"
            ;;
        *)
            # failed or budget_exhausted.
            fail "last run $status, finished $finished"
            ;;
    esac
    if [ "$healthy" = 'true' ]; then
        printf 'ok    health: healthy\n'
    else
        fail "health: $healthy"
    fi
    case $problems in
        '' | '(unknown)' | '[]') : ;;
        *) fail "health problems: $problems" ;;
    esac
    if [ "$followup_passes" != '(none)' ]; then
        # A finished loop: how many passes it used and why it stopped (resolved, attempts,
        # time_of_day, or budget_exhausted).
        printf 'ok    follow-ups: %s used, %s\n' "$followup_passes" "$followup_stopped"
    fi
fi

if [ "$freshness" = 1 ]; then
    case $age_s in
        '' | *[!0-9]*)
            fail "cannot read the age of the last run: $age_s"
            ;;
        *)
            if [ "$age_s" -le "$max_run_age_s" ]; then
                printf 'ok    fresh: last run started %s ago (limit %s)\n' \
                    "$(human "$age_s")" "$(human "$max_run_age_s")"
            else
                fail "stale: last run started $(human "$age_s") ago (limit $(human "$max_run_age_s"))"
            fi
            ;;
    esac
fi

# Each warning is a fact from the verdict that does not make the morning bad, so it never becomes
# a FAIL. One per line, after the status and health lines it belongs to.
printf '%s' "$warnings" | while IFS= read -r warning; do
    [ -z "$warning" ] || warn "$warning"
done

case $open_age_s in
    '(none)')
        printf 'ok    no run left running\n'
        ;;
    '' | *[!0-9]*)
        fail "cannot read the age of the open run: $open_age_s"
        ;;
    *)
        if [ "$open_age_s" -gt "$max_running_s" ]; then
            fail "a run has been 'running' for $(human "$open_age_s") (limit $(human "$max_running_s")): killed or hung"
        else
            printf '      run: in progress (%s)\n' "$(human "$open_age_s")"
        fi
        ;;
esac

# The account's own user manager knows what the database cannot: whether the unit for the last run
# is failed (a startup crash, an OOM kill, a timeout, exit 1 or 3) and whether the timer that
# starts the next run is still active. Where no user manager answers (a bare cron job, a laptop)
# the state is reported as not checked rather than guessed.
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    if systemctl --user is-failed --quiet paperwatch-ingest.service; then
        fail 'paperwatch-ingest.service is in the failed state (systemctl --user status paperwatch-ingest)'
    else
        printf 'ok    unit: paperwatch-ingest.service is not failed\n'
    fi
    if systemctl --user is-active --quiet paperwatch-ingest.timer; then
        printf 'ok    timer: paperwatch-ingest.timer is active\n'
    else
        fail 'paperwatch-ingest.timer is not active: no morning run is scheduled'
    fi
else
    warn 'systemd: no user manager reachable, unit and timer state not checked'
fi

# Facts, not checks: how much the run did, and how big the corpus is now.
printf '      counts: %s\n' "$counts"
printf '      items: %s admitted\n' "$admitted"

if [ ! -d "$backup_dir" ]; then
    fail "no backup directory $backup_dir"
else
    # -t sorts by modification time, newest first, and the glob is expanded by the shell, so an
    # empty directory leaves ls with a path that does not exist and an empty result.
    # shellcheck disable=SC2012 # a backup name has no spaces or newlines, so ls is unambiguous
    dump=$(ls -1t "$backup_dir"/paperwatch-*.dump 2>/dev/null | head -n 1)
    if [ -z "$dump" ]; then
        fail "no backup in $backup_dir"
    else
        backup_bytes=$(wc -c < "$dump")
        if [ "$((backup_bytes))" -lt 1048576 ]; then
            backup_size=$(printf '%s KB' "$(( (backup_bytes + 1023) / 1024 ))")
        else
            backup_size=$(printf '%s MB' "$(( backup_bytes / 1048576 ))")
        fi
        backup_age=$(( $(date +%s) - $(date -r "$dump" +%s) ))
        if [ "$((backup_bytes))" -gt 0 ] && [ "$backup_age" -le "$max_backup_age_s" ]; then
            printf 'ok    backup: %s, %s ago (limit %s)\n' \
                "$backup_size" "$(human "$backup_age")" "$(human "$max_backup_age_s")"
        else
            fail "backup: $backup_size, $(human "$backup_age") ago (limit $(human "$max_backup_age_s"))"
        fi
    fi
fi

# df -P keeps one line per filesystem whatever the device name length; field 4 is available KB.
# shellcheck disable=SC2046 # df's columns are space separated; the split is the point
set -- $(df -P -k "$HOME" | tail -n 1)
if [ $# -lt 4 ]; then
    fail "cannot read the free space of $HOME"
else
    free_kb=$4
    if [ "$free_kb" -ge "$min_free_kb" ]; then
        printf 'ok    disk: %s GB free (limit %s GB)\n' \
            "$(( free_kb / 1048576 ))" "$min_free_gb"
    else
        fail "disk: $(( free_kb / 1048576 )) GB free (limit $min_free_gb GB)"
    fi
fi

if [ "$failed" -ne 0 ]; then
    printf 'check failed (%s problem(s))\n' "$failed"
    exit 1
fi
printf 'check passed\n'
