#!/bin/sh
# Morning check for Paper Watch on the shared VPS: what the last ingestion run did, how big the
# corpus is, whether last night's backup landed, and how much disk is left. Exits 0 when every
# check passes and 1 when any of them is bad, so it works as an ssh one-liner, in a journal, or
# in a `&&` chain. See docs/deploy/vps.md.
#
# Needs only psql and coreutils on Debian 13, and reads only the `paperwatch` database and the
# backup directory: never the scraper's database, never another service.
#
#   PW_CHECK_DB                  database to read (paperwatch)
#   PW_CHECK_BACKUP_DIR          where the nightly dumps go ($HOME/backups)
#   PW_CHECK_MAX_RUN_AGE_S       limit on the age of the last finished run (93600, 26 h)
#   PW_CHECK_MAX_BACKUP_AGE_S    limit on the age of the newest dump (93600, 26 h)
#   PW_CHECK_MIN_FREE_GB         free disk the home must keep (2)
set -u

db=${PW_CHECK_DB:-paperwatch}
backup_dir=${PW_CHECK_BACKUP_DIR:-$HOME/backups}
# A daily run plus slack: the design's freshness SLO is 26 hours, and a run that has not finished
# by then means a missed morning, however healthy the run before it was.
max_run_age_s=${PW_CHECK_MAX_RUN_AGE_S:-93600}
max_backup_age_s=${PW_CHECK_MAX_BACKUP_AGE_S:-93600}
# Below this the next dump, or the next Postgres write, is the thing that fills the disk.
min_free_gb=${PW_CHECK_MIN_FREE_GB:-2}
min_free_kb=$(( min_free_gb * 1048576 ))

failed=0
fail() {
    printf 'FAIL  %s\n' "$*"
    failed=$((failed + 1))
}

# Seconds as "2h05m"; a missing or unreadable value is reported as unknown rather than crashing
# the arithmetic, which in sh would end the script with the wrong exit status.
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
# {"healthy": bool, "problems": [...]}), runs.counts (JSON) and items.status.
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
    (SELECT (extract(epoch FROM now() - finished_at))::bigint::text FROM last_run), '(unknown)')
UNION ALL
SELECT 'took_s', coalesce(
    (SELECT (extract(epoch FROM finished_at - started_at))::bigint::text FROM last_run),
    '(unknown)')
UNION ALL
SELECT 'counts', coalesce((SELECT counts::text FROM last_run), '{}')
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
admitted='(unknown)'
while IFS='|' read -r key value; do
    case $key in
        status) status=$value ;;
        healthy) healthy=$value ;;
        problems) problems=$value ;;
        finished) finished=$value ;;
        age_s) age_s=$value ;;
        took_s) took_s=$value ;;
        counts) counts=$value ;;
        admitted) admitted=$value ;;
    esac
done <<EOF
$rows
EOF

if [ "$status" = '(none)' ]; then
    fail 'no finished ingestion run in the database'
else
    case $status in
        succeeded | partial)
            printf 'ok    last run %s, finished %s, %s ago, took %s\n' \
                "$status" "$finished" "$(human "$age_s")" "$(human "$took_s")"
            ;;
        *)
            # failed, budget_exhausted or a row left 'running' by a killed process.
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
    case $age_s in
        '' | *[!0-9]*)
            fail "cannot read the age of the last run: $age_s"
            ;;
        *)
            if [ "$age_s" -le "$max_run_age_s" ]; then
                printf 'ok    fresh: last run %s ago (limit %s)\n' \
                    "$(human "$age_s")" "$(human "$max_run_age_s")"
            else
                fail "stale: last run $(human "$age_s") ago (limit $(human "$max_run_age_s"))"
            fi
            ;;
    esac
fi

# Facts, not checks: how much the run did, and how big the corpus is now.
printf '      counts: %s\n' "$counts"
printf '      items: %s admitted\n' "$admitted"

if [ ! -d "$backup_dir" ]; then
    fail "no backup directory $backup_dir"
else
    # -t sorts by modification time, newest first; the glob is expanded by the shell, so an empty
    # directory leaves ls with a path that does not exist and an empty result. No GNU-only
    # find -printf, so the same script runs under review on a developer's Mac.
    dump=$(ls -1t "$backup_dir"/paperwatch-*.dump 2>/dev/null | head -n 1)
    if [ -z "$dump" ]; then
        fail "no backup in $backup_dir"
    else
        backup_bytes=$(wc -c < "$dump")
        backup_mb=$(( backup_bytes / 1048576 ))
        backup_age=$(( $(date +%s) - $(date -r "$dump" +%s) ))
        if [ "$((backup_bytes))" -gt 0 ] && [ "$backup_age" -le "$max_backup_age_s" ]; then
            printf 'ok    backup: %s MB, %s ago (limit %s)\n' \
                "$backup_mb" "$(human "$backup_age")" "$(human "$max_backup_age_s")"
        else
            fail "backup: $backup_mb MB, $(human "$backup_age") ago (limit $(human "$max_backup_age_s"))"
        fi
    fi
fi

# df -P keeps one line per filesystem whatever the device name length; field 4 is available KB.
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
    printf 'check failed (%s problems)\n' "$failed"
    exit 1
fi
printf 'check passed\n'
