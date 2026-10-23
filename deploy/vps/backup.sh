#!/bin/sh
# Nightly logical backup of the `paperwatch` database, run by paperwatch-backup.timer.
#
# -Fc is Postgres' own custom format: compressed, and restored with pg_restore into a fresh
# database. Peer authentication over the local socket means no password and no network, so the
# same command works interactively. The dump is the whole corpus and every stored row, so the
# directory is 0700 and the files 0600.
#
#   PW_BACKUP_DIR         destination directory ($HOME/backups)
#   PW_BACKUP_KEEP_DAYS   dumps to keep, one per night (7)
#   PW_BACKUP_DB          database to dump (paperwatch)
set -eu

backup_dir=${PW_BACKUP_DIR:-$HOME/backups}
# `-`, not `:-`: an explicitly empty PW_BACKUP_KEEP_DAYS (a unit `Environment=` line, say) is a
# configuration mistake worth reporting, not a silent seven.
keep_days=${PW_BACKUP_KEEP_DAYS-7}
db=${PW_BACKUP_DB:-paperwatch}

# A typo must be a clean exit 1 with a message, not a shell arithmetic error that ends the script
# with exit 2 and no reason (the shape check.sh's number() has). A leading zero goes with the
# typos because dash reads it as octal: `010` would keep eight nights, and `08` would end the
# script with "Illegal number" *after* the dump was written but before retention ran.
case $keep_days in
    '' | *[!0-9]* | 0?*)
        printf "PW_BACKUP_KEEP_DAYS must be a whole number without leading zeros, got '%s'\n" \
            "$keep_days" >&2
        exit 1
        ;;
esac
# Four digits is far past any sane retention and well inside every shell's integer range: testing a
# longer number with -lt is itself what fails in dash with "Illegal number".
if [ "${#keep_days}" -gt 4 ] || [ "$keep_days" -lt 1 ]; then
    printf 'PW_BACKUP_KEEP_DAYS must be between 1 and 9999, got %s\n' "$keep_days" >&2
    exit 1
fi

umask 077
mkdir -p "$backup_dir"

out=$backup_dir/paperwatch-$(date -u +%Y%m%dT%H%M%SZ).dump
# One fixed temporary name, so an interrupted run overwrites it rather than accumulating parts.
tmp=$backup_dir/.paperwatch.dump.part

# One non-blocking lock, because that fixed temporary name is only safe while one backup runs at a
# time: a manual run during the timer's run would otherwise have both writing and renaming
# `.paperwatch.dump.part`. flock is held by the kernel for the life of this shell's descriptor 9,
# so any exit releases it, SIGKILL included, and the lock file itself can stay. Where there is no
# flock (macOS, a minimal install) a directory holding this run's PID is the lock, and the EXIT
# trap has to remove it; a lock whose process is gone is stale and cleared instead of blocking
# every later night.
lock=$backup_dir/.lock
if command -v flock >/dev/null 2>&1; then
    # dash ends a non-interactive shell when a redirection on a special builtin fails (`:` and
    # `exec` are), `if` or not, so the lock file is opened by a probe in a subshell that can fail
    # cleanly first (a directory at the lock path, an unwritable backup directory).
    if [ -d "$lock" ] || ! ( : >>"$lock" ) 2>/dev/null; then
        printf 'cannot open the lock file %s\n' "$lock" >&2
        exit 1
    fi
    exec 9>"$lock"
    if ! flock -n 9; then
        printf 'another backup is running (lock %s is held)\n' "$lock" >&2
        exit 1
    fi
    trap 'rm -f "$tmp"' EXIT
else
    if ! mkdir "$lock" 2>/dev/null; then
        # No pid file yet means a run is between mkdir and the write below, or a crash left the
        # directory behind: only a pid the kernel no longer has makes the lock safe to clear.
        if [ -f "$lock/pid" ] && ! kill -0 "$(cat "$lock/pid")" 2>/dev/null; then
            printf 'clearing the lock a killed run left at %s\n' "$lock" >&2
            rm -f "$lock/pid"
            rmdir "$lock" 2>/dev/null
        fi
        if ! mkdir "$lock" 2>/dev/null; then
            # A path that still exists after the stale check is another run's lock; one that does
            # not means mkdir failed for a different reason (a read-only or full filesystem).
            if [ -e "$lock" ]; then
                printf 'another backup is running (lock %s is held): remove it if no backup is running\n' \
                    "$lock" >&2
            else
                printf 'cannot create the lock %s\n' "$lock" >&2
            fi
            exit 1
        fi
    fi
    printf '%s\n' "$$" >"$lock/pid"
    trap 'rm -f "$tmp"; rm -f "$lock/pid"; rmdir "$lock" 2>/dev/null' EXIT
fi
# The order above is the point: the EXIT trap is armed only once the lock is won, so a run that
# loses the race exits without deleting the winner's `.paperwatch.dump.part`. TERM, INT and HUP
# are turned into an exit so a `systemctl --user stop` or the unit's timeout runs the trap too:
# Debian's dash ends on SIGTERM without running it otherwise. SIGKILL cannot be trapped, and that
# is why the rename below matters more than the trap: the temporary name is hidden and fixed, so
# it matches no `paperwatch-*.dump` glob, and a dump only ever appears under its final name,
# complete.
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

pg_dump --format=custom --no-password --dbname="$db" --file="$tmp"
if [ ! -s "$tmp" ]; then
    printf 'pg_dump wrote an empty file\n' >&2
    rm -f "$tmp"
    exit 1
fi
mv "$tmp" "$out"

# -mtime counts whole days, so +6 removes anything older than six days: today plus six.
find "$backup_dir" -maxdepth 1 -type f -name 'paperwatch-*.dump' \
    -mtime +"$((keep_days - 1))" -delete

printf 'backup: %s (%s bytes)\n' "$out" "$(wc -c < "$out")"
