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
keep_days=${PW_BACKUP_KEEP_DAYS:-7}
db=${PW_BACKUP_DB:-paperwatch}

# A typo must be a clean exit 1 with a message, not a shell arithmetic error that ends the script
# with exit 2 and no reason (the shape check.sh's number() has). Zero is refused with the rest:
# it is not a retention policy, and `find -mtime +-1` is not a valid expression.
case $keep_days in
    '' | *[!0-9]*)
        printf "PW_BACKUP_KEEP_DAYS must be a whole number, got '%s'\n" "$keep_days" >&2
        exit 1
        ;;
esac
if [ "$keep_days" -lt 1 ]; then
    printf 'PW_BACKUP_KEEP_DAYS must be at least 1, got %s\n' "$keep_days" >&2
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
# flock (macOS, a minimal install) a directory is the lock, and the EXIT trap has to remove it; a
# run that cannot (SIGKILL) leaves it behind, and the message below says how to clear it.
lock=$backup_dir/.lock
if command -v flock >/dev/null 2>&1; then
    if ! exec 9>"$lock"; then
        printf 'cannot open the lock file %s\n' "$lock" >&2
        exit 1
    fi
    if ! flock -n 9; then
        printf 'another backup is running (lock %s is held)\n' "$lock" >&2
        exit 1
    fi
    trap 'rm -f "$tmp"' EXIT
else
    if ! mkdir "$lock" 2>/dev/null; then
        printf 'another backup is running, or a killed run left %s: remove it if no backup is running\n' \
            "$lock" >&2
        exit 1
    fi
    trap 'rm -f "$tmp"; rmdir "$lock" 2>/dev/null' EXIT
fi
# TERM, INT and HUP are turned into an exit so a `systemctl --user stop` or the unit's timeout runs
# the EXIT trap too: Debian's dash ends on SIGTERM without running it otherwise. SIGKILL cannot be
# trapped, and that is why the rename below matters more than the trap: the temporary name is
# hidden and fixed, so it matches no `paperwatch-*.dump` glob, and a dump only ever appears under
# its final name, complete.
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
