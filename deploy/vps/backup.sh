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

umask 077
mkdir -p "$backup_dir"

out=$backup_dir/paperwatch-$(date -u +%Y%m%dT%H%M%SZ).dump
# One fixed temporary name, so an interrupted run overwrites it rather than accumulating parts.
tmp=$backup_dir/.paperwatch.dump.part
# The EXIT trap removes the partial file however the script ends, and TERM, INT and HUP are turned
# into an exit so a `systemctl --user stop` or the unit's timeout runs it too: Debian's dash ends
# on SIGTERM without running the EXIT trap otherwise. SIGKILL cannot be trapped, and that is why
# the rename below matters more than the trap: the temporary name is hidden and fixed, so it
# matches no `paperwatch-*.dump` glob, and a dump only ever appears under its final name, complete.
trap 'rm -f "$tmp"' EXIT
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
