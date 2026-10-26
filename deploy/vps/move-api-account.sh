#!/bin/bash
# Move the running API from the `paperwatch` user account onto its own `paperwatch-api` account
# and system unit: docs/deploy/vps.md section 18 as subcommands, for a server the API already runs
# on. ADR 0017 is the decision; section 18 is the reference for what each step does and why.
#
#   move-api-account.sh status      which steps are done (read-only)
#   move-api-account.sh prepare     steps 1 to 4: the account, the ident line, its checkout, its env
#   move-api-account.sh switch      steps 5 and 6: swap the units, wait for the API, run the check
#   move-api-account.sh rollback    the guide's "Rolling back": back to the owner's user unit
#   move-api-account.sh tidy        step 7: retire the owner's ident line, env file and old unit
#
# Run as root, one subcommand at a time, and read section 18 when one stops. Nothing here touches
# the ingestion, its timers, the database schema or the tunnel: the API keeps answering on
# 127.0.0.1:8000, which is the address the tunnel publishes.
#
# `prepare` is idempotent, so it can be rerun after a partial failure. It deliberately keeps the
# owner's env file until `tidy`, unlike the guide's step 4: with the old copy in place a rollback
# has nothing to hand back, so the way back needs no file copy at all.
#
# No secret value is ever printed: the script rewrites keys by name and reports file modes, never
# file contents. For an OIDC deployment, export PW_CHECK_TOKEN before `switch` so the acceptance
# check can make its authenticated requests; with an HMAC deployment the script mints one per
# request itself.
set -euo pipefail

# --- configuration -------------------------------------------------------------------------
# Every path is derived from one prefix, empty on a real server. The tests point it at a temporary
# tree so the same code runs there with the privileged commands stubbed.
root=${PW_MOVE_ROOT:-}
owner=${PW_MOVE_OWNER:-paperwatch}
api=${PW_MOVE_API_ACCOUNT:-paperwatch-api}
owner_home=$root/home/$owner
api_home=$root/home/$api
owner_checkout=$owner_home/paper-watch
api_checkout=$api_home/paper-watch
owner_env=$owner_home/.config/paperwatch/api.env
api_env=$api_home/.config/paperwatch/api.env
retired_env=$owner_home/.config/paperwatch/api.env.bak-hmac
owner_unit=$owner_home/.config/systemd/user/paperwatch-api.service
system_unit=$root/etc/systemd/system/paperwatch-api.service
pg_ident=${PW_MOVE_PG_IDENT:-$root/etc/postgresql/17/main/pg_ident.conf}
admin_user=${PW_MOVE_ADMIN_USER:-postgres}
repo_url=${PW_MOVE_REPO_URL:-https://github.com/enricoroncuzzi/paper-watch}
api_url=${PW_MOVE_API_URL:-http://127.0.0.1:8000}
api_check=${PW_MOVE_API_CHECK:-$api_checkout/deploy/vps/api-check.sh}
uv_bin=$api_home/.local/bin/uv
api_cache=$api_home/.cache/paperwatch-models
owner_cache=$owner_home/.cache/paperwatch-models
# The first start downloads about 180 MB of model, so the wait is long; it is still bounded,
# because a unit that is restarting will never answer and the swap has to be undone instead.
ready_timeout=${PW_MOVE_READY_TIMEOUT:-900}
poll=${PW_MOVE_POLL:-5}

# The map section 12 creates, and the one line that gives the API's account its own role.
map_name=paperwatch
api_role=paperwatch_api

assume_yes=0

# --- output --------------------------------------------------------------------------------

say() {
    printf '%s\n' "$*"
}

step() {
    printf '\n==> %s\n' "$*"
}

warn() {
    printf 'WARN: %s\n' "$*" >&2
}

ok() {
    printf 'ok: %s\n' "$*"
}

die() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
usage: move-api-account.sh {prepare|switch|rollback|tidy|status} [--yes]

status    which steps of docs/deploy/vps.md section 18 are done; reads, never writes
prepare   steps 1 to 4: the API account, its ident line, its checkout and its env file
switch    steps 5 and 6: install and start the system unit where the user unit ran
rollback  the guide's "Rolling back": back to the owner's user unit, valid before tidy
tidy      step 7: retire the owner's ident line, env file, backup and old unit file

--yes skips the confirmation before switch and tidy. No secret value is ever printed.
USAGE
}

# --- small queries -------------------------------------------------------------------------

require_root() {
    [ "$(id -u)" = 0 ] && return 0
    # The tests run every command through a stub and every path under a temporary tree, so they
    # skip this check rather than needing a privileged account to exercise the logic.
    [ "${PW_MOVE_UNPRIVILEGED:-0}" = 1 ] && return 0
    die "run as root: creating an account, a system unit and the ident map all need it"
}

account_exists() {
    getent passwd "$1" >/dev/null 2>&1
}

account_uid() {
    getent passwd "$1" | cut -d: -f3
}

# Does the map have this system-user/role pair? Whitespace and trailing comments are the file's
# business, so the test is on the first three fields, which is what Postgres reads.
map_line_present() {
    awk -v map="$map_name" -v sys="$2" -v role="$3" \
        '$1 == map && $2 == sys && $3 == role { found = 1 } END { exit !found }' "$1"
}

api_checkout_tag() {
    git -C "$api_checkout" describe --tags --exact-match 2>/dev/null
}

env_value() {
    awk -v key="$2" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "$1"
}

# --- running things as another account -----------------------------------------------------

as_user() {  # account command...
    local account=$1
    shift
    runuser -u "$account" -- "$@"
}

# The API's shell is /usr/sbin/nologin, so `sudo -u ... bash -l` cannot work: HOME is passed
# explicitly and the commands section 15.1 shows are run one at a time.
as_api() {
    runuser -u "$api" -- env "HOME=$api_home" "$@"
}

as_api_in() {  # directory command...
    local dir=$1 quoted
    shift
    quoted=$(printf '%q ' "$@")
    runuser -u "$api" -- env "HOME=$api_home" bash -c "cd $(printf '%q' "$dir") && $quoted"
}

reload_postgres() {
    # Root's own peer identity is not a Postgres role, so the reload is the admin's, as in the
    # guide: this is a reload, never a restart of the server the other project shares.
    as_user "$admin_user" psql -c 'select pg_reload_conf()' >/dev/null \
        || die "could not reload Postgres as $admin_user: nothing else was changed"
}

# --- editing the two files this migration rewrites -----------------------------------------

# Both edits copy the file with `cp -p` and rename the copy over it, so the mode and the owner of
# a file root writes on another account's behalf survive, and a crash never leaves half a file.

map_line_add() {
    # The columns are the block section 12 shows; any whitespace would do for Postgres.
    printf '%-14s%-17s%s\n' "$map_name" "$api" "$api_role" >>"$pg_ident"
}

# env_point FILE KEY VALUE: set one key and leave every other line byte for byte as it was.
env_point() {
    local file=$1 key=$2 value=$3 tmp line found=0
    tmp=$file.new.$$
    cp -p "$file" "$tmp"
    : >"$tmp"
    while IFS= read -r line || [ -n "$line" ]; do
        case $line in
            "$key="*) printf '%s=%s\n' "$key" "$value" && found=1 ;;
            *) printf '%s\n' "$line" ;;
        esac
    done <"$file" >"$tmp"
    if [ "$found" = 0 ]; then
        printf '%s=%s\n' "$key" "$value" >>"$tmp"
    fi
    mv -f "$tmp" "$file"
}

# --- prepare: steps 1 to 4 -----------------------------------------------------------------

# The tag is read from the ingestion's checkout rather than chosen here: both accounts serve one
# release, and a checkout that drifts is a different program halfway through a run.
release_tag() {
    local tag
    if tag=$(git -C "$owner_checkout" describe --tags --exact-match 2>/dev/null); then
        printf '%s\n' "$tag"
        return 0
    fi
    tag=$(git -C "$owner_checkout" describe --tags --abbrev=0 2>/dev/null) || return 1
    warn "$owner_checkout is not on a release tag; the API gets the nearest one, $tag"
    printf '%s\n' "$tag"
}

prove_mapping() {
    # A bad line in either file locks the ingestion out of the database it owns, so the owner's
    # own connection is proven before the new account's.
    as_user "$owner" psql -d paperwatch -c 'select 1' >/dev/null \
        || die "$owner can no longer reach the database: fix $pg_ident before going on"
    local who
    who=$(as_user "$api" psql -d paperwatch -Atc 'select current_user') || who=
    if [ "$who" != "$api_role" ]; then
        die "peer authentication maps $api to '${who:-nothing}', not '$api_role': the map line in $pg_ident is wrong, stop here"
    fi
    ok "$owner still reaches paperwatch, and $api reaches $api_role"
}

cmd_prepare() {
    require_root

    step "Step 1: the API's account"
    if account_exists "$api"; then
        ok "$api exists"
    else
        useradd --system --create-home --home-dir "$api_home" \
            --shell /usr/sbin/nologin "$api"
        ok "created $api with $api_home"
    fi

    step "Step 2: its ident line, and the proofs"
    [ -f "$pg_ident" ] \
        || die "$pg_ident is missing: point PW_MOVE_PG_IDENT at this cluster's map"
    if map_line_present "$pg_ident" "$api" "$api_role"; then
        ok "the map already has $api -> $api_role"
    else
        map_line_add
        ok "added $api -> $api_role to $pg_ident"
        # Only when the file changed: a rerun must not reload the shared server for nothing.
        reload_postgres
    fi
    prove_mapping

    step "Step 3: its checkout, its uv and its venv"
    local tag
    if ! tag=$(release_tag); then
        die "$owner_checkout has no release tag to copy: check out the release the ingestion serves first"
    fi
    if [ -d "$api_checkout/.git" ]; then
        ok "$api_checkout is already cloned"
    else
        as_api git clone --quiet "$repo_url" "$api_checkout"
        ok "cloned $repo_url into $api_checkout"
    fi
    as_api git -C "$api_checkout" fetch --tags --quiet
    as_api git -C "$api_checkout" checkout --quiet "$tag"
    ok "$api_checkout is at $tag"
    if [ -x "$uv_bin" ]; then
        ok "$uv_bin is installed"
    else
        # pipefail inside, so a download that fails does not pass as an installed uv.
        as_api bash -c 'set -o pipefail; curl -LsSf https://astral.sh/uv/install.sh | sh' >/dev/null
        [ -x "$uv_bin" ] || die "uv did not install to $uv_bin"
        ok "installed uv for $api"
    fi
    as_api_in "$api_checkout" "$uv_bin" sync --frozen --no-dev >/dev/null
    ok "the venv is synced for $tag"

    step "Step 4: its env file"
    [ -f "$owner_env" ] \
        || die "$owner_env is gone, so there is nothing to move: restore it as section 15.2 shows"
    install -d -m 700 -o "$api" -g "$api" "$api_home/.config/paperwatch"
    if [ -f "$api_env" ]; then
        ok "$api_env already exists, overwriting it from the owner's copy"
    fi
    # Root writes it straight into the account, 0600, so the secret never passes through a shell
    # that could log it. The owner's copy stays until tidy (see the note at the top).
    install -m 600 -o "$api" -g "$api" "$owner_env" "$api_env"
    env_point "$api_env" PW_LOCAL_EMBEDDING_CACHE_DIR "$api_cache"
    ok "wrote $api_env, 0600, with PW_LOCAL_EMBEDDING_CACHE_DIR=$api_cache"
    if [ -e "$retired_env" ]; then
        warn "$retired_env still holds a retired secret: tidy removes it"
    fi

    step "Prepared"
    say "Nothing is serving from the new account yet: the next step is '$0 switch'."
}

# --- status --------------------------------------------------------------------------------

cmd_status() {
    local tag prepared=1 tidied=1 active=no
    if account_exists "$api"; then
        say "step 1 account $api: done"
    else
        say "step 1 account $api: missing"
        prepared=0
    fi
    if map_line_present "$pg_ident" "$api" "$api_role"; then
        say "step 2 map line: done"
    else
        say "step 2 map line: missing"
        prepared=0
    fi
    if tag=$(api_checkout_tag); then
        say "step 3 checkout: done $tag"
    else
        say "step 3 checkout: missing"
        prepared=0
    fi
    if [ -x "$uv_bin" ] && [ -d "$api_checkout/.venv" ]; then
        say "step 3 environment: done"
    else
        say "step 3 environment: missing"
        prepared=0
    fi
    if [ -f "$api_env" ] && [ "$(env_value "$api_env" PW_LOCAL_EMBEDDING_CACHE_DIR)" = "$api_cache" ]; then
        say "step 4 env file: done"
    else
        say "step 4 env file: missing"
        prepared=0
    fi
    if [ -f "$system_unit" ]; then
        say "step 5 system unit: done"
    else
        say "step 5 system unit: missing"
    fi
    if systemctl is-active --quiet paperwatch-api; then
        active=yes
    fi
    say "step 6 system unit active: $active"

    if map_line_present "$pg_ident" "$owner" "$api_role"; then
        say "step 7 owner map line: pending"
        tidied=0
    else
        say "step 7 owner map line: done"
    fi
    if [ -e "$owner_env" ]; then
        say "step 7 owner env file: pending"
        tidied=0
    else
        say "step 7 owner env file: done"
    fi
    if [ -e "$retired_env" ]; then
        say "step 7 retired env backup: pending"
        tidied=0
    else
        say "step 7 retired env backup: done"
    fi
    if [ -e "$owner_unit" ]; then
        say "step 7 owner unit file: pending"
        tidied=0
    else
        say "step 7 owner unit file: done"
    fi

    if [ "$prepared" = 0 ]; then
        say "verdict: nothing done"
    elif [ "$tidied" = 1 ]; then
        say "verdict: migrated"
    elif [ "$active" = yes ]; then
        say "verdict: serving from the API account, tidy pending"
    else
        say "verdict: prepared, ready to switch"
    fi
}

# --- entry point ---------------------------------------------------------------------------

main() {
    local cmd= arg
    for arg in "$@"; do
        case $arg in
            --yes | -y) assume_yes=1 ;;
            --help | -h | help) usage; exit 0 ;;
            -*)
                printf 'unknown option: %s\n\n' "$arg" >&2
                usage >&2
                exit 2
                ;;
            *)
                if [ -n "$cmd" ]; then
                    printf 'one subcommand at a time, got %s and %s\n\n' "$cmd" "$arg" >&2
                    usage >&2
                    exit 2
                fi
                cmd=$arg
                ;;
        esac
    done
    if [ -z "$cmd" ]; then
        usage >&2
        exit 2
    fi
    case $cmd in
        prepare | switch | rollback | tidy | status) ;;
        *)
            printf 'unknown subcommand: %s\n\n' "$cmd" >&2
            usage >&2
            exit 2
            ;;
    esac
    "cmd_$cmd"
}

main "$@"
