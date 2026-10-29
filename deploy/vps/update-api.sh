#!/bin/bash
# Update the API on this server to one named release tag: docs/deploy/vps.md section 11 as a single
# command. It is installed root-owned as /usr/local/sbin/paperwatch-update-api, and the routine
# update's login, `paperwatch`, may run that one command through sudo
# (deploy/vps/sudoers-paperwatch-update-api, ADR 0018).
#
#   update-api.sh paperwatch-vX.Y.Z
#
# Every repository step runs as the API's account, `paperwatch-api`, in its own checkout: the tags
# are fetched from that clone's existing origin (never a URL from here), the release's unit is
# compared with the installed one, the tag is checked out, `uv sync --frozen --no-dev` runs, then
# the unit is restarted, /healthz is waited for and the release's api-check.sh runs. A failure after
# the checkout moves puts the previous release back: checkout, sync, restart, and it exits non-zero.
#
# It never installs a unit. When the release's deploy/vps/paperwatch-api.service differs from
# /etc/systemd/system/paperwatch-api.service it stops before touching the checkout and says the
# owner has to install the unit by hand (section 11.1).
#
# An ingestion run does not hold this back and is not waited for: the API and the ingestion have no
# ordering and no relationship (ADR 0013, 0017), and a restart of the API neither stops nor delays
# a run. No secret is ever printed: the env file is read for PW_API_AUTH, by name, and for nothing
# else.
set -euo pipefail

# --- configuration -------------------------------------------------------------------------
# sudo runs this with env_reset's clean environment and the sudoers drop-in grants no SETENV, so a
# real run takes every path, account and URL from the lines below and none from the environment.
# The PW_UPDATE_* values exist for the unit tests, which run the real script against a temporary
# tree; they are only read when PW_UPDATE_UNPRIVILEGED=1, which sudo cannot pass either.
api=paperwatch-api
api_home=/home/$api
checkout=$api_home/paper-watch
api_env=$api_home/.config/paperwatch/api.env
uv_bin=$api_home/.local/bin/uv
system_unit=/etc/systemd/system/paperwatch-api.service
unit_in_release=deploy/vps/paperwatch-api.service
api_check=$checkout/deploy/vps/api-check.sh
api_url=http://127.0.0.1:8000
ready_timeout=300
poll=5
test_mode=0
if [ "${PW_UPDATE_UNPRIVILEGED:-0}" = 1 ]; then
    test_mode=1
    root=${PW_UPDATE_ROOT:-}
    api_home=$root/home/$api
    checkout=$api_home/paper-watch
    api_env=$api_home/.config/paperwatch/api.env
    uv_bin=$api_home/.local/bin/uv
    system_unit=$root/etc/systemd/system/paperwatch-api.service
    api_check=$checkout/deploy/vps/api-check.sh
    api_url=${PW_UPDATE_API_URL:-$api_url}
    ready_timeout=${PW_UPDATE_READY_TIMEOUT:-$ready_timeout}
    poll=${PW_UPDATE_POLL:-$poll}
fi
# The commands run as the API's account get this PATH and nothing else. Under sudo it is the system
# path; the tests point it at their stubs, which is the only reason it is ever different.
api_path=/usr/local/bin:/usr/bin:/bin
if [ "$test_mode" = 1 ]; then
    api_path=$PATH
fi

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

# The release's unit is revealed into a root-only file to be compared byte for byte. Global rather
# than local, so the trap still has a name for it after main returns.
released_unit=
cleanup() {
    [ -z "$released_unit" ] || rm -f "$released_unit"
}
trap cleanup EXIT

usage() {
    cat <<'USAGE'
usage: update-api.sh paperwatch-vX.Y.Z

Update the Paper Watch API to one release tag, from the API's own checkout and as its own account.
Runs as root from /usr/local/sbin/paperwatch-update-api, which the `paperwatch` login may run
through sudo; the release tag is the only thing it takes from the caller.

It fetches the tags from the checkout's existing origin, checks the release's unit against the
installed one, checks the tag out, syncs the venv, restarts paperwatch-api, waits for /healthz and
runs the release's api-check.sh. A failed update checks the previous release back out and restarts
it, then exits non-zero.

It stops before touching the checkout when the release's deploy/vps/paperwatch-api.service differs
from the installed /etc/systemd/system/paperwatch-api.service: install that unit by hand first
(docs/deploy/vps.md section 11.1).
USAGE
}

# --- running things as the API's account ---------------------------------------------------

# env -i on purpose: the account's own HOME and a fixed PATH, and no GIT_* variable from the caller
# can point git at another repository, another config or another transport.
git_as_api() {  # git arguments, run in the API's checkout
    runuser -u "$api" -- env -i "HOME=$api_home" "PATH=$api_path" git -C "$checkout" "$@"
}

# The API's shell is /usr/sbin/nologin, so a login shell cannot work: the directory and the
# environment are passed explicitly.
as_api_in() {  # directory command...
    local dir=$1 quoted
    shift
    quoted=$(printf '%q ' "$@")
    runuser -u "$api" -- env -i "HOME=$api_home" "PATH=$api_path" \
        bash -c "cd $(printf '%q' "$dir") && $quoted"
}

env_value() {
    awk -v key="$2" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "$1"
}

# --- the release this checkout serves ------------------------------------------------------

# The way back for a failed update, read before anything moves. A checkout past its release (the
# state a hand edit or a merged branch leaves) still goes back to the nearest release tag, which is
# what the server is meant to serve; a checkout with no release behind it at all has no way back and
# must stop here.
previous_release() {
    if git_as_api describe --tags --exact-match 2>/dev/null; then
        return 0
    fi
    git_as_api describe --tags --abbrev=0 --match 'paperwatch-v*' 2>/dev/null
}

# --- entry point ---------------------------------------------------------------------------

main() {
    if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
        usage
        exit 0
    fi
    if [ "$#" -ne 1 ]; then
        printf 'one release tag and nothing else, got %s argument(s)\n\n' "$#" >&2
        usage >&2
        exit 2
    fi
    local tag=$1
    # The same shape release-please publishes. It is checked here, not in the sudoers line: a sudo
    # command entry permits any argument, so the script is the boundary.
    if [[ ! $tag =~ ^paperwatch-v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf 'FAIL: %s is not a release tag: expected paperwatch-vX.Y.Z, digits only\n\n' \
            "$tag" >&2
        usage >&2
        exit 2
    fi

    if [ "$(id -u)" != 0 ] && [ "$test_mode" != 1 ]; then
        die "run as root: the API's checkout and its system unit are not this account's"
    fi
    if [ ! -d "$checkout/.git" ]; then
        die "$checkout is not a git checkout: install the API's checkout as section 15.1 shows"
    fi
    if [ ! -f "$system_unit" ]; then
        die "$system_unit is missing: install the unit as section 15.3 shows, then rerun"
    fi

    step "Fetching the tags into $checkout"
    git_as_api fetch --tags --quiet origin \
        || die "fetching from the checkout's origin failed: check its network and credentials"
    if ! git_as_api rev-parse --verify --quiet "refs/tags/$tag^{commit}" >/dev/null; then
        die "origin has no release tag $tag: check the release name, then rerun"
    fi

    local previous
    if ! previous=$(previous_release); then
        die "cannot tell which release $checkout serves, so a failed update has no previous release to go back to: check it out by hand, then rerun"
    fi

    step "Checking $tag's unit against the installed one"
    released_unit=$(mktemp "${TMPDIR:-/tmp}/paperwatch-update-api.XXXXXX") \
        || die "cannot create a temporary file for the release's unit"
    if ! git_as_api show "$tag:$unit_in_release" >"$released_unit" 2>/dev/null; then
        die "$tag has no $unit_in_release, so it is not a Paper Watch release: nothing was changed"
    fi
    if ! cmp -s "$released_unit" "$system_unit"; then
        die "$tag's $unit_in_release differs from $system_unit, and installing a unit is the owner's step, never this script's: install it by hand as docs/deploy/vps.md section 11.1 shows, then rerun. Nothing was changed."
    fi
    ok "$system_unit is the unit $tag publishes"

    step "Checking out $tag and syncing its venv"
    git_as_api checkout --quiet "$tag" || die "checking out $tag failed: nothing was restarted"
    if [ "$(git_as_api describe --tags --exact-match 2>/dev/null)" != "$tag" ]; then
        die "$checkout is not at $tag after the checkout: nothing was restarted"
    fi
    as_api_in "$checkout" "$uv_bin" sync --frozen --no-dev \
        || die "uv sync failed for $tag: the API is still running $previous"
    ok "the venv is synced for $tag"

    die "$tag is checked out and synced, but this build stops before the restart"
}

main "$@"
