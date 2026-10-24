#!/bin/sh
# The API acceptance check: the gate, the limits and the units, from the server or a laptop.
#
#   api-check.sh BASE_URL
#
# Every check prints `ok` or `FAIL`; a systemd question that cannot be asked off the server prints
# `WARN`, like `check.sh`. The exit status is non-zero if anything printed `FAIL`. No token, header
# or body is ever printed.
#
# Credential: a literal PW_CHECK_TOKEN is sent on every authenticated check. An HMAC token is
# single use, so for the HMAC rehearsal set PW_CHECK_TOKEN_CMD to a command that prints a fresh
# one, for example:
#
#   PW_CHECK_TOKEN_CMD='uv run --frozen --no-dev --env-file ~/.config/paperwatch/api.env \
#     paperwatch api-token --live' ./api-check.sh http://127.0.0.1:8000
#
# A Vercel OIDC token is reusable, so PW_CHECK_TOKEN is enough for it.
set -u

BASE_URL=${1:-}
FAILS=0
TMP=${TMPDIR:-/tmp}/paperwatch-api-check.$$
BODY=$TMP.body
BODY_A=$TMP.a
BODY_B=$TMP.b
BODY_C=$TMP.c
TOKEN=
CURL=

cleanup() {
    rm -f "$BODY" "$BODY_A" "$BODY_B" "$BODY_C" "$TMP.body20k"
}
trap cleanup EXIT HUP INT TERM

ok() {
    printf 'ok: %s\n' "$1"
}

fail() {
    FAILS=$((FAILS + 1))
    printf 'FAIL: %s\n' "$1"
}

warn() {
    printf 'WARN: %s\n' "$1"
}

if [ -z "$BASE_URL" ]; then
    printf 'usage: %s BASE_URL\n' "$0" >&2
    exit 2
fi
case "$BASE_URL" in
    http://* | https://*) ;;
    *)
        printf 'FAIL: BASE_URL must start with http:// or https://\n' >&2
        exit 2
        ;;
esac
if ! command -v curl >/dev/null 2>&1; then
    printf 'FAIL: curl is required\n' >&2
    exit 2
fi
CURL=$(command -v curl)

# One token per authenticated check when a minting command is given.
new_token() {
    if [ -n "${PW_CHECK_TOKEN_CMD:-}" ]; then
        TOKEN=$(sh -c "$PW_CHECK_TOKEN_CMD" 2>/dev/null) || TOKEN=
    else
        TOKEN=${PW_CHECK_TOKEN:-}
    fi
    [ -n "$TOKEN" ]
}

# call METHOD PATH BODY CONTENT_TYPE AUTH ORIGIN HOST
# Prints the HTTP status and leaves the response body in $BODY. An empty field omits the header or
# the body.
call() {
    _method=$1
    _path=$2
    _body=$3
    _type=$4
    _auth=$5
    _origin=$6
    _host=$7
    set -- -sS -o "$BODY" -w '%{http_code}' --max-time 20 -X "$_method"
    if [ -n "$_type" ]; then
        set -- "$@" -H "Content-Type: $_type"
    fi
    if [ "$_auth" = "1" ]; then
        # Minted after the previous call, so an HMAC token is never reused.
        if ! new_token; then
            printf '000'
            return 0
        fi
        set -- "$@" -H "Authorization: Bearer $TOKEN"
    fi
    if [ -n "$_origin" ]; then
        set -- "$@" -H "Origin: $_origin"
    fi
    if [ -n "$_host" ]; then
        set -- "$@" -H "Host: $_host"
    fi
    if [ -n "$_body" ]; then
        set -- "$@" --data-binary "$_body"
    fi
    "$CURL" "$@" "$BASE_URL$_path" 2>/dev/null
}

expect() {
    # expect DESCRIPTION EXPECTED ACTUAL
    if [ "$2" = "$3" ]; then
        ok "$1"
    else
        fail "$1 (expected $2, got $3)"
    fi
}

# The liveness probe is the one route without a credential (the captain's slice 2 requirement).
expect "healthz answers without a token" 200 "$(call GET /healthz '' '' 0 '' '')"

# Every other route, and the docs, answer the same 401 without a credential, before anything is
# read.
CODE_DOCS=$(call GET /docs '' '' 0 '' '')
cp "$BODY" "$BODY_A"
CODE_OPENAPI=$(call GET /openapi.json '' '' 0 '' '')
cp "$BODY" "$BODY_B"
CODE_ASK=$(call POST /ask '{"question":"check"}' application/json 0 '' '')
cp "$BODY" "$BODY_C"
expect "docs need a credential" 401 "$CODE_DOCS"
expect "the OpenAPI document needs a credential" 401 "$CODE_OPENAPI"
expect "ask needs a credential" 401 "$CODE_ASK"
if cmp -s "$BODY_A" "$BODY_B" && cmp -s "$BODY_B" "$BODY_C"; then
    ok "every credential refusal has the same body"
else
    fail "credential refusals differ, so a caller can tell the reasons apart"
fi

# The authenticated checks. Without a token they cannot run; say so rather than pass silently.
if [ -z "${PW_CHECK_TOKEN:-}" ] && [ -z "${PW_CHECK_TOKEN_CMD:-}" ]; then
    warn "PW_CHECK_TOKEN and PW_CHECK_TOKEN_CMD are unset, so the authenticated checks are skipped"
else
    expect "readyz answers with a credential" 200 "$(call GET /readyz '' '' 1 '' '')"
    expect "status answers with a credential" 200 "$(call GET /status '' '' 1 '' '')"
    expect "an Origin header is refused" 403 \
        "$(call GET /status '' '' 1 'https://example.org' '')"
    expect "a non-JSON POST is refused" 415 \
        "$(call POST /ask 'x' text/plain 1 '' '')"
    BIG=$TMP.body20k
    head -c 20000 /dev/zero | tr '\0' 'x' >"$BIG"
    # A leading @ makes curl read the body from the file instead of sending its path.
    expect "a body over 16 KB is refused" 413 \
        "$(call POST /ask "@$BIG" application/json 1 '' '')"
    rm -f "$BIG"
    case "$BASE_URL" in
        *://127.0.0.1* | *://localhost* | *://\[::1\]*)
            expect "a foreign Host is refused on loopback" 400 \
                "$(call GET /status '' '' 1 '' 'evil.example')"
            ;;
        *)
            warn "BASE_URL is not loopback, so the Host allow-list check is skipped"
            ;;
    esac
fi

# The units, when this shell can ask a user manager (the server) and not on a laptop.
if ! systemctl --user show-environment >/dev/null 2>&1; then
    warn "systemctl --user is not available, so the unit checks are skipped"
else
    ACTIVE=$(systemctl --user is-active paperwatch-api.service 2>/dev/null || true)
    expect "paperwatch-api is active" active "$ACTIVE"
    MEMORY=$(systemctl --user show --property=MemoryMax --value paperwatch-api.service 2>/dev/null || true)
    case "$MEMORY" in
        700M | 734003200) ok "the API unit caps memory at 700M" ;;
        *) fail "the API unit MemoryMax is not 700M (got ${MEMORY:-nothing})" ;;
    esac
    SWAP=$(systemctl --user show --property=MemorySwapMax --value paperwatch-api.service 2>/dev/null || true)
    expect "the API unit forbids swap" 0 "$SWAP"
    OOM=$(systemctl --user show --property=OOMScoreAdjust --value paperwatch-api.service 2>/dev/null || true)
    expect "the API unit raises its OOM score" 900 "$OOM"
    SLICE=$(systemctl --user show --property=Slice --value paperwatch-api.service 2>/dev/null || true)
    expect "the API unit joins the shared slice" paperwatch.slice "$SLICE"
    INGEST_SLICE=$(systemctl --user show --property=Slice --value paperwatch-ingest.service 2>/dev/null || true)
    expect "the ingestion unit joins the shared slice" paperwatch.slice "$INGEST_SLICE"
fi

if [ "$FAILS" -ne 0 ]; then
    printf '%s check(s) failed\n' "$FAILS" >&2
    exit 1
fi
printf 'all checks passed\n'
