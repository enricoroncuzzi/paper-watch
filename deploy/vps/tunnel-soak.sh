#!/bin/bash
# The 72-hour soak of the public tunnel, run from a machine outside the server.
#
#   tunnel-soak.sh run BASE_URL --stream-url URL [options]
#   tunnel-soak.sh summary [LOG]
#
# `run` probes the public HTTPS address every five minutes: GET /healthz must answer 200 with
# {"status":"ok"}, and GET /status must answer 401 without a credential. Once an hour it also holds
# a five-minute stream against the soak-only test server (deploy/vps/soak-stream.py), reconnects
# when it drops, and records the frames it saw, the longest gap between them and the outage from
# the last frame before a drop to the first one after it. The stream is the only measurement fine
# enough to prove that a tunnel restart recovers in under two minutes: the five-minute probes can
# only confirm health at their next tick.
#
# One line per probe goes to the log, and `summary` reads the pass rule back from it: no failed
# handshake (curl reached no HTTP response at all), no failed probe or stream, and every outage
# under two minutes. The log is a flat list of key=value tokens so a human and awk can both read it.
#
# bash rather than sh: timing a stream needs process substitution, which keeps the frame counters
# in this shell while curl writes to a pipe. Both macOS and Linux have bash, and curl is the only
# network tool: no jq, no python.
set -u

DEFAULT_LOG=$HOME/paperwatch-soak.log
INTERVAL=300       # seconds between the two probes
STREAM_WINDOW=300  # seconds a single stream is held open
STREAM_EVERY=3600  # seconds between streams
HOURS=72           # 0 runs until interrupted
MAX_TIME=20        # seconds a single HTTP probe may take
HEARTBEAT=15       # the soak stream's heartbeat, written to the log so the summary knows the noise

usage() {
    cat >&2 <<'USAGE'
usage: tunnel-soak.sh run BASE_URL --stream-url URL [options]
       tunnel-soak.sh summary [LOG]

run     probes BASE_URL every --interval seconds (300) and appends one line per probe to
        --log (default ~/paperwatch-soak.log). Every --stream-every seconds (3600) it holds a
        --stream-window second stream (300) against --stream-url, for --hours hours (72; 0
        runs until interrupted).

summary prints the numbers the pass rule is read from and exits 1 when the log fails it.

options: --log FILE  --interval SECONDS  --stream-url URL  --stream-window SECONDS
         --stream-every SECONDS  --hours HOURS  --max-time SECONDS  --heartbeat SECONDS
USAGE
}

# A typo must be a clean exit with a message, not a shell arithmetic error later.
require_number() {
    case $2 in
        '' | *[!0-9]*)
            printf '%s must be a whole number of %s, got %s\n' "$1" "$3" "$2" >&2
            exit 2
            ;;
    esac
}

require_value() {
    [ "$1" -ge 2 ] || {
        printf '%s needs a value\n' "$2" >&2
        usage
        exit 2
    }
}

# A trapped signal runs only once the command the shell is waiting for returns, and that command
# can be a curl a --max-time long or the sleep between ticks an --interval long: stopping the soak
# would wait for the tunnel. So every command that can block runs in the background, `wait` (a
# builtin, which a trapped signal does interrupt) is what blocks, and the trap kills the child it
# holds.
CHILD=
bg_wait() {
    "$@" &
    CHILD=$!
    wait "$CHILD"
    local status=$?
    CHILD=
    return $status
}

stop_soak() {
    [ -z "$CHILD" ] || kill "$CHILD" 2>/dev/null
    CHILD=
    # To fd 3, the operator's stdout saved before `run`: the trap runs inside whatever redirection
    # the interrupted command had, and a probe's curl writes to a temporary file.
    printf '\nstopped. read the numbers with: %s summary %s\n' "$0" "$LOG" >&3
    exit 130
}

# curl prints %{time_total} as seconds with a fraction; the log wants whole milliseconds. Pure
# shell arithmetic, because awk's printf and bc are not on every machine this runs on.
to_ms() {
    local total=$1 seconds fraction
    case $total in
        *.*)
            seconds=${total%%.*}
            fraction=${total#*.}
            ;;
        *)
            seconds=$total
            fraction=0
            ;;
    esac
    case $seconds in '' | *[!0-9]*) seconds=0 ;; esac
    fraction="${fraction}000"
    fraction=${fraction:0:3}
    case $fraction in *[!0-9]*) fraction=0 ;; esac
    printf '%s' "$((10#$seconds * 1000 + 10#$fraction))"
}

# One clock read for both the readable stamp and the epoch: the summary does its arithmetic on the
# epoch, so the two can never disagree. Both date flags are in BSD and GNU date.
stamp_now() {
    local pair
    pair=$(date -u '+%Y-%m-%dT%H:%M:%SZ %s')
    STAMP=${pair%% *}
    AT=${pair##* }
}

# The status of the first header line a `curl -D` dump wrote, or 000 when no HTTP response arrived.
header_code() {
    local first
    [ -s "$1" ] || {
        printf '000'
        return
    }
    IFS= read -r first <"$1" || first=
    case $first in
        HTTP/*)
            # Word splitting is the point: "HTTP/1.1 200 OK" -> $2 is the code.
            set -- $first
            printf '%s' "${2:-000}"
            ;;
        *) printf '000' ;;
    esac
}

# record PROBE RESULT HTTP MS [FIELD=...]... appends the line and shows it live.
record() {
    local line="ts=$STAMP at=$AT probe=$1 result=$2 http=$3 ms=$4"
    shift 4
    while [ $# -gt 0 ]; do
        line="$line $1"
        shift
    done
    printf '%s\n' "$line" | tee -a "$LOG"
}

# http_probe PROBE PATH EXPECTED_CODE [EXPECTED_BODY]: one unauthenticated GET through the tunnel.
http_probe() {
    local probe=$1 path=$2 want=$3 want_body=${4-}
    local out code total curl_status=0 ms why=
    out=
    # Read back from a file rather than captured, for the reason bg_wait states above: a captured
    # response blocks the shell, and a stop must not wait for one.
    bg_wait curl -sS --connect-timeout 10 --max-time "$MAX_TIME" \
        -o "$TMP/body" -w '%{http_code} %{time_total}\n' "$BASE_URL$path" \
        >"$TMP/probe" 2>/dev/null
    curl_status=$?
    IFS= read -r out <"$TMP/probe"
    code=${out%% *}
    total=${out##* }
    case $out in
        *' '*) ;;
        *) code=000 total=0 ;;
    esac
    ms=$(to_ms "$total")
    if [ "$code" = "000" ]; then
        # The tunnel never carried an HTTP response: DNS, TCP or the TLS handshake failed.
        why=handshake
    elif [ "$code" != "$want" ]; then
        why=http
    elif [ -n "$want_body" ] && [ "$(tr -d '[:space:]' <"$TMP/body")" != "$want_body" ]; then
        why=body
    fi
    stamp_now
    if [ -z "$why" ]; then
        record "$probe" ok "$code" "$ms"
    else
        record "$probe" fail "$code" "$ms" "why=$why" "curl=$curl_status"
    fi
}

# stream_probe holds the soak stream open for STREAM_WINDOW seconds and measures it. A healthy
# stream ends because curl's own --max-time reaches the window's end. One that ends earlier is a
# drop: the probe reconnects while the window has time left, and the outage is the time from the
# last frame before the drop to the first frame after it, which is the restart's recovery time.
# A connection that stays open but goes silent for three heartbeats (plus curl's few seconds of
# speed averaging) is abandoned as a drop too; the outage is still timed from its last frame.
stream_probe() {
    local headers=$TMP/stream.headers code=000 attempt frames=0 gap=0 last= started end left
    local drops=0 down_since= outage=0 ms why= line stall=$((HEARTBEAT * 3))
    started=$SECONDS
    end=$((started + STREAM_WINDOW))
    while :; do
        left=$((end - SECONDS))
        [ "$left" -gt 0 ] || break
        : >"$headers"
        while IFS= read -r line; do
            case $line in
                data:*)
                    frames=$((frames + 1))
                    if [ -n "$down_since" ]; then
                        if [ $((SECONDS - down_since)) -gt "$outage" ]; then
                            outage=$((SECONDS - down_since))
                        fi
                        down_since=
                    fi
                    if [ -n "$last" ] && [ $((SECONDS - last)) -gt "$gap" ]; then
                        gap=$((SECONDS - last))
                    fi
                    last=$SECONDS
                    ;;
            esac
        done < <(curl -sS -N --connect-timeout 10 --max-time "$left" \
            --speed-limit 1 --speed-time "$stall" -D "$headers" -o - "$STREAM_URL" 2>/dev/null)
        attempt=$(header_code "$headers")
        [ "$code" != "000" ] || code=$attempt
        # SECONDS counts whole seconds, so only an end more than a second early is a drop.
        [ $((end - SECONDS)) -gt 1 ] || break
        if [ -z "$down_since" ]; then
            drops=$((drops + 1))
            down_since=${last:-$started}
        fi
        # A refused connection returns at once: the pause keeps the retries from spinning.
        bg_wait sleep 1
    done
    if [ -n "$down_since" ] && [ $((SECONDS - down_since)) -gt "$outage" ]; then
        outage=$((SECONDS - down_since))
    fi
    # Silence from the last frame to the window's end is a gap like any other.
    if [ -n "$last" ] && [ $((SECONDS - last)) -gt "$gap" ]; then
        gap=$((SECONDS - last))
    fi
    ms=$(((SECONDS - started) * 1000))
    if [ "$code" = "000" ]; then
        why=handshake
    elif [ "$frames" -eq 0 ] || [ -n "$down_since" ]; then
        why=unrecovered
    elif [ "$drops" -gt 0 ]; then
        why=drop
    elif [ "$gap" -ge "$stall" ]; then
        why=gap
    fi
    stamp_now
    set -- "frames=$frames" "gap=$gap" "drops=$drops" "outage=$outage" "hb=$HEARTBEAT"
    if [ -z "$why" ]; then
        record stream ok "$code" "$ms" "$@"
    else
        record stream fail "$code" "$ms" "$@" "why=$why"
    fi
}

run() {
    local tick=0 start target sleep_for
    start=$(date -u +%s)
    printf 'soaking %s with the stream at %s, logging to %s\n' "$BASE_URL" "$STREAM_URL" "$LOG"
    if [ "$HOURS" -ne 0 ]; then
        printf 'this run ends after %s hours\n' "$HOURS"
    else
        printf 'this run continues until it is interrupted\n'
    fi
    while :; do
        if [ "$HOURS" -ne 0 ] && [ $((tick * INTERVAL)) -ge $((HOURS * 3600)) ]; then
            break
        fi
        http_probe healthz /healthz 200 '{"status":"ok"}'
        http_probe status /status 401
        if [ "$STREAM_EVERY" -gt 0 ] && [ $(((tick * INTERVAL) % STREAM_EVERY)) -eq 0 ]; then
            stream_probe
        fi
        tick=$((tick + 1))
        # Absolute targets, not sleeps of a fixed length: a slow probe must not drift the cadence.
        target=$((start + tick * INTERVAL))
        sleep_for=$((target - $(date -u +%s)))
        if [ "$sleep_for" -gt 0 ]; then
            bg_wait sleep "$sleep_for"
        fi
    done
    printf 'soak finished. read the numbers with: %s summary %s\n' "$0" "$LOG"
}

summary() {
    local log=$1
    [ -f "$log" ] || {
        printf 'no soak log at %s\n' "$log" >&2
        return 2
    }
    # `path`, not `log`: log is a built-in awk function, and a variable named log prints -inf.
    awk -v path="$log" '
    function duration(seconds, hours, minutes) {
        if (seconds < 0) seconds = 0
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        if (hours > 0) return sprintf("%dh %dm %ds", hours, minutes, seconds % 60)
        if (minutes > 0) return sprintf("%dm %ds", minutes, seconds % 60)
        return sprintf("%ds", seconds)
    }
    {
        # One probe per line as key=value tokens (split("", f) is the portable way to clear it).
        split("", f)
        for (i = 1; i <= NF; i++) {
            equals = index($i, "=")
            if (equals > 0) f[substr($i, 1, equals - 1)] = substr($i, equals + 1)
        }
        if (!("probe" in f) || !("at" in f)) next
        kind = f["probe"]
        at = f["at"] + 0
        total[kind]++
        if (f["result"] == "ok") {
            ok[kind]++
        } else {
            failed[kind]++
            if (f["why"] == "handshake" || f["http"] == "000") handshakes++
        }
        if (kind == "healthz" || kind == "status") {
            if (seen[kind] && at - seen_at[kind] > longest_gap) longest_gap = at - seen_at[kind]
            seen[kind] = 1
            seen_at[kind] = at
            if (f["result"] != "ok") {
                if (!(kind in started_at)) started_at[kind] = at
            } else if (kind in started_at) {
                # A probe confirms health only at its own tick, so an outage is measured to the
                # next successful probe of the same kind, not to the last failure.
                if (at - started_at[kind] > longest_outage) longest_outage = at - started_at[kind]
                delete started_at[kind]
            }
            last_at[kind] = at
        } else if (kind == "stream") {
            # A gap of more than two heartbeats is a break; one missing heartbeat is not.
            if (f["gap"] + 0 > 2 * (f["hb"] + 0) && f["gap"] + 0 > longest_outage) {
                longest_outage = f["gap"] + 0
            }
            if (f["outage"] + 0 > longest_outage) longest_outage = f["outage"] + 0
        }
        if (first_at == 0) {
            first_at = at
            first_ts = f["ts"]
        }
        last_at_any = at
        last_ts = f["ts"]
    }
    END {
        if (first_at == 0) {
            printf "no probes in %s\n", path > "/dev/stderr"
            exit 2
        }
        # A failure run the log ends on is still an outage; the last probe is where it was seen.
        for (kind in started_at) {
            if (last_at[kind] - started_at[kind] > longest_outage) {
                longest_outage = last_at[kind] - started_at[kind]
            }
        }
        printf "log: %s\n", path
        printf "window: %s to %s (%s)\n", first_ts, last_ts, duration(last_at_any - first_at)
        printf "probes: %d, %d ok, %d failed (healthz %d, status %d)\n", \
            total["healthz"] + total["status"], ok["healthz"] + ok["status"], \
            failed["healthz"] + failed["status"], total["healthz"] + 0, total["status"] + 0
        printf "failed handshakes: %d\n", handshakes + 0
        printf "streams: %d, %d ok, %d failed\n", \
            total["stream"] + 0, ok["stream"] + 0, failed["stream"] + 0
        printf "longest outage: %s\n", duration(longest_outage)
        printf "longest gap between probes: %s\n", duration(longest_gap)
        reasons = ""
        if (handshakes > 0) reasons = handshakes " failed handshakes"
        if (failed["healthz"] + failed["status"] > 0) {
            reasons = reasons (reasons == "" ? "" : ", ") failed["healthz"] + failed["status"] \
                " failed probes"
        }
        if (failed["stream"] > 0) {
            reasons = reasons (reasons == "" ? "" : ", ") failed["stream"] " failed streams"
        }
        if (longest_outage >= 120) {
            reasons = reasons (reasons == "" ? "" : ", ") "longest outage " \
                duration(longest_outage) " is over 2m"
        }
        if (reasons == "") {
            printf "verdict: PASS\n"
            exit 0
        }
        printf "verdict: FAIL - %s\n", reasons
        exit 1
    }
    ' "$log"
}

[ $# -ge 1 ] || {
    usage
    exit 2
}
MODE=$1
shift

case $MODE in
    run)
        [ $# -ge 1 ] || {
            usage
            exit 2
        }
        BASE_URL=$1
        shift
        case $BASE_URL in
            http://* | https://*) ;;
            *)
                printf 'BASE_URL must start with http:// or https://, got %s\n' "$BASE_URL" >&2
                exit 2
                ;;
        esac
        # A trailing slash would make every path "//healthz" through the tunnel's rewrite.
        while [ "${BASE_URL%/}" != "$BASE_URL" ]; do BASE_URL=${BASE_URL%/}; done
        LOG=$DEFAULT_LOG
        STREAM_URL=
        while [ $# -gt 0 ]; do
            case $1 in
                --log)
                    require_value $# "$1"
                    LOG=$2
                    shift 2
                    ;;
                --stream-url)
                    require_value $# "$1"
                    STREAM_URL=$2
                    shift 2
                    ;;
                --interval)
                    require_value $# "$1"
                    INTERVAL=$2
                    shift 2
                    ;;
                --stream-window)
                    require_value $# "$1"
                    STREAM_WINDOW=$2
                    shift 2
                    ;;
                --stream-every)
                    require_value $# "$1"
                    STREAM_EVERY=$2
                    shift 2
                    ;;
                --hours)
                    require_value $# "$1"
                    HOURS=$2
                    shift 2
                    ;;
                --max-time)
                    require_value $# "$1"
                    MAX_TIME=$2
                    shift 2
                    ;;
                --heartbeat)
                    require_value $# "$1"
                    HEARTBEAT=$2
                    shift 2
                    ;;
                *)
                    printf 'unknown argument: %s\n' "$1" >&2
                    usage
                    exit 2
                    ;;
            esac
        done
        [ -n "$STREAM_URL" ] || {
            printf 'run needs --stream-url URL (the soak stream server, port 8443)\n' >&2
            exit 2
        }
        require_number --interval "$INTERVAL" seconds
        require_number --stream-window "$STREAM_WINDOW" seconds
        require_number --stream-every "$STREAM_EVERY" seconds
        require_number --hours "$HOURS" hours
        require_number --max-time "$MAX_TIME" seconds
        require_number --heartbeat "$HEARTBEAT" seconds
        [ "$INTERVAL" -gt 0 ] && [ "$MAX_TIME" -gt 0 ] && [ "$STREAM_WINDOW" -gt 0 ] &&
            [ "$HEARTBEAT" -gt 0 ] || {
            printf 'the interval, window, max-time and heartbeat must be at least a second\n' >&2
            exit 2
        }
        command -v curl >/dev/null 2>&1 || {
            printf 'curl is required\n' >&2
            exit 2
        }
        mkdir -p "$(dirname "$LOG")" 2>/dev/null
        : >>"$LOG" 2>/dev/null || {
            printf 'cannot write the log at %s\n' "$LOG" >&2
            exit 2
        }
        TMP=$(umask 077; mktemp -d "${TMPDIR:-/tmp}/paperwatch-soak.XXXXXX") || {
            printf 'cannot create a private temporary directory\n' >&2
            exit 2
        }
        trap 'rm -rf "$TMP"' EXIT
        trap stop_soak INT TERM
        exec 3>&1
        run
        ;;
    summary)
        LOG=$DEFAULT_LOG
        if [ $# -ge 1 ]; then
            LOG=$1
            shift
        fi
        [ $# -eq 0 ] || {
            printf 'summary takes one log path\n' >&2
            exit 2
        }
        summary "$LOG"
        ;;
    *)
        usage
        exit 2
        ;;
esac
