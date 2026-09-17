#!/usr/bin/env bash
# Starts the two OMP services that make OMP's own logins callable as models:
#
#   auth-broker   holds the credentials (including OAuth subscription plans)
#   auth-gateway  OpenAI-compatible forward proxy in front of the broker
#
# The backend points LiteLLM's `litellm_proxy/` prefix at the gateway, so a
# model picked from OMP spends OMP's credential without gigachad ever seeing
# a token. Both services are loopback-only.
#
# Idempotent: a service already listening on its port is reused rather than
# restarted, so a second gigachad instance (or a hand-started `omp
# auth-broker serve`) does not collide. Only services this script owns are
# stopped on exit.
set -euo pipefail

broker_bind="${GIGACHAD_OMP_BROKER_BIND:-127.0.0.1:8765}"
gateway_bind="${GIGACHAD_OMP_GATEWAY_BIND:-127.0.0.1:4000}"
owned_pids=()

if ! command -v omp >/dev/null 2>&1; then
    printf 'omp not on PATH; skipping OMP model source\n' >&2
    exit 0
fi
if [[ ! -e "${PI_CONFIG_DIR:-$HOME/.omp}/agent/agent.db" ]]; then
    printf 'No OMP credential store; skipping OMP model source\n' >&2
    exit 0
fi

cleanup() {
    local pid
    for (( i = ${#owned_pids[@]} - 1; i >= 0; i-- )); do
        pid="${owned_pids[i]}"
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 0' INT TERM

port_open() {
    local host="${1%%:*}" port="${1##*:}"
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
        exec 3<&- 3>&-
        return 0
    fi
    return 1
}

# Launches "$@" unless the bind address already answers. Sets `started_pid` to
# the new child's pid, or to an empty string when an existing service is
# reused. Returns non-zero only when nothing ends up listening.
start_service() {
    local label="$1" bind="$2"
    shift 2
    started_pid=''

    if port_open "$bind"; then
        printf 'Reusing the OMP %s already listening on %s\n' "$label" "$bind"
        return 0
    fi

    printf 'Starting OMP %s on %s\n' "$label" "$bind"
    "$@" &
    local pid=$! tries=0
    # Give up as soon as the child dies instead of waiting out the timeout:
    # a crashed service and a slow one need different reporting.
    while (( tries < 120 )); do
        if port_open "$bind"; then
            started_pid="$pid"
            owned_pids+=("$pid")
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            # A racing instance may have won the port while this child died.
            if port_open "$bind"; then
                printf 'OMP %s lost a startup race; reusing the winner on %s\n' "$label" "$bind" >&2
                return 0
            fi
            printf 'OMP %s exited during startup\n' "$label" >&2
            return 1
        fi
        sleep 0.25
        tries=$(( tries + 1 ))
    done

    printf 'OMP %s did not answer on %s\n' "$label" "$bind" >&2
    kill -TERM "$pid" 2>/dev/null || true
    return 1
}

if ! start_service "auth-broker" "$broker_bind" omp auth-broker serve --bind="$broker_bind"; then
    printf 'Skipping OMP model source\n' >&2
    exit 0
fi

export OMP_AUTH_BROKER_URL="http://$broker_bind"
OMP_AUTH_BROKER_TOKEN="$(omp auth-broker token | tail -n 1)"
export OMP_AUTH_BROKER_TOKEN

# The gateway keeps its bearer in ~/.omp/auth-gateway.token, which the backend
# reads for itself — so the proxy is not left open to every local process.
if ! start_service "auth-gateway" "$gateway_bind" omp auth-gateway serve --bind="$gateway_bind"; then
    printf 'Skipping OMP model source\n' >&2
    exit 0
fi

printf 'OMP model source ready at http://%s/v1\n' "$gateway_bind"

if (( ${#owned_pids[@]} == 0 )); then
    printf 'Both OMP services were already running; nothing to supervise\n'
    exit 0
fi

# Exit when any owned service stops, so the trap tears down its sibling.
wait -n "${owned_pids[@]}" || true
