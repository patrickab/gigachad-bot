#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
profile=development
tailscale=false
postgres_started=false
backend_pid=''
c4_pid=''

usage() {
    printf 'usage: %s [--prod] [--tailscale]\n' "${BASH_SOURCE[0]}" >&2
}

while (($#)); do
    case "$1" in
        --prod)
            profile=production
            ;;
        --tailscale)
            tailscale=true
            ;;
        *)
            usage
            exit 2
            ;;
    esac
    shift
done

if [[ "$profile" == production && "$tailscale" == true ]]; then
    printf '%s\n' '--tailscale is a development-only option' >&2
    exit 2
fi

project="gigachad-dev"
port=8001
c4_port=8011
pg_host=127.0.0.1
pg_port=5432
if [[ "$profile" == production ]]; then
    project="gigachad-prod"
    port=8002
    c4_port=8012
    pg_host=127.0.0.2
    pg_port=5433
fi

compose() {
    docker compose --project-directory "$root_dir" --project-name "$project" "$@"
}

postgres_is_running() {
    local services
    services="$(compose ps --status running --services)"
    [[ $'\n'"$services"$'\n' == *$'\npostgres\n'* ]]
}

stop_postgres() {
    if [[ "$postgres_started" == true ]]; then
        printf 'Stopping PostgreSQL container for %s\n' "$profile"
        compose stop postgres || true
    fi
}

# A process and everything it started, children before parents.
tree() {
    local child
    for child in $(pgrep -P "$1"); do
        tree "$child"
    done
    printf '%s\n' "$1"
}

# Asks each child to stop, and kills whatever is left of its tree after 10s. uvicorn's
# worker holds the port itself, and one hung in shutdown would otherwise keep it, with
# or without its reloader, until killed by hand.
stop_children() {
    local pid alive
    local -a pids
    for pid in "$backend_pid" "$c4_pid"; do
        if [[ -n "$pid" ]]; then
            mapfile -t pids < <(tree "$pid")
            kill -TERM "$pid" 2>/dev/null || true
            for _ in {1..20}; do
                alive=$(ps -o stat=,pid= -p "$(IFS=,; echo "${pids[*]}")" 2>/dev/null | awk '$1 !~ /^Z/ { print $2 }')
                [[ -z "$alive" ]] && break
                sleep 0.5
            done
            if [[ -n "$alive" ]]; then
                printf 'warning: %s did not stop within 10s, killing it\n' "$(tr '\n' ' ' <<<"$alive")" >&2
                kill -KILL "${pids[@]}" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
        fi
    done
    backend_pid=''
    c4_pid=''
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_children
    stop_postgres
    exit "$status"
}

interrupt() {
    stop_children
    exit 0
}

trap cleanup EXIT
trap interrupt INT TERM

cd "$root_dir"
source "$HOME/.secrets"
source "$root_dir/deploy/load-env.sh"

if [[ "$profile" == production ]]; then
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD_PROD:-}"
else
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD_DEV:-}"
fi
if [[ -z "$POSTGRES_PASSWORD" ]]; then
    printf 'error: POSTGRES_PASSWORD_%s is required for the managed PostgreSQL container\n' "${profile^^}" >&2
    exit 1
fi
export POSTGRES_PASSWORD
export PGPASSWORD="$POSTGRES_PASSWORD"
export POSTGRES_USER="$(whoami)"
export POSTGRES_HOST="$pg_host"
export POSTGRES_PORT="$pg_port"
export GIGACHAD_DATABASE_URL="postgresql://${POSTGRES_USER}@${pg_host}:${pg_port}/gigachad"

# Production sits behind Tailscale Serve, which injects the requesting login.
# Development binds loopback only, so headerless requests are trusted as the
# local user. Set explicitly so an inherited value can never relax production.
if [[ "$profile" == production ]]; then
    export GIGACHAD_ENV=production
else
    export GIGACHAD_ENV=development
    export GIGACHAD_DEV_USER="$(whoami)"
fi

if ! postgres_is_running; then
    printf 'Starting PostgreSQL container for %s\n' "$profile"
    compose up --detach --wait postgres
    postgres_started=true
fi

if [[ "$profile" == development ]]; then
    printf 'Installing backend dependencies\n'
    uv sync
fi
# A user unit started at boot has systemd's bare PATH, without the mise shims node lives behind.
export PATH="$root_dir/.venv/bin:$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
if ! command -v node >/dev/null; then
    printf 'error: node is not on PATH (%s); the C4 service cannot start\n' "$PATH" >&2
    exit 1
fi
if [[ "$profile" == development || ! -d "$root_dir/src/c4/node_modules" ]]; then
    printf 'Installing C4 service dependencies\n'
    npm ci --prefix "$root_dir/src/c4" --no-audit --no-fund
fi

# The C4 service parses and edits LikeC4 sources for the architecture routes.
# Loopback only, stateless, owned by this script's lifetime.
printf 'Starting C4 service on http://127.0.0.1:%s\n' "$c4_port"
C4_SERVICE_HOST=127.0.0.1 C4_SERVICE_PORT="$c4_port" node "$root_dir/src/c4/server.ts" &
c4_pid=$!
export C4_SERVICE_URL="http://127.0.0.1:${c4_port}"
# The backend answers 503 on every architecture route while this service is down, so it
# only starts once the service is healthy, and one that never gets there fails the unit.
for _ in {1..60}; do
    curl -sf -o /dev/null "$C4_SERVICE_URL/health" && break
    if ! kill -0 "$c4_pid" 2>/dev/null; then
        printf 'error: the C4 service exited during startup\n' >&2
        c4_pid=''
        exit 1
    fi
    sleep 0.5
done
if ! curl -sf -o /dev/null "$C4_SERVICE_URL/health"; then
    printf 'error: the C4 service did not become healthy within 30s\n' >&2
    exit 1
fi

if [[ "$tailscale" == true ]]; then
    printf 'Configuring Tailnet-only HTTPS ingress\n'
    "$root_dir/deploy/tailscale-serve.sh" "$port"
fi

if [[ "$profile" == production ]]; then
    workers="${GIGACHAD_UVICORN_WORKERS:-4}"
    printf 'Starting production backend on http://127.0.0.1:%s (%s workers)\n' "$port" "$workers"
    uvicorn src.backend.server:app --host 127.0.0.1 --port "$port" --workers "$workers" &
else
    printf 'Starting development backend on http://127.0.0.1:%s\n' "$port"
    uvicorn src.backend.server:app --host 127.0.0.1 --port "$port" --reload --reload-dir src &
fi
backend_pid=$!

# Either process ending takes the other down with it and fails the script, so systemd's
# Restart=on-failure brings both back instead of leaving a backend without its C4 service.
wait -n "$backend_pid" "$c4_pid" || true
if ! kill -0 "$backend_pid" 2>/dev/null; then
    if wait "$backend_pid"; then
        backend_pid=''
        exit 0
    fi
    status=$?
    backend_pid=''
    exit "$status"
fi
printf 'error: the C4 service exited; stopping the backend so it is restarted together\n' >&2
c4_pid=''
exit 1
