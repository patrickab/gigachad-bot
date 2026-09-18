#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
profile=development
tailscale=false
postgres_started=false
backend_pid=''

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
pg_host=127.0.0.1
pg_port=5432
if [[ "$profile" == production ]]; then
    project="gigachad-prod"
    port=8002
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

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$backend_pid" ]]; then
        kill -TERM "$backend_pid" 2>/dev/null || true
        wait "$backend_pid" 2>/dev/null || true
    fi
    stop_postgres
    exit "$status"
}

interrupt() {
    if [[ -n "$backend_pid" ]]; then
        kill -TERM "$backend_pid" 2>/dev/null || true
        wait "$backend_pid" 2>/dev/null || true
        backend_pid=''
    fi
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
export PATH="$root_dir/.venv/bin:$PATH"

if [[ "$tailscale" == true ]]; then
    printf 'Configuring Tailnet-only HTTPS ingress\n'
    "$root_dir/deploy/tailscale-serve.sh" "$port"
fi

if [[ "$profile" == production ]]; then
    printf 'Starting production backend on http://127.0.0.1:%s\n' "$port"
    uvicorn src.backend.server:app --host 127.0.0.1 --port "$port" &
else
    printf 'Starting development backend on http://127.0.0.1:%s\n' "$port"
    uvicorn src.backend.server:app --host 127.0.0.1 --port "$port" --reload --reload-dir src &
fi
backend_pid=$!

if wait "$backend_pid"; then
    backend_pid=''
    exit 0
fi
status=$?
backend_pid=''
exit "$status"
