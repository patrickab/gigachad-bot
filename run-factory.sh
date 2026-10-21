#!/usr/bin/env bash
# Serves the agent factory engine behind the frontend's factory boards on 127.0.0.1:8090.
#
#   source    Agents-in-a-box (`agent_sandbox.factory`), run from its checkout
#   config    ~/.config/agent-sandbox/factory.yaml (listen, database_url: ${FACTORY_DATABASE_URL}, ...)
#   state     database `factory` in the PostgreSQL started by run-backend.sh, the production
#             one (127.0.0.2:5433) with --prod
#
# Idempotent: an engine already answering on the port is reused.
set -euo pipefail

repo="${GIGACHAD_FACTORY_REPO_DIR:-$HOME/git/Agents-in-a-box}"
bind=127.0.0.1:8090
profile=development
pg_host=127.0.0.1
pg_port=5432

case "${1:-}" in
    '') ;;
    --prod) profile=production pg_host=127.0.0.2 pg_port=5433 ;;
    *)
        printf 'usage: %s [--prod]\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac

port_open() {
    local host="${1%%:*}" port="${1##*:}"
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
        exec 3<&- 3>&-
        return 0
    fi
    return 1
}

if port_open "$bind"; then
    printf 'Reusing the factory engine already listening on %s\n' "$bind"
    exit 0
fi
if [[ ! -f "$repo/pyproject.toml" ]]; then
    printf 'No Agents-in-a-box checkout at %s (set GIGACHAD_FACTORY_REPO_DIR); skipping the factory engine\n' "$repo" >&2
    [[ $profile == production ]] && exit 1
    exit 0
fi
# run.sh starts PostgreSQL through run-backend.sh in parallel, so give it time to come up.
for _ in {1..60}; do
    port_open "$pg_host:$pg_port" && break
    sleep 1
done
if ! port_open "$pg_host:$pg_port"; then
    printf 'PostgreSQL is not up on %s:%s, start it with ./run-backend.sh first\n' "$pg_host" "$pg_port" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$HOME/.secrets"
password_var="POSTGRES_PASSWORD_$([[ $profile == production ]] && echo PROD || echo DEV)"
if [[ -z "${!password_var:-}" ]]; then
    printf 'error: %s is required in ~/.secrets\n' "$password_var" >&2
    exit 1
fi
export FACTORY_DATABASE_URL="postgresql://${USER}:${!password_var}@${pg_host}:${pg_port}/factory"

printf 'Starting the factory engine on %s (%s)\n' "$bind" "$profile"
exec uv run --directory "$repo" agent-sandbox factory serve
