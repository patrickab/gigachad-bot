#!/usr/bin/env bash
# Serves the agent factory engine behind the frontend's Factory board.
#
#   source    Agents-in-a-box (`agent_sandbox.factory`), run from its checkout
#   config    ~/.config/agent-sandbox/factory.yaml (listen, worktree root, defaults directory)
#   state     database `factory` in the dev PostgreSQL started by run-backend.sh
#
# Idempotent: an engine already answering on the port is reused.
set -euo pipefail

repo="${GIGACHAD_FACTORY_REPO_DIR:-$HOME/git/Agents-in-a-box}"
bind="${GIGACHAD_FACTORY_BIND:-127.0.0.1:8090}"
pg_host=127.0.0.1
pg_port=5432

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
if [[ -z "${POSTGRES_PASSWORD_DEV:-}" ]]; then
    printf 'error: POSTGRES_PASSWORD_DEV is required in ~/.secrets\n' >&2
    exit 1
fi
export FACTORY_DATABASE_URL="postgresql://${USER}:${POSTGRES_PASSWORD_DEV}@${pg_host}:${pg_port}/factory"

printf 'Starting the factory engine on %s\n' "$bind"
exec uv run --directory "$repo" agent-sandbox factory serve
