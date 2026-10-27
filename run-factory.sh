#!/usr/bin/env bash
# Serves the agent factory engine behind the frontend's factory boards on 127.0.0.1:8090.
#
#   source    Agents-in-a-box (`agent_sandbox.factory`), run from its checkout
#   config    ~/.config/agent-sandbox/factory.yaml (created on first start)
#   state     database `factory` in the PostgreSQL started by run-backend.sh, the production
#             one (127.0.0.2:5433) with --prod
#
# Idempotent: an engine already answering on the port is reused.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir/deploy/lib.sh"
repo="${GIGACHAD_FACTORY_REPO_DIR:-$HOME/git/Agents-in-a-box}"
config="${AGENT_SANDBOX_FACTORY_CONFIG:-$HOME/.config/agent-sandbox/factory.yaml}"
bind=127.0.0.1:8090
profile=development

case "${1:-}" in
    '') ;;
    --prod) profile=production ;;
    *)
        printf 'usage: %s [--prod]\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac
postgres_profile "$profile"

if port_open "$bind"; then
    printf 'Reusing the factory engine already listening on %s\n' "$bind"
    exit 0
fi
if [[ ! -e "$repo" ]]; then
    printf 'Cloning the factory engine into %s\n' "$repo"
    mkdir -p "$(dirname -- "$repo")"
    git clone https://github.com/patrickab/Agents-in-a-box.git "$repo"
fi
if [[ ! -f "$repo/pyproject.toml" ]]; then
    printf 'error: %s is not an Agents-in-a-box checkout (set GIGACHAD_FACTORY_REPO_DIR)\n' "$repo" >&2
    exit 1
fi
# run.sh starts PostgreSQL through run-backend.sh in parallel, so give it time to come up.
if ! wait_until 60 1 port_open "$pg_host:$pg_port"; then
    printf 'PostgreSQL is not up on %s:%s, start it with ./run-backend.sh first\n' "$pg_host" "$pg_port" >&2
    exit 1
fi

load_secrets
if [[ -z "${!password_var:-}" ]]; then
    printf 'error: %s is required in ~/.secrets\n' "$password_var" >&2
    exit 1
fi
export FACTORY_DATABASE_URL="postgresql://${USER}:${!password_var}@${pg_host}:${pg_port}/factory"

printf 'Installing factory engine dependencies\n'
uv sync --directory "$repo" --locked --extra factory --no-dev

if [[ ! -e "$config" ]]; then
    if [[ -n "${AGENT_SANDBOX_FACTORY_CONFIG:-}" ]]; then
        printf 'error: factory config does not exist: %s\n' "$config" >&2
        exit 1
    fi
    mkdir -p "$(dirname -- "$config")"
    (umask 077 && cat > "$config" <<'YAML'
listen: 127.0.0.1:8090
database_url: ${FACTORY_DATABASE_URL}
preview_ports: [8100, 8199]
limits:
  in_progress: 4
YAML
    )
fi

# Existing PostgreSQL volumes predate the factory, so image initialization is not enough.
uv run --directory "$repo" --no-sync python - <<'PY'
import os

import psycopg
from psycopg.conninfo import make_conninfo

connection = make_conninfo(os.environ["FACTORY_DATABASE_URL"], dbname="postgres")
with psycopg.connect(connection, autocommit=True) as database:
    if not database.execute("SELECT 1 FROM pg_database WHERE datname = 'factory'").fetchone():
        database.execute("CREATE DATABASE factory")
PY

printf 'Starting the factory engine on %s (%s)\n' "$bind" "$profile"
exec uv run --directory "$repo" --no-sync agent-sandbox factory serve --config "$config"
