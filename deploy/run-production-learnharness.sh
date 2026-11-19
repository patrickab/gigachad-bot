#!/usr/bin/env bash
# Production learn mode, picked by the argument:
#
#   server  LearnHarness plus learnharness-overlay/ on GIGACHAD_LEARNHARNESS_BIND (default 127.0.0.1:30181,
#           run-learnharness.sh), which runs it as the sandboxed transient user unit gigachad-learnharness-<port> and stops
#           that unit when it is stopped. Each start first updates to the newest release tag the overlay checks pass
#           against, falling back to the last working commit (unless GIGACHAD_LEARNHARNESS_REF pins one or
#           GIGACHAD_AUTO_UPDATE=0), and regenerates its models. Framed by the frontend origins in GIGACHAD_CORS_ORIGINS
#           unless GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS is set
#   models  regenerate LearnHarness's models without restarting it (redeploy.sh)
#   gate    the Tailnet identity gate (src/backend/learnharness_gate.py) on
#           127.0.0.1:${GIGACHAD_LEARNHARNESS_GATE_PORT:-30182}, Tailscale Serve's :8444 points here
#
# server and models read the model catalog of GIGACHAD_LEARNHARNESS_LOGIN (required, a Tailscale login) from the
# production PostgreSQL, the one run-backend.sh --prod serves, and the models production's pi-host gateway serves.
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -z ${GIGACHAD_ENV_FILE+x} ]]; then
    GIGACHAD_ENV_FILE="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
fi
source "${repo_root}/deploy/lib.sh"
# Like run-backend.sh, ~/.secrets first: the generator needs the database password, LearnHarness BRAVE_API_KEY (the
# launcher passes it only its allowlist). The gate needs none.
if [[ ${1:-} == server || ${1:-} == models ]]; then
    load_secrets
fi
source "${repo_root}/deploy/load-env.sh"

# The generator's catalog and gateway. The OS user names no gigachad user in production, logins are Tailscale's. A
# configuration error exits 78 (EX_CONFIG), which the unit does not restart on.
production_catalog() {
    if [[ -z ${GIGACHAD_LEARNHARNESS_LOGIN:-} ]]; then
        printf 'error: set GIGACHAD_LEARNHARNESS_LOGIN in %s to the Tailscale login whose models learn mode serves\n' \
            "$GIGACHAD_ENV_FILE" >&2
        exit 78
    fi
    postgres_profile production
    if [[ -z ${!password_var:-} ]]; then
        printf 'error: %s is required for the production PostgreSQL\n' "$password_var" >&2
        exit 78
    fi
    # Set like run-backend.sh --prod, so an inherited value never points it at another database.
    export PGPASSWORD="${!password_var}"
    export GIGACHAD_DATABASE_URL="postgresql://$(id -un)@${pg_host}:${pg_port}/gigachad"
    # Production's pi-host (gigachad-bot-pi-host.service), LearnHarness's custom endpoint.
    export GIGACHAD_PI_HOST_URL="${GIGACHAD_PI_HOST_URL:-http://127.0.0.1:8301}"
}

case "${1:-}" in
    server)
        production_catalog
        if [[ -z ${GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS:-} && -n ${GIGACHAD_CORS_ORIGINS:-} ]]; then
            export GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS="${GIGACHAD_CORS_ORIGINS//,/ }"
        fi
        exec "${repo_root}/run-learnharness.sh"
        ;;
    models)
        production_catalog
        PYTHONPATH="${repo_root}/src" exec "${repo_root}/.venv/bin/python" -m lib.learnharness_config write-config
        ;;
    gate)
        exec "${repo_root}/.venv/bin/uvicorn" backend.learnharness_gate:app --app-dir "${repo_root}/src" \
            --host 127.0.0.1 --port "${GIGACHAD_LEARNHARNESS_GATE_PORT:-30182}" --no-access-log
        ;;
    *)
        printf 'usage: %s server|models|gate\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac
