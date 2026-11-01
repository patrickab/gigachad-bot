#!/usr/bin/env bash
# Production omp-web for the omp tabs, one of three processes picked by the argument:
#
#   server  upstream omp-web plus ompweb-overlay/ on 127.0.0.1:30177 (run-omp-web.sh), framed by the frontend origins
#           in GIGACHAD_CORS_ORIGINS unless GIGACHAD_OMP_WEB_FRAME_ANCESTORS is set
#   host    `ompweb host`, the daemon that runs this machine's omp sessions, from the checkout the server serves
#   gate    the Tailnet identity gate (src/backend/omp_web_gate.py) on
#           127.0.0.1:${GIGACHAD_OMP_WEB_GATE_PORT:-30179}, Tailscale Serve's :8443 points here
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -z ${GIGACHAD_ENV_FILE+x} ]]; then
    GIGACHAD_ENV_FILE="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
fi
source "${repo_root}/deploy/load-env.sh"

case "${1:-}" in
    server)
        if [[ -z ${GIGACHAD_OMP_WEB_FRAME_ANCESTORS:-} && -n ${GIGACHAD_CORS_ORIGINS:-} ]]; then
            export GIGACHAD_OMP_WEB_FRAME_ANCESTORS="${GIGACHAD_CORS_ORIGINS//,/ }"
        fi
        exec "${repo_root}/run-omp-web.sh"
        ;;
    host)
        # The daemon runs from the checkout the server serves, so both ship one commit of the controlled ref. Missing before
        # the server's first start: the unit's Restart=always retries.
        served_dir="${GIGACHAD_OMP_WEB_DIR:-$HOME/.local/share/gigachad-bot/ompweb-upstream}"
        if [[ ! -f "$served_dir/bin/omp-web.js" ]]; then
            printf 'omp-web checkout %s is not built yet: the server unit creates it\n' "$served_dir" >&2
            exit 1
        fi
        # Node >= 24 comes from the project venv, like the server's.
        export PATH="${repo_root}/.venv/bin:$PATH"
        exec node "$served_dir/bin/omp-web.js" host
        ;;
    gate)
        exec "${repo_root}/.venv/bin/uvicorn" backend.omp_web_gate:app --app-dir "${repo_root}/src" \
            --host 127.0.0.1 --port "${GIGACHAD_OMP_WEB_GATE_PORT:-30179}" --no-access-log
        ;;
    *)
        printf 'usage: %s server|host|gate\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac
