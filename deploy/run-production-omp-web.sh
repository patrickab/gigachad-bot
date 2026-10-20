#!/usr/bin/env bash
# Production omp-web for the omp tabs, one of two processes picked by the argument:
#
#   server  omp-web on 127.0.0.1:30177 (run-omp-web.sh), framed by the frontend origins
#           in GIGACHAD_CORS_ORIGINS unless GIGACHAD_OMP_WEB_FRAME_ANCESTORS is set
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
    gate)
        exec "${repo_root}/.venv/bin/uvicorn" backend.omp_web_gate:app --app-dir "${repo_root}/src" \
            --host 127.0.0.1 --port "${GIGACHAD_OMP_WEB_GATE_PORT:-30179}" --no-access-log
        ;;
    *)
        printf 'usage: %s server|gate\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac
