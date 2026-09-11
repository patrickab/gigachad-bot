#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -z ${GIGACHAD_ENV_FILE+x} ]]; then
    GIGACHAD_ENV_FILE="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
fi

source "${repo_root}/deploy/load-env.sh"
exec "${repo_root}/.venv/bin/uvicorn" src.backend.server:app --host 127.0.0.1 --port 8001
