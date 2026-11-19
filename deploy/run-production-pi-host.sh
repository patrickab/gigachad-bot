#!/usr/bin/env bash
# Production pi-host (run-pi-host.sh): the Pi sessions, completion gateway and usage service of the production backend and
# the LiteLLM proxy, on 127.0.0.1:$PI_HOST_PORT (gigachad-bot-pi-host.service sets 8301, development's pi-host holds
# 8300). It shares development's token file and ~/.pi/agent: one host user, and pi-host's session-file lock (M1) keeps
# one writer per session across both instances.
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -z ${GIGACHAD_ENV_FILE+x} ]]; then
    GIGACHAD_ENV_FILE="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
fi
source "${repo_root}/deploy/lib.sh"
# Like run.sh, ~/.secrets first: the gateway's API-key providers read their keys from pi-host's environment. Pi sessions
# get only pi-host's child allowlist (M6), never these keys.
load_secrets
source "${repo_root}/deploy/load-env.sh"

exec "${repo_root}/run-pi-host.sh"
