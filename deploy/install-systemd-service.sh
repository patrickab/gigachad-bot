#!/usr/bin/env bash
# Install the user services from a production worktree. It never starts them, retired ones it stops and removes.
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly worktree="$(cd -- "$script_dir/.." && pwd)"
readonly unit_dir="${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user"
readonly backend="${worktree}/.venv/bin/uvicorn"

require_runner() {
    if [[ ! -x $2 ]]; then
        printf 'Production %s runner is missing or not executable: %s\n' "$1" "$2" >&2
        exit 1
    fi
}

if [[ ! -x $backend ]]; then
    printf 'Production backend executable is missing: %s\nRun: cd %s && uv sync --python 3.12 --locked --no-dev\n' "$backend" "$worktree" >&2
    exit 1
fi
require_runner backend "${worktree}/run-backend.sh"
require_runner pi-host "${script_dir}/run-production-pi-host.sh"
require_runner omp-web "${script_dir}/run-production-omp-web.sh"
require_runner factory "${worktree}/run-factory.sh"
require_runner "preview gate" "${worktree}/run-preview-gate.sh"
require_runner "learn mode" "${script_dir}/run-production-learnharness.sh"
require_runner "LearnHarness backup" "${script_dir}/backup-learnharness.sh"

readonly units=(
    gigachad-bot.service
    gigachad-bot-pi-host.service
    gigachad-bot-omp-web.service
    gigachad-bot-omp-web-host.service
    gigachad-bot-omp-web-gate.service
    gigachad-bot-factory.service
    gigachad-bot-preview-gate.service
    gigachad-bot-litellm-proxy.service
    gigachad-bot-learnharness.service
    gigachad-bot-learnharness-gate.service
)
# Started by its timer, not enabled itself.
readonly timer_units=(gigachad-bot-learnharness-backup.service gigachad-bot-learnharness-backup.timer)
# Units an earlier version installed: the OMP model source (its runner is gone, pi-host replaced it).
readonly retired_units=(gigachad-bot-omp.service)
for unit in "${retired_units[@]}"; do
    if [[ -e "${unit_dir}/${unit}" ]]; then
        systemctl --user disable --now "$unit" || true
        rm -f "${unit_dir}/${unit}"
        printf 'Removed the retired %s\n' "$unit"
    fi
done
for unit in "${units[@]}" "${timer_units[@]}"; do
    install -D --mode=0644 "${script_dir}/${unit}" "${unit_dir}/${unit}"
done
systemctl --user daemon-reload
systemctl --user enable "${units[@]}" gigachad-bot-learnharness-backup.timer
printf 'User services installed and enabled. Start them with: systemctl --user start %s gigachad-bot-learnharness-backup.timer\n' \
    "${units[*]}"
