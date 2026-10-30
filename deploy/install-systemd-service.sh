#!/usr/bin/env bash
# Install the user services from a production worktree. It never starts them.
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
require_runner OMP "${script_dir}/run-production-omp.sh"
require_runner omp-web "${script_dir}/run-production-omp-web.sh"
require_runner factory "${worktree}/run-factory.sh"
require_runner "learn mode" "${script_dir}/run-production-learnharness.sh"
require_runner "LearnHarness backup" "${script_dir}/backup-learnharness.sh"

readonly units=(
    gigachad-bot.service
    gigachad-bot-omp.service
    gigachad-bot-omp-web.service
    gigachad-bot-omp-web-gate.service
    gigachad-bot-factory.service
    gigachad-bot-litellm-proxy.service
    gigachad-bot-learnharness.service
    gigachad-bot-learnharness-gate.service
)
# Started by its timer, not enabled itself.
readonly timer_units=(gigachad-bot-learnharness-backup.service gigachad-bot-learnharness-backup.timer)
for unit in "${units[@]}" "${timer_units[@]}"; do
    install -D --mode=0644 "${script_dir}/${unit}" "${unit_dir}/${unit}"
done
systemctl --user daemon-reload
systemctl --user enable "${units[@]}" gigachad-bot-learnharness-backup.timer
printf 'User services installed and enabled. Start them with: systemctl --user start %s gigachad-bot-learnharness-backup.timer\n' \
    "${units[*]}"
