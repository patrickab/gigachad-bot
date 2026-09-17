#!/usr/bin/env bash
# Install the user services from a production worktree. It never starts them.
set -euo pipefail

readonly backend_unit='gigachad-bot.service'
readonly omp_unit='gigachad-bot-omp.service'
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly worktree="$(cd -- "$script_dir/.." && pwd)"
readonly unit_dir="${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user"
readonly backend="${worktree}/.venv/bin/uvicorn"
readonly backend_runner="${script_dir}/run-production-backend.sh"
readonly omp_runner="${script_dir}/run-production-omp.sh"

if [[ ! -x $backend ]]; then
    printf 'Production backend executable is missing: %s\nRun: cd %s && uv sync --python 3.12 --no-dev\n' "$backend" "$worktree" >&2
    exit 1
fi
if [[ ! -x $backend_runner ]]; then
    printf 'Production backend runner is missing or not executable: %s\n' "$backend_runner" >&2
    exit 1
fi
if [[ ! -x $omp_runner ]]; then
    printf 'Production OMP runner is missing or not executable: %s\n' "$omp_runner" >&2
    exit 1
fi

install -D --mode=0644 "${script_dir}/${backend_unit}" "${unit_dir}/${backend_unit}"
install -D --mode=0644 "${script_dir}/${omp_unit}" "${unit_dir}/${omp_unit}"
systemctl --user daemon-reload
systemctl --user enable "$backend_unit" "$omp_unit"
printf 'User services installed and enabled. Start them with: systemctl --user start %s %s\n' "$backend_unit" "$omp_unit"
