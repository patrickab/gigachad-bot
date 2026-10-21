#!/usr/bin/env bash
# Install the user services from a production worktree. It never starts them.
set -euo pipefail

readonly backend_unit='gigachad-bot.service'
readonly omp_unit='gigachad-bot-omp.service'
readonly omp_web_unit='gigachad-bot-omp-web.service'
readonly omp_web_gate_unit='gigachad-bot-omp-web-gate.service'
readonly factory_unit='gigachad-bot-factory.service'
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly worktree="$(cd -- "$script_dir/.." && pwd)"
readonly unit_dir="${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user"
readonly backend="${worktree}/.venv/bin/uvicorn"
readonly backend_runner="${worktree}/run-backend.sh"
readonly omp_runner="${script_dir}/run-production-omp.sh"
readonly omp_web_runner="${script_dir}/run-production-omp-web.sh"
readonly factory_runner="${worktree}/run-factory.sh"

if [[ ! -x $backend ]]; then
    printf 'Production backend executable is missing: %s\nRun: cd %s && uv sync --python 3.12 --locked --no-dev\n' "$backend" "$worktree" >&2
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
if [[ ! -x $omp_web_runner ]]; then
    printf 'Production omp-web runner is missing or not executable: %s\n' "$omp_web_runner" >&2
    exit 1
fi
if [[ ! -x $factory_runner ]]; then
    printf 'Production factory runner is missing or not executable: %s\n' "$factory_runner" >&2
    exit 1
fi

readonly units=("$backend_unit" "$omp_unit" "$omp_web_unit" "$omp_web_gate_unit" "$factory_unit")
for unit in "${units[@]}"; do
    install -D --mode=0644 "${script_dir}/${unit}" "${unit_dir}/${unit}"
done
systemctl --user daemon-reload
systemctl --user enable "${units[@]}"
printf 'User services installed and enabled. Start them with: systemctl --user start %s\n' "${units[*]}"
