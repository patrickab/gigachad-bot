#!/usr/bin/env bash
# Install the user service from a production worktree. It never starts it.
set -euo pipefail

readonly unit_name='gigachad-bot.service'
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly worktree="$(cd -- "$script_dir/.." && pwd)"
readonly unit_source="${script_dir}/${unit_name}"
readonly unit_destination="${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user/${unit_name}"
readonly backend="${worktree}/.venv/bin/uvicorn"
readonly backend_runner="${script_dir}/run-production-backend.sh"

if [[ ! -x $backend ]]; then
    printf 'Production backend executable is missing: %s\nRun: cd %s && uv sync --python 3.12 --no-dev\n' "$backend" "$worktree" >&2
    exit 1
fi
if [[ ! -x $backend_runner ]]; then
    printf 'Production backend runner is missing or not executable: %s\n' "$backend_runner" >&2
    exit 1
fi

install -D --mode=0644 "$unit_source" "$unit_destination"
systemctl --user daemon-reload
systemctl --user enable "$unit_name"
printf 'User service installed and enabled. Start it with: systemctl --user start %s\n' "$unit_name"
