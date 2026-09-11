#!/usr/bin/env bash
# Run manually as root after preparing the noob production worktree and environment file.
# This performs only privileged systemd installation steps; it never starts the service.
set -euo pipefail

readonly unit_name='gigachad-bot.service'
readonly unit_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${unit_name}"
readonly unit_destination="/etc/systemd/system/${unit_name}"
readonly release_worktree='/home/noob/git/gigachad-bot-prod'
readonly environment_file='/home/noob/.config/gigachad-bot/env'
readonly backend="${release_worktree}/.venv/bin/uvicorn"
readonly backend_runner="${release_worktree}/deploy/run-production-backend.sh"

if (( EUID != 0 )); then
    printf 'Run this installer as root.\n' >&2
    exit 1
fi

if [[ ! -x "$backend" ]]; then
    printf 'Production backend executable is missing: %s\n' "$backend" >&2
    printf 'Create %s as noob and run: uv sync --python 3.12 --no-dev\n' "$release_worktree" >&2
    exit 1
fi

if [[ ! -x "$backend_runner" ]]; then
    printf 'Production backend runner is missing or not executable: %s\n' "$backend_runner" >&2
    printf 'Update %s as noob so it includes the executable deployment runner.\n' "$release_worktree" >&2
    exit 1
fi
install --owner=root --group=root --mode=0644 "$unit_source" "$unit_destination"
systemctl daemon-reload
systemctl enable "$unit_name"

if [[ ! -f "$environment_file" ]]; then
    printf 'Before starting, create %s with the required strict KEY=value runtime settings; the runner loads it directly.\n' "$environment_file" >&2
else
    printf 'The service is installed and enabled. Start it when the runtime settings are ready: systemctl start %s\n' "$unit_name"
fi
