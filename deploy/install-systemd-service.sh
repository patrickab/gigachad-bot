#!/usr/bin/env bash
# Run manually as root after installing the backend in /opt/gigachad-bot.
# This performs privileged account, /etc, and systemd changes; it never starts the service.
set -euo pipefail

readonly unit_name='gigachad-bot.service'
readonly unit_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${unit_name}"
readonly unit_destination="/etc/systemd/system/${unit_name}"
readonly environment_file='/etc/gigachad-bot/gigachad-bot.env'
readonly backend='/opt/gigachad-bot/.venv/bin/uvicorn'

if (( EUID != 0 )); then
    printf 'Run this installer as root.\n' >&2
    exit 1
fi

if [[ ! -x "$backend" ]]; then
    printf 'Backend executable is missing: %s\n' "$backend" >&2
    exit 1
fi

getent group gigachad >/dev/null || groupadd --system gigachad
id --user gigachad >/dev/null 2>&1 || useradd --system --gid gigachad --home-dir /var/lib/gigachad-bot --shell /usr/bin/nologin gigachad

install --directory --owner=root --group=root --mode=0750 /etc/gigachad-bot
if [[ ! -e "$environment_file" ]]; then
    install --owner=root --group=root --mode=0600 /dev/null "$environment_file"
else
    chown root:root "$environment_file"
    chmod 0600 "$environment_file"
fi
install --owner=root --group=root --mode=0644 "$unit_source" "$unit_destination"
systemctl daemon-reload
systemctl enable "$unit_name"

printf 'Set GIGACHAD_CORS_ORIGINS to the Vercel origin(s) in %s, then run: systemctl start %s\n' "$environment_file" "$unit_name"
