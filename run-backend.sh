#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tailscale=false

case "$#" in
    0) ;;
    1)
        if [[ "$1" == "--tailscale" ]]; then
            tailscale=true
        else
            printf 'usage: %s [--tailscale]\n' "${BASH_SOURCE[0]}" >&2
            exit 2
        fi
        ;;
    *)
        printf 'usage: %s [--tailscale]\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac

cd "$root_dir"
source "$root_dir/deploy/load-env.sh"

printf 'Installing backend deps\n'
uv sync
export PATH="$root_dir/.venv/bin:$PATH"

if [[ "$tailscale" == true ]]; then
    printf 'Configuring Tailnet-only HTTPS ingress\n'
    "$root_dir/deploy/tailscale-serve.sh"
fi

printf 'Starting backend on http://127.0.0.1:8001\n'
exec uvicorn src.backend.server:app --host 127.0.0.1 --port 8001 --reload --reload-dir src
