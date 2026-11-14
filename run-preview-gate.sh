#!/usr/bin/env bash
# Serves the factory preview gate on 127.0.0.1:30185, the one address a browser reaches every factory preview through
# (src/backend/preview_gate.py).
#
#   (no argument)  development: the local dev login passes, like the backend run-backend.sh starts
#   --prod         only Tailnet logins granted the host capability pass, Tailscale Serve's :8445 points here
#
# Idempotent: a gate already listening on the port is reused.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir/deploy/lib.sh"
port=30185

case "${1:-}" in
    '')
        export GIGACHAD_ENV=development
        export GIGACHAD_DEV_USER="$(whoami)"
        ;;
    --prod) export GIGACHAD_ENV=production ;;
    *)
        printf 'usage: %s [--prod]\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac

if port_open "127.0.0.1:$port"; then
    printf 'Reusing the preview gate already listening on 127.0.0.1:%s\n' "$port"
    exit 0
fi
exec "$root_dir/.venv/bin/uvicorn" backend.preview_gate:app --app-dir "$root_dir/src" --host 127.0.0.1 --port "$port" --no-access-log
