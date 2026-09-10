#!/usr/bin/env bash
# Configures Tailnet-only HTTPS ingress; it never makes the backend public.
set -euo pipefail

if ! command -v tailscale >/dev/null 2>&1; then
    printf 'error: tailscale CLI is required; install it and join this node to the Tailnet first.\n' >&2
    exit 127
fi

# --bg persists the named HTTPS listener and replaces this exact port/path handler
# on repeated runs. Tailscale Serve rejects unavailable or unauthenticated daemons.
exec tailscale serve --bg --https=443 / http://127.0.0.1:8001
