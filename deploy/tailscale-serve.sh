#!/usr/bin/env bash
# Configures Tailnet-only HTTPS ingress; it never makes the backend public.
set -euo pipefail

if ! command -v tailscale >/dev/null 2>&1; then
    printf 'error: tailscale CLI is required; install it and join this node to the Tailnet first.\n' >&2
    exit 127
fi

if (($# < 1 || $# > 2)); then
    printf 'usage: %s <local-port> [https-port]\n' "${BASH_SOURCE[0]}" >&2
    printf '  backend: %s 8002, omp-web gate: %s 30179 8443, LearnHarness gate: %s 30182 8444\n' \
        "${BASH_SOURCE[0]}" "${BASH_SOURCE[0]}" "${BASH_SOURCE[0]}" >&2
    exit 2
fi

# --bg persists the named HTTPS listener and replaces its handler on repeated runs.
# Tailscale Serve rejects unavailable or unauthenticated daemons. Any HTTPS port works for Serve, only Funnel (never used
# here) is limited to 443, 8443 and 10000.
# --accept-app-caps (Tailscale v1.92+) forwards the host grant as `Tailscale-App-Capabilities`, see backend/identity.py.
exec tailscale serve --bg --https="${2:-443}" --accept-app-caps=gigachad.local/cap/host "http://127.0.0.1:${1}"
