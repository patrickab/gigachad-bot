#!/usr/bin/env bash
# Starts pi-host (pi-host/): the Pi session supervisor, completion gateway and usage service the backend talks to.
#
# Loopback only, bearer token in ${XDG_STATE_HOME:-~/.local/state}/gigachad-bot/pi-host.token (pi-host creates it).
# Idempotent: a pi-host already listening on its port is reused rather than restarted, so a second gigachad
# instance does not collide. Only a pi-host this script started is stopped on exit; its Pi children get SIGTERM
# and persist their sessions.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir/deploy/lib.sh"

bind="${PI_HOST_BIND_HOST:-127.0.0.1}:${PI_HOST_PORT:-8300}"
pi_host_dir="$root_dir/pi-host"
# Sandboxed sessions run through the agent-sandbox CLI of this checkout's venv (the pinned Agents-in-a-box)
export PI_HOST_AGENT_SANDBOX_BIN="${PI_HOST_AGENT_SANDBOX_BIN:-$root_dir/.venv/bin/agent-sandbox}"
owned_pid=''

if ! command -v bun >/dev/null 2>&1; then
    printf 'bun not on PATH; skipping pi-host\n' >&2
    exit 0
fi

cleanup() {
    if [[ -n "$owned_pid" ]]; then
        kill -TERM "$owned_pid" 2>/dev/null || true
        wait "$owned_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 0' INT TERM

if port_open "$bind"; then
    printf 'Reusing the pi-host already listening on %s\n' "$bind"
    exit 0
fi

# Install when node_modules is missing or older than the manifest or lockfile.
if [[ ! -d "$pi_host_dir/node_modules" || "$pi_host_dir/package.json" -nt "$pi_host_dir/node_modules" \
    || "$pi_host_dir/bun.lock" -nt "$pi_host_dir/node_modules" ]]; then
    printf 'Installing pi-host deps\n'
    (cd -- "$pi_host_dir" && bun install --frozen-lockfile)
    touch "$pi_host_dir/node_modules"
fi

# True once `bind` answers or the child has exited; an exit sets `child_exited`.
pi_host_settled() {
    port_open "$bind" && return 0
    kill -0 "$owned_pid" 2>/dev/null && return 1
    child_exited=true
}

printf 'Starting pi-host on %s\n' "$bind"
# Plain `bun`, not `bun run`: that puts pi-host/node_modules/.bin first on PATH, so Pi sessions would start the npm
# package's `pi` (cli.js) instead of the installed binary, and their shells would see pi-host's tools
(cd -- "$pi_host_dir" && exec bun src/server.ts) &
owned_pid=$!
child_exited=false
if ! wait_until 120 0.25 pi_host_settled; then
    # The EXIT trap stops the unresponsive child.
    printf 'pi-host did not answer on %s\n' "$bind" >&2
    exit 1
fi
if [[ "$child_exited" == true ]]; then
    wait "$owned_pid" 2>/dev/null || true
    owned_pid=''
    # A racing instance may have won the port while this child died.
    if port_open "$bind"; then
        printf 'pi-host lost a startup race; reusing the winner on %s\n' "$bind" >&2
        exit 0
    fi
    printf 'pi-host exited during startup\n' >&2
    exit 1
fi

wait "$owned_pid"
