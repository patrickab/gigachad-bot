#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! [[ $# -eq 0 || ($# -eq 1 && "$1" == --tailscale) ]]; then
    printf 'usage: %s [--tailscale]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
source "$root_dir/deploy/lib.sh"

load_secrets
if [[ -z "${POSTGRES_PASSWORD_DEV:-}" ]]; then
    "$root_dir/scripts/setup-postgres.sh"
    load_secrets
    if [[ -z "${POSTGRES_PASSWORD_DEV:-}" ]]; then
        printf 'Startup cancelled: development database credentials are missing.\n' >&2
        exit 1
    fi
fi
backend_pid=''
frontend_pid=''
# Optional services, kept separate because only the backend and frontend end the run.
service_pids=()
cleaned_up=false

cleanup() {
    if [[ "$cleaned_up" == true ]]; then
        return
    fi
    cleaned_up=true

    printf 'Shutting down...\n'
    for pid in "$backend_pid" "$frontend_pid" "${service_pids[@]}"; do
        if [[ -n "$pid" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f "mineru.cli.fast_api" 2>/dev/null || true
    for pid in "$backend_pid" "$frontend_pid" "${service_pids[@]}"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    printf 'Done.\n'
}

trap cleanup EXIT
trap 'exit 0' INT TERM

# Install sandbox-only Python packages into the shared venv before OMP starts. The
# scripts started below skip their own sync, which would race this one.
sync_python_deps --extra sandbox-plot
export GIGACHAD_DEPS_SYNCED=1

# `npm ci` rewrites src/frontend/node_modules: finish it before anything packs chat-ui with its tsc and before the
# backend's reloader starts watching. An unchanged install is kept, so this is instant on most starts.
printf 'Installing frontend deps\n'
install_frontend_deps
export GIGACHAD_FRONTEND_DEPS_INSTALLED=1

# Opt out with GIGACHAD_OMP=0. The script exits quietly when OMP is absent,
# and the backend hides the OMP model source when its gateway is unreachable.
if [[ "${GIGACHAD_OMP:-1}" != "0" ]]; then
    "$root_dir/run-omp.sh" &
    service_pids+=($!)
fi

# Opt out with GIGACHAD_LEARNHARNESS=0. Serves learn mode (LearnHarness, started below with omp-web) and the
# LiteLLM proxy its models go through, each reusing an instance already on its port.
if [[ "${GIGACHAD_LEARNHARNESS:-1}" != "0" ]]; then
    "$root_dir/run-litellm-proxy.sh" &
    service_pids+=($!)
fi

# Opt out with GIGACHAD_FACTORY=0. Serves the Factory board's engine, reusing an
# instance already on its port. It waits for the PostgreSQL run-backend.sh starts.
if [[ "${GIGACHAD_FACTORY:-1}" != "0" ]]; then
    "$root_dir/run-factory.sh" &
    service_pids+=($!)
fi

"$root_dir/run-backend.sh" "$@" &
backend_pid=$!

"$root_dir/run-frontend.sh" &
frontend_pid=$!

# omp-web and LearnHarness both install the same @gigachad/chat-ui tarball: pack it once, while the services above
# start. Without it (a failed pack) each launcher packs its own and reports the failure.
if [[ "${GIGACHAD_OMP_WEB:-1}" != "0" || "${GIGACHAD_LEARNHARNESS:-1}" != "0" ]]; then
    chat_ui_dir="$(mktemp -d)"
    if GIGACHAD_CHAT_UI_TARBALL="$(PATH="$root_dir/.venv/bin:$PATH" pack_chat_ui "$chat_ui_dir")"; then
        export GIGACHAD_CHAT_UI_TARBALL
    else
        unset GIGACHAD_CHAT_UI_TARBALL
        printf 'warning: packing @gigachad/chat-ui failed, omp-web and LearnHarness pack it themselves\n' >&2
    fi
fi

# Opt out with GIGACHAD_OMP_WEB=0. Serves the omp tabs' chat and settings
# (omp-web), reusing an instance already on its port.
if [[ "${GIGACHAD_OMP_WEB:-1}" != "0" ]]; then
    "$root_dir/run-omp-web.sh" &
    service_pids+=($!)
fi

# LearnHarness runs as a sandboxed systemd user unit: the TERM cleanup sends its
# launcher stops the unit it started (a reused instance is left alone).
if [[ "${GIGACHAD_LEARNHARNESS:-1}" != "0" ]]; then
    "$root_dir/run-learnharness.sh" &
    service_pids+=($!)
fi

wait -n "$backend_pid" "$frontend_pid"
