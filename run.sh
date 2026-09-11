#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
backend_args=()

case "$#" in
    0) ;;
    1)
        if [[ "$1" == "--tailscale" ]]; then
            backend_args=("--tailscale")
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

backend_pid=''
frontend_pid=''
cleaned_up=false

cleanup() {
    if [[ "$cleaned_up" == true ]]; then
        return
    fi
    cleaned_up=true

    printf 'Shutting down...\n'
    for pid in "$backend_pid" "$frontend_pid"; do
        if [[ -n "$pid" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f "mineru.cli.fast_api" 2>/dev/null || true
    docker compose -f "$root_dir/docker-compose.vane.yml" down 2>/dev/null || true
    for pid in "$backend_pid" "$frontend_pid"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    printf 'Done.\n'
}

trap cleanup EXIT
trap 'exit 0' INT TERM

"$root_dir/run-backend.sh" "${backend_args[@]}" &
backend_pid=$!

"$root_dir/run-frontend.sh" &
frontend_pid=$!

wait -n "$backend_pid" "$frontend_pid"
