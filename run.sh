#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($# > 1))
then
    printf 'usage: %s [--tailscale]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
if (($# == 1)) && [[ "$1" != "--tailscale" ]]
then
    printf 'usage: %s [--tailscale]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
backend_pid=''
frontend_pid=''
omp_pid=''
cleaned_up=false

cleanup() {
    if [[ "$cleaned_up" == true ]]; then
        return
    fi
    cleaned_up=true

    printf 'Shutting down...\n'
    for pid in "$backend_pid" "$frontend_pid" "$omp_pid"; do
        if [[ -n "$pid" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f "mineru.cli.fast_api" 2>/dev/null || true
    for pid in "$backend_pid" "$frontend_pid" "$omp_pid"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    printf 'Done.\n'
}

trap cleanup EXIT
trap 'exit 0' INT TERM

# Opt out with GIGACHAD_OMP=0. The script exits quietly when OMP is absent,
# and the backend hides the OMP model source when its gateway is unreachable.
if [[ "${GIGACHAD_OMP:-1}" != "0" ]]; then
    "$root_dir/run-omp.sh" &
    omp_pid=$!
fi

"$root_dir/run-backend.sh" "$@" &
backend_pid=$!

"$root_dir/run-frontend.sh" &
frontend_pid=$!

wait -n "$backend_pid" "$frontend_pid"
