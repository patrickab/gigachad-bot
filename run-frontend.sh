#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"
source "$REPO_ROOT/deploy/lib.sh"
source "$REPO_ROOT/deploy/load-env.sh"

echo "Installing frontend deps"
sync_python_deps
install_frontend_deps

echo "Starting frontend on http://127.0.0.1:2999"
# `next dev` restarts itself, and with it reloads the open page, once its heap is 80% full. After the first compile
# about 3 GB are in use, so a 4 GB heap filled within half an hour of agent edits.
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=6144}"
exec .venv/bin/npm --prefix src/frontend run dev
