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
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}"
exec .venv/bin/npm --prefix src/frontend run dev
