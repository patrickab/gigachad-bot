#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_ROOT"
source "$REPO_ROOT/deploy/load-env.sh"

echo "Installing frontend deps"
uv run gigachad-install

echo "Starting frontend on http://127.0.0.1:2999"
exec .venv/bin/npm --prefix src/frontend run dev
