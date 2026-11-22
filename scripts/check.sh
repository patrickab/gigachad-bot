#!/usr/bin/env bash
# Moved to factory/gates.sh. Kept for factory tasks started before the move: their `factory.yaml` still names this path.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../factory/gates.sh" "$@"
