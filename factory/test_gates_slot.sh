#!/usr/bin/env bash
# Shows the command of gates.sh's test steps with and without `factory slot`, and with FACTORY_SLOT_HELD set.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bin="$(mktemp -d)"; export TMPDIR="$bin"; trap 'rm -rf "$bin"' EXIT
printf '#!/bin/sh\n[ "$1" = --help ] && echo "usage: factory <action> | factory slot -- <command...>"\n' >"$bin/factory"; chmod +x "$bin/factory"
run() { rm -f "$bin/gates-dry-run"; "$@" factory/gates.sh --dry-run pi-host >/dev/null; cat "$bin/gates-dry-run"; }
expect() { [[ $2 == "$1" ]] || { printf 'FAIL: want [%s] got [%s]\n' "$1" "$2"; exit 1; }; }
expect 'factory slot -- bun test' "$(PATH="$bin:$PATH" run env)"
expect 'bun test' "$(PATH="$bin:$PATH" run env FACTORY_SLOT_HELD=1)"
printf '#!/bin/sh\nexit 1\n' >"$bin/factory"
expect 'bun test' "$(PATH="$bin:$PATH" run env)"
echo ok
