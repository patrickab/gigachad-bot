#!/usr/bin/env bash
# The test pipeline: one entry for local runs, the factory's gates (factory.yaml) and checks before a redeploy.
#
#   scripts/check.sh            every step
#   scripts/check.sh STEP...    some of: lint types python frontend pi-host
#
# - Hermetic: the steps run under `env -i` with only HOME, USER, PATH, LANG, TERM and TMPDIR, so nothing a parent
#   exports (run.sh's GIGACHAD_*, the production unit, a Pi session, the factory engine) changes a result
# - Postgres-backed tests run against a throwaway database (`gigachad_check_<random>`, dropped on exit) on the
#   development PostgreSQL, else the production one, whichever is up with its password in ~/.secrets. Without either
#   they skip and the run says so. POSTGRES_TEST_DATABASE_URL, when set, is used as is
# - A worktree uses its main checkout's .venv (factory worktrees have none), node_modules must be installed
# - lint checks Python files changed against the merge base with origin/master (the tree predates ruff's rules) and
#   all frontend files with eslint
# - pi-extensions/ has no packages of its own: types and pi-host cover it with pi-host's node_modules
#   (pi-extensions/tsconfig.json)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
all_steps=(lint types python frontend pi-host)

if [[ -z ${GIGACHAD_CHECK_CLEAN:-} ]]; then
    exec env -i \
        GIGACHAD_CHECK_CLEAN=1 HOME="$HOME" USER="${USER:-$(id -un)}" LANG="${LANG:-C.UTF-8}" TERM="${TERM:-dumb}" \
        TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
        ${POSTGRES_TEST_DATABASE_URL:+POSTGRES_TEST_DATABASE_URL="$POSTGRES_TEST_DATABASE_URL"} \
        bash "${BASH_SOURCE[0]}" "$@"
fi

cd -- "$root"
source deploy/lib.sh

steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && steps=("${all_steps[@]}")
for step in "${steps[@]}"; do
    [[ " ${all_steps[*]} " == *" $step "* ]] || { printf 'unknown step %s (one of: %s)\n' "$step" "${all_steps[*]}" >&2; exit 2; }
done

python_bin() {
    local common
    if [[ -x .venv/bin/python3 ]]; then
        echo "$root/.venv/bin/python3"
        return
    fi
    common="$(cd -- "$(git rev-parse --git-common-dir)" && pwd)"
    if [[ -x "$(dirname -- "$common")/.venv/bin/python3" ]]; then
        echo "$(dirname -- "$common")/.venv/bin/python3"
        return
    fi
    echo "no .venv here or in the main checkout, run uv sync there" >&2
    exit 1
}
python="$(python_bin)"

# A throwaway database for the Postgres-backed tests, its URL on stdout, nothing when no PostgreSQL is up.
throwaway_database() {
    local profile url
    load_secrets
    for profile in development production; do
        postgres_profile "$profile"
        [[ -n ${!password_var:-} ]] && port_open "$pg_host:$pg_port" || continue
        url="postgresql://${USER}:${!password_var}@${pg_host}:${pg_port}/postgres"
        CHECK_ADMIN_URL="$url" "$python" - <<'PY' && return 0
import os, secrets
import psycopg
from psycopg.conninfo import make_conninfo

admin = os.environ["CHECK_ADMIN_URL"]
name = f"gigachad_check_{secrets.token_hex(6)}"
with psycopg.connect(admin, autocommit=True) as database:
    database.execute(f'CREATE DATABASE "{name}"')
print(make_conninfo(admin, dbname=name))
PY
    done
    return 0
}

drop_database() {
    CHECK_DATABASE_URL="$1" "$python" - <<'PY' || true
import os
import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

url = os.environ["CHECK_DATABASE_URL"]
name = conninfo_to_dict(url)["dbname"]
with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as database:
    database.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
PY
}

step_lint() {
    local base files=()
    base="$(git merge-base HEAD origin/master 2>/dev/null || git rev-parse HEAD)"
    while IFS= read -r file; do
        [[ -f $file ]] && files+=("$file")
    done < <({ git diff --name-only "$base" -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)
    local status=0
    if [[ ${#files[@]} -gt 0 ]]; then
        "$python" -m ruff check "${files[@]}" || status=1
    fi
    (cd src/frontend && node_modules/.bin/eslint .) || status=1
    return "$status"
}

step_types() {
    local status=0
    (cd src/frontend && node_modules/.bin/tsc --noEmit -p .) || status=1
    (cd pi-host && node_modules/.bin/tsc --noEmit) || status=1
    (cd pi-host && node_modules/.bin/tsc --noEmit -p ../pi-extensions) || status=1
    return "$status"
}

step_python() {
    local url="${POSTGRES_TEST_DATABASE_URL:-}" status=0
    if [[ -z $url ]]; then
        url="$(throwaway_database)"
        if [[ -n $url ]]; then
            check_database="$url"
            trap 'drop_database "$check_database"' EXIT
        else
            printf 'note: no PostgreSQL is up, the Postgres-backed tests skip\n' >&2
        fi
    fi
    POSTGRES_TEST_DATABASE_URL="$url" "$python" -m pytest -q -p no:cacheprovider || status=$?
    return "$status"
}

step_frontend() {
    (cd src/frontend && node_modules/.bin/vitest run)
}

step_pi_host() {
    local status=0
    (cd pi-host && bun test) || status=1
    (cd pi-extensions && bun test) || status=1
    return "$status"
}

failed=()
for step in "${steps[@]}"; do
    printf '\n== %s\n' "$step"
    "step_${step//-/_}" || failed+=("$step")
done
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '\nfailed: %s\n' "${failed[*]}" >&2
    exit 1
fi
printf '\nall passed: %s\n' "${steps[*]}"
