#!/usr/bin/env bash
# The test pipeline: one entry for local runs, the factory's gates (factory.yaml) and checks before a redeploy.
#
#   scripts/check.sh                      every step
#   scripts/check.sh STEP...              some of: lint types python frontend pi-host
#   scripts/check.sh --changed [STEP...]  only the steps (and frontend tests) the change since the target branch reaches
#
# - Hermetic: the steps run under `env -i` with only HOME, USER, PATH, LANG, TERM and TMPDIR, so nothing a parent
#   exports (run.sh's GIGACHAD_*, the production unit, a Pi session, the factory engine) changes a result
# - Steps run in parallel (vitest on half the cores then), the heaviest (vitest, bun) in one of 2 slots per machine
#   (`in_slot`), shared by every check that runs at the time. Each step's output is printed whole once all ended,
#   failed steps last (the factory feeds the end of the output back to the coder)
# - Exit 0 passed, 1 a step failed, 75 (EX_TEMPFAIL) the environment is broken (missing tools or packages, an empty
#   submodule): the factory blocks the task as `infra` instead of asking the coder to fix it. Last line: the reason
# - --changed: the change is the diff from the merge base with the target branch (local `main`, else `master`) to the
#   working tree, untracked files included. Paths map to steps (`select_steps`); any path no rule knows, and a missing
#   merge base, run every step. vitest then runs only the tests the changed files reach (`vitest --changed`), unless a
#   package or config file changed
# - pytest runs on 3 workers, each with a database of its own. Flaky tests: failed pytest tests run once more, alone
#   (`--last-failed`), vitest retries a failed test once. A test that
#   passes on the rerun passes the step, the output says it was flaky
# - Postgres-backed tests run against a throwaway database (`gigachad_check_<random>`, dropped on exit) on the
#   development PostgreSQL, else the production one, whichever is up with its password in ~/.secrets. Without either
#   they skip and the run says so. POSTGRES_TEST_DATABASE_URL, when set, is used as is
# - A worktree uses its main checkout's .venv (factory worktrees have none). Missing node_modules (a worktree whose
#   copy is still running) are waited for, an empty submodule is copied from the main checkout
# - lint checks Python files changed against the merge base with origin/master (the tree predates ruff's rules) and
#   all frontend files with eslint
# - pi-extensions/ has no packages of its own: types and pi-host cover it with pi-host's node_modules
#   (pi-extensions/tsconfig.json)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
all_steps=(lint types python frontend pi-host)
INFRA=75

if [[ -z ${GIGACHAD_CHECK_CLEAN:-} ]]; then
    exec env -i \
        GIGACHAD_CHECK_CLEAN=1 HOME="$HOME" USER="${USER:-$(id -un)}" LANG="${LANG:-C.UTF-8}" TERM="${TERM:-dumb}" \
        TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
        ${POSTGRES_TEST_DATABASE_URL:+POSTGRES_TEST_DATABASE_URL="$POSTGRES_TEST_DATABASE_URL"} \
        bash "${BASH_SOURCE[0]}" "$@"
fi

cd -- "$root"
source deploy/lib.sh

changed_mode=0
if [[ ${1:-} == --changed ]]; then
    changed_mode=1
    shift
fi
steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && steps=("${all_steps[@]}")
for step in "${steps[@]}"; do
    [[ " ${all_steps[*]} " == *" $step "* ]] || { printf 'unknown step %s (one of: %s)\n' "$step" "${all_steps[*]}" >&2; exit 2; }
done

primary="$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd)")"

# The merge base with the target branch, empty when there is none.
change_base() {
    local branch
    for branch in main master; do
        if git show-ref --verify --quiet "refs/heads/$branch"; then
            git merge-base HEAD "refs/heads/$branch" 2>/dev/null || true
            return
        fi
    done
}

# Paths changed since $1, the working tree and untracked files included.
changed_paths() {
    { git diff --name-only --no-renames "$1"; git ls-files --others --exclude-standard; } | sort -u
}

# The steps the changed paths on stdin reach. A path no rule knows reaches every step.
select_steps() {
    local path wanted=""
    while IFS= read -r path; do
        case "$path" in
            .docs/*|docs/*|README.md|CONTEXT.md|LICENSE) ;;
            src/frontend/*) wanted+=" lint types frontend" ;;
            pi-host/*|pi-extensions/*) wanted+=" types pi-host" ;;
            *) wanted+=" ${all_steps[*]}"; break ;;
        esac
    done
    printf '%s\n' "$wanted"
}

base=""
frontend_all=1
if [[ $changed_mode -eq 1 ]]; then
    base="$(change_base)"
    if [[ -n $base ]]; then
        paths="$(changed_paths "$base")"
        wanted="$(select_steps <<<"$paths")"
        selected=()
        for step in "${steps[@]}"; do
            [[ " $wanted " == *" $step "* ]] && selected+=("$step")
        done
        steps=("${selected[@]}")
        # vitest follows imports, not packages or config: a change to those runs every frontend test.
        grep -qE '^src/frontend/([^/]+\.(json|ts|mjs|js)|packages/)' <<<"$paths" || frontend_all=0
    fi
    if [[ ${#steps[@]} -eq 0 ]]; then
        printf 'no step reaches the change\n'
        exit 0
    fi
fi

infra() {
    printf '%s\n' "$1"
    return "$INFRA"
}

# Waits up to 5 minutes for a node_modules a worktree is still copying (`agent-worktree create --background`).
need_node_modules() {
    local dir="$1" tries=0
    while [[ ! -d $dir/node_modules ]]; do
        ((tries++ < 60)) || { infra "$dir/node_modules is missing, install the packages in the main checkout"; return; }
        sleep 5
    done
}

# An empty submodule in a worktree gets the files of the main checkout's copy, as `agent-worktree create` does.
fill_submodules() {
    local path
    while read -r _ _ _ path; do
        [[ -n $(ls "$path" 2>/dev/null) ]] && continue  # dotfiles alone (a bare .git) are empty
        if [[ $root != "$primary" && -n $(ls "$primary/$path" 2>/dev/null) ]]; then
            mkdir -p "$path"
            find "$primary/$path" -mindepth 1 -maxdepth 1 ! -name .git -exec cp -a --reflink=auto -t "$path" {} +
        else
            infra "submodule $path is empty, run git submodule update --init in the main checkout"
            return
        fi
    done < <(git ls-files -s | awk '$1 == "160000"')
}

fill_submodules || exit "$INFRA"

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

# Runs "$@" in one of $2 machine-wide slots named $1, waiting for one to be free: the heaviest steps of every check
# running on this machine (four factory tasks side by side) share the cores instead of starving each other.
in_slot() {
    local name="$1" slots="$2" index fd
    shift 2
    while true; do
        for ((index = 1; index <= slots; index++)); do
            exec {fd}>"${TMPDIR:-/tmp}/gigachad-check-$name-$index.lock"
            if flock -n "$fd"; then
                "$@"
                local status=$?
                exec {fd}>&-
                return "$status"
            fi
            exec {fd}>&-
        done
        sleep 1
    done
}

step_lint() {
    local base files=()
    need_node_modules src/frontend || return
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
    need_node_modules src/frontend || return
    need_node_modules pi-host || return
    (cd src/frontend && node_modules/.bin/tsc --noEmit -p .) || status=1
    (cd pi-host && node_modules/.bin/tsc --noEmit) || status=1
    (cd pi-host && node_modules/.bin/tsc --noEmit -p ../pi-extensions) || status=1
    return "$status"
}

step_python() {
    local url="${POSTGRES_TEST_DATABASE_URL:-}" status=0 cache
    if [[ -z $url ]]; then
        url="$(throwaway_database)"
        if [[ -n $url ]]; then
            check_database="$url"
            trap 'drop_database "$check_database"' EXIT
        else
            printf 'note: no PostgreSQL is up, the Postgres-backed tests skip\n' >&2
        fi
    fi
    cache="$(mktemp -d)"
    # Three workers, each on a database of its own (tests/conftest.py): fixed, not one per core, so four test suites
    # side by side (four factory tasks) share 8 cores without starving each other.
    POSTGRES_TEST_DATABASE_URL="$url" "$python" -m pytest -q -o cache_dir="$cache" -n 3 --dist loadfile || status=$?
    # Exit 1 is failed tests: those run once more, alone. Anything else (collection, usage) is not flaky.
    if [[ $status -eq 1 ]]; then
        printf '\n-- rerunning the failed tests once\n'
        status=0
        POSTGRES_TEST_DATABASE_URL="$url" "$python" -m pytest -q -o cache_dir="$cache" --last-failed || status=$?
        [[ $status -eq 0 ]] && printf 'flaky: the failed tests passed on the rerun\n'
    fi
    rm -rf "$cache"
    return "$status"
}

step_frontend() {
    local args=(run --retry=1)
    need_node_modules src/frontend || return
    [[ $frontend_all -eq 0 ]] && args+=(--changed "$base" --passWithNoTests)
    # Beside other steps, vitest gets half the cores: oversubscribed, its tests hit their timeouts.
    [[ ${#steps[@]} -gt 1 ]] && args+=(--maxWorkers=50%)
    # A test that is slow on a loaded machine is not failing: 15 s instead of vitest's 5 s.
    [[ ${#steps[@]} -gt 1 ]] && args+=(--testTimeout=15000)
    in_slot vitest 2 bash -c 'cd src/frontend && exec node_modules/.bin/vitest "$@"' vitest "${args[@]}"
}

step_pi_host() {
    local status=0
    need_node_modules pi-host || return
    in_slot bun 2 bash -c 'status=0; (cd pi-host && bun test) || status=1; (cd pi-extensions && bun test) || status=1; exit $status' || status=1
    return "$status"
}

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
declare -A pids
for step in "${steps[@]}"; do
    ("step_${step//-/_}") >"$logs/$step" 2>&1 &
    pids[$step]=$!
done
failed=() broken=() reasons=()
for step in "${steps[@]}"; do
    code=0
    wait "${pids[$step]}" || code=$?
    if [[ $code -eq $INFRA ]]; then
        broken+=("$step")
        reasons+=("$(tail -n 1 "$logs/$step")")
    elif [[ $code -ne 0 ]]; then
        failed+=("$step")
    fi
done
for step in "${steps[@]}"; do
    [[ " ${failed[*]} ${broken[*]} " == *" $step "* ]] || printf '\n== %s\n%s\n' "$step" "$(cat "$logs/$step")"
done
for step in "${failed[@]}" "${broken[@]}"; do
    printf '\n== %s\n%s\n' "$step" "$(cat "$logs/$step")"
done
if [[ ${#broken[@]} -gt 0 ]]; then
    printf '\nbroken environment: %s\n%s\n' "${broken[*]}" "${reasons[-1]}" >&2
    exit "$INFRA"
fi

if [[ ${#failed[@]} -gt 0 ]]; then
    printf '\nfailed: %s\n' "${failed[*]}" >&2
    exit 1
fi
printf '\nall passed: %s\n' "${steps[*]}"
