#!/usr/bin/env bash
# The test pipeline: one entry for local runs, the factory's gates (factory.yaml) and checks before a redeploy.
#
#   factory/gates.sh                      every step
#   factory/gates.sh STEP...              some of: lint types python frontend pi-host
#   factory/gates.sh --changed [STEP...]  only the steps (and frontend tests) the change since the target branch reaches
#   factory/gates.sh --reuse ...          skip a step that passed on this exact tree before (the factory's gates)
#
# - Hermetic: the steps run under `env -i` with only HOME, USER, PATH, LANG, TERM and TMPDIR, so nothing a parent
#   exports (run.sh's GIGACHAD_*, the production unit, a Pi session, the factory engine) changes a result
# - Steps run in parallel (vitest on half the cores then), at low priority (`nice`): the app, its backend and pi-host
#   stay responsive while a check runs
# - Output, once all ended: a passed step is one line, a failed one the summary of its failures (`summary`: failing
#   tests with their error, type and lint errors), at most 50 lines, and the path of its full log, kept in
#   `<git common dir>/check-logs/` (the last 20 runs). The factory hands this output to the coder
# - Exit 0 passed, 1 a step failed, 75 (EX_TEMPFAIL) the environment is broken (missing tools or packages, an empty
#   submodule): the factory blocks the task as `infra` instead of asking the coder to fix it. Last line: the reason
# - --changed: the change is the diff from the merge base with the target branch (the repo's default branch, origin's
#   HEAD, else local `main`, else `master`) to the working tree, untracked files included. Paths map to steps
#   (`select_steps`); any path no rule knows, and a missing merge base, run every step. Within a step only what the
#   change reaches: `types` the TypeScript projects it touches, `lint` eslint on the changed frontend files (all of them
#   when eslint's config or the packages changed), vitest the tests the changed files import (`vitest --changed`,
#   all when a package or config file changed)
# - --reuse: a step that passed records the tree it passed on (working tree with untracked files), the step and the
#   merge base in `<git common dir>/check-passed`. Run again on the same, it is skipped: a factory merge whose rebase
#   moved nothing does not repeat the lint and types its last round passed
# - pytest runs on a worker per core but one, each with a database of its own. Flaky tests: failed pytest tests run once more, alone
#   (`--last-failed`), vitest retries a failed test once. A test that
#   passes on the rerun passes the step, the output says it was flaky
# - Postgres-backed tests run against a throwaway database (`gigachad_check_<random>`, dropped on exit) on the
#   development PostgreSQL, else the production one, whichever is up with its password in ~/.secrets. Without either
#   they skip and the run says so. POSTGRES_TEST_DATABASE_URL, when set, is used as is
# - A worktree uses its main checkout's .venv (factory worktrees have none). Missing node_modules (a worktree whose
#   copy is still running) are waited for, an empty submodule is copied from the main checkout
# - lint checks Python files changed against the merge base with origin/master (the tree predates ruff's rules) and
#   the frontend with eslint
# - pi-extensions/ has no packages of its own: types and pi-host cover it with pi-host's node_modules
#   (pi-extensions/tsconfig.json)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
all_steps=(lint types python frontend pi-host)
INFRA=75

if [[ -z ${GIGACHAD_CHECK_CLEAN:-} ]]; then
    exec nice -n 10 env -i \
        GIGACHAD_CHECK_CLEAN=1 HOME="$HOME" USER="${USER:-$(id -un)}" LANG="${LANG:-C.UTF-8}" TERM="${TERM:-dumb}" \
        TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
        ${POSTGRES_TEST_DATABASE_URL:+POSTGRES_TEST_DATABASE_URL="$POSTGRES_TEST_DATABASE_URL"} \
        bash "${BASH_SOURCE[0]}" "$@"
fi

cd -- "$root"
source deploy/lib.sh

changed_mode=0
reuse=0
while [[ ${1:-} == --* ]]; do
    case "$1" in
        --changed) changed_mode=1 ;;
        --reuse) reuse=1 ;;
        *) printf 'unknown option %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done
steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && steps=("${all_steps[@]}")
for step in "${steps[@]}"; do
    [[ " ${all_steps[*]} " == *" $step "* ]] || { printf 'unknown step %s (one of: %s)\n' "$step" "${all_steps[*]}" >&2; exit 2; }
done

primary="$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd)")"

# The merge base with the target branch, empty when there is none.
change_base() {
    local branch default
    default="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null || true)"
    for branch in ${default#origin/} main master; do
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
# What `types` checks and `lint` hands eslint: everything, unless --changed narrows it below.
type_projects=(frontend pi-host pi-extensions)
eslint_files=(.)
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
        # A path outside the three projects (a step's "every step" rule) keeps every project and every file.
        if ! grep -qvE '^(src/frontend/|pi-host/|pi-extensions/|\.docs/|docs/|README\.md$|CONTEXT\.md$|LICENSE$)' <<<"$paths"; then
            type_projects=()
            grep -q '^src/frontend/' <<<"$paths" && type_projects+=(frontend)
            grep -q '^pi-host/' <<<"$paths" && type_projects+=(pi-host)
            # The extensions compile against pi-host's packages and types.
            grep -qE '^(pi-host|pi-extensions)/' <<<"$paths" && type_projects+=(pi-extensions)
            if [[ $frontend_all -eq 0 ]]; then
                eslint_files=()
                while IFS= read -r path; do
                    [[ -f $path ]] && eslint_files+=("${path#src/frontend/}")
                done < <(grep -E '^src/frontend/.*\.(ts|tsx|js|jsx|mjs|cjs)$' <<<"$paths")
            fi
        fi
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
    if [[ ${#eslint_files[@]} -gt 0 ]]; then
        (cd src/frontend && node_modules/.bin/eslint --no-warn-ignored "${eslint_files[@]}") || status=1
    fi
    return "$status"
}

step_types() {
    local status=0 project
    for project in "${type_projects[@]}"; do
        case "$project" in
            frontend) need_node_modules src/frontend || return
                (cd src/frontend && node_modules/.bin/tsc --noEmit -p .) || status=1 ;;
            pi-host) need_node_modules pi-host || return
                (cd pi-host && node_modules/.bin/tsc --noEmit) || status=1 ;;
            pi-extensions) need_node_modules pi-host || return
                (cd pi-host && node_modules/.bin/tsc --noEmit -p ../pi-extensions) || status=1 ;;
        esac
    done
    printf 'checked: %s\n' "${type_projects[*]:-nothing}"
    return "$status"
}

step_python() {
    local url="${POSTGRES_TEST_DATABASE_URL:-}" status=0 cache workers
    workers=$(($(nproc) > 2 ? $(nproc) - 1 : 1))
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
    # One worker per core but one, each on a database of its own (tests/conftest.py). The factory's merge queue runs
    # one test suite per repository at a time, so it may take the machine.
    POSTGRES_TEST_DATABASE_URL="$url" "$python" -m pytest -q -o cache_dir="$cache" -n "$workers" --dist loadfile || status=$?
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
    (cd src/frontend && node_modules/.bin/vitest "${args[@]}")
}

step_pi_host() {
    local status=0
    need_node_modules pi-host || return
    (cd pi-host && bun test) || status=1
    (cd pi-extensions && bun test) || status=1
    return "$status"
}

# The tree the working tree holds now, untracked files included, without touching the real index.
working_tree() {
    local index
    index="$(mktemp)"
    cp "$(git rev-parse --git-path index)" "$index" 2>/dev/null || true
    GIT_INDEX_FILE="$index" git add -A >/dev/null 2>&1 && GIT_INDEX_FILE="$index" git write-tree
    rm -f "$index"
}

passed_dir="$(git rev-parse --git-common-dir)/check-passed"
passed_key=""
if [[ $reuse -eq 1 ]]; then
    mkdir -p "$passed_dir"
    find "$passed_dir" -type f -mtime +7 -delete 2>/dev/null || true
    passed_key="$(working_tree)-${base:-all}"
fi

# The failures in step $1's log $2: the lines that name them, each once, at most 50. The log's end when none match.
summary() {
    local pattern
    case "$1" in
        python) pattern='^(FAILED|ERROR) |^[0-9]+ (failed|passed|error)' ;;
        frontend) pattern='^ FAIL |AssertionError|^(Type|Reference|Syntax)?Error:|^Expected|^Received|Tests +[0-9]' ;;
        types) pattern='error TS[0-9]+' ;;
        lint) pattern='^[A-Z]+[0-9]+ |^ +--> |^/.*\.(ts|tsx|js|mjs)$|^ +[0-9]+:[0-9]+ +error ' ;;
        pi-host) pattern='^\(fail\)|^error:|^Expected|^Received|^ *[0-9]+ fail$' ;;
    esac
    { grep -E "$pattern" "$2" || tail -n 30 "$2"; } | awk '!seen[$0]++' | head -n 50
}

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
kept="$(git rev-parse --git-common-dir)/check-logs/$(date +%Y%m%d-%H%M%S)-$$"
declare -A pids
for step in "${steps[@]}"; do
    if [[ -n $passed_key && -e $passed_dir/$passed_key-$step ]]; then
        printf 'passed on this tree before, skipped\n' >"$logs/$step"
        (exit 0) &
    else
        ("step_${step//-/_}") >"$logs/$step" 2>&1 &
    fi
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
    elif [[ -n $passed_key ]]; then
        touch "$passed_dir/$passed_key-$step"
    fi
done
mkdir -p "$kept" && cp "$logs"/* "$kept"/ 2>/dev/null || true
# Keep the last 20 runs' logs.
ls -1dt "$(dirname "$kept")"/*/ 2>/dev/null | tail -n +21 | xargs -r rm -rf
for step in "${steps[@]}"; do
    [[ " ${failed[*]} ${broken[*]} " == *" $step "* ]] || printf '== %s: %s\n' "$step" "$(tail -n 1 "$logs/$step" | grep -q 'passed on this tree before' && echo 'passed before on this tree' || echo passed)"
done
for step in "${failed[@]}"; do
    printf '\n== %s: failed\n%s\n(full log: %s)\n' "$step" "$(summary "$step" "$logs/$step")" "$kept/$step"
done
for step in "${broken[@]}"; do
    printf '\n== %s: broken environment\n%s\n' "$step" "$(tail -n 20 "$logs/$step")"
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
