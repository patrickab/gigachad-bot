# Shared helpers for the launcher scripts. Source it, do not run it.

# Succeeds when host:port accepts a TCP connection.
port_open() {
    local host="${1%%:*}" port="${1##*:}"
    (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null
}

# Runs `cmd` up to `tries` times, `interval` seconds apart, plus once more at the deadline.
# True as soon as it passes.
wait_until() {
    local tries=$1 interval=$2 attempt
    shift 2
    for ((attempt = 0; attempt < tries; attempt++)); do
        "$@" && return 0
        sleep "$interval"
    done
    "$@"
}

# Prints every descendant of `pid`, deepest first.
descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do
        descendants "$child"
        printf '%s\n' "$child"
    done
}

# Whether `pid` still runs: an exited child its parent has not reaped yet (a zombie) does not.
running() {
    local state
    state=$(ps -o stat= -p "$1" 2>/dev/null) && [[ "$state" != Z* ]]
}

# Stops `pids`: TERM, up to `grace` seconds to exit, then KILL for the rest.
stop_processes() {
    local grace=$1 pid alive tick
    shift
    (($# > 0)) || return 0
    kill -TERM "$@" 2>/dev/null || true
    for ((tick = 0; tick < grace * 4; tick++)); do
        alive=()
        for pid in "$@"; do running "$pid" && alive+=("$pid"); done
        ((${#alive[@]} == 0)) && return 0
        sleep 0.25
    done
    kill -KILL "$@" 2>/dev/null || true
}

# Stops `pid` and its descendants: TERM to `pid`, up to `grace` seconds for the whole tree, then KILL for the rest.
stop_tree() {
    local grace=$1 pid tick
    local -a tree alive
    mapfile -t tree < <(descendants "$2"; printf '%s\n' "$2")
    kill -TERM "$2" 2>/dev/null || true
    for ((tick = 0; ; tick++)); do
        alive=()
        for pid in "${tree[@]}"; do running "$pid" && alive+=("$pid"); done
        ((${#alive[@]} == 0)) && return 0
        ((tick < grace * 4)) || break
        sleep 0.25
    done
    printf 'warning: %s did not stop within %ss, killing it\n' "${alive[*]}" "$grace" >&2
    kill -KILL "${alive[@]}" 2>/dev/null || true
}

# Kills what listens on `port` when it was started from `dir` and no `run*.sh` launcher owns it any more:
# the leftovers of a run that was killed before its cleanup ran. A live stack's processes are left alone.
free_orphaned_port() {
    local port=$1 dir=$2 pid ancestor owned
    for pid in $(ss -Hltnp "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
        [[ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" == "$dir"* ]] || continue
        owned=false
        ancestor=$pid
        while ancestor=$(ps -o ppid= -p "$ancestor" 2>/dev/null | tr -d ' ') && ((ancestor > 1)); do
            if ps -o args= -p "$ancestor" | grep -qE '(^|[ /])run(-[a-z-]+)?\.sh( |$)'; then owned=true; break; fi
        done
        if [[ "$owned" == false ]]; then
            printf 'Stopping orphaned process %s on port %s\n' "$pid" "$port" >&2
            stop_tree 5 "$pid"
        fi
    done
}

# Repairs a stale development uv.lock, then runs `uv sync --locked` with the given options.
# run.sh syncs once and exports GIGACHAD_DEPS_SYNCED=1, so the scripts it starts skip
# their own sync, which would race it on the shared venv.
sync_python_deps() {
    if [[ -n "${GIGACHAD_DEPS_SYNCED:-}" ]]; then
        return 0
    fi
    local root
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    python3 "$root/src/backend/install_deps.py" python-lock "$root"
    uv --directory "$root" sync --locked "$@"
}

# Runs `npm ci` for src/frontend, which deletes and rewrites its node_modules. run.sh does it once up front and
# exports GIGACHAD_FRONTEND_DEPS_INSTALLED=1: LearnHarness's chat-ui pack (tsc from that node_modules) and the
# backend's reloader would otherwise run into the rewrite.
install_frontend_deps() {
    if [[ -n "${GIGACHAD_FRONTEND_DEPS_INSTALLED:-}" ]]; then
        return 0
    fi
    # The frontend imports the deepfocus-ambient submodule's source (src/frontend/packages/deepfocus-ambient)
    git -C "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" submodule update --init
    uv --directory "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" run --no-sync gigachad-install
}

# Packs @gigachad/chat-ui into directory $1 and prints the tarball path (node >= 24 on PATH).
pack_chat_ui() {
    (cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../src/frontend" && node packages/chat-ui/pack.mjs --out "$1")
}

# Loads ~/.secrets into the caller's shell when it exists.
load_secrets() {
    if [[ -f "$HOME/.secrets" ]]; then
        source "$HOME/.secrets"
    fi
}

# Sets pg_host, pg_port and password_var for the development or production PostgreSQL.
postgres_profile() {
    if [[ "$1" == production ]]; then
        pg_host=127.0.0.2 pg_port=5433 password_var=POSTGRES_PASSWORD_PROD
    else
        pg_host=127.0.0.1 pg_port=5432 password_var=POSTGRES_PASSWORD_DEV
    fi
}
