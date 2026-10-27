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

# Repairs a stale development uv.lock, then runs `uv sync --locked` with the given options.
# run.sh syncs once and exports GIGACHAD_DEPS_SYNCED=1, so the scripts it starts skip
# their own sync: a parallel exact sync without the sandbox-plot extra would uninstall it.
sync_python_deps() {
    if [[ -n "${GIGACHAD_DEPS_SYNCED:-}" ]]; then
        return 0
    fi
    local root
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    python3 "$root/src/backend/install_deps.py" python-lock "$root"
    uv --directory "$root" sync --locked "$@"
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
