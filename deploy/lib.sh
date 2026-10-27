# Shared helpers for the launcher scripts. Source it, do not run it.

# Succeeds when host:port accepts a TCP connection.
port_open() {
    local host="${1%%:*}" port="${1##*:}"
    (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null
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
