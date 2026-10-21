#!/usr/bin/env bash
# Bootstrap only the development database, never the application servers.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
secrets_file="$HOME/.secrets"
inherited_password="${POSTGRES_PASSWORD_DEV:-}"
unset POSTGRES_PASSWORD_DEV

if [[ -e "$secrets_file" ]]; then
    source "$secrets_file"
fi

saved_password="${POSTGRES_PASSWORD_DEV:-}"
POSTGRES_PASSWORD_DEV="${saved_password:-$inherited_password}"
if [[ -z "$POSTGRES_PASSWORD_DEV" ]]; then
    printf 'POSTGRES_PASSWORD_DEV is not set.\n'
    printf 'Create a local development database and generate a password saved in %s? [y/N] ' "$secrets_file"
    if ! IFS= read -r answer || [[ ! "$answer" =~ ^[yY]([eE][sS])?$ ]]; then
        printf 'Setup cancelled. No changes made.\n'
        exit 0
    fi
else
    printf 'POSTGRES_PASSWORD_DEV is set. Starting the development database.\n'
fi

# Query the daemon before writing credentials, so Docker failures leave secrets untouched.
volumes="$(docker volume ls --format '{{.Name}}')"
if [[ -z "$POSTGRES_PASSWORD_DEV" ]]; then
    if [[ $'\n'"$volumes"$'\n' == *$'\ngigachad-dev_gigachad-postgres\n'* ]]; then
        printf 'error: an existing development database has no saved password. Restore POSTGRES_PASSWORD_DEV in %s, then rerun. No data was changed.\n' "$secrets_file" >&2
        exit 1
    fi
    POSTGRES_PASSWORD_DEV="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
fi
if [[ -z "$saved_password" ]]; then
    (umask 077 && touch "$secrets_file")
    chmod 600 "$secrets_file"
    printf '\nexport POSTGRES_PASSWORD_DEV=%q\n' "$POSTGRES_PASSWORD_DEV" >> "$secrets_file"
    printf 'Saved development database credentials in %s\n' "$secrets_file"
fi

export POSTGRES_PASSWORD="$POSTGRES_PASSWORD_DEV"
export PGPASSWORD="$POSTGRES_PASSWORD"
export POSTGRES_USER="$(whoami)"
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432

compose() {
    docker compose --project-directory "$root_dir" --project-name gigachad-dev "$@"
}

compose up --detach --wait postgres
# pg_isready alone does not verify that the saved password authenticates.
compose exec -T -e PGPASSWORD postgres psql -h 127.0.0.1 -U "$POSTGRES_USER" -d gigachad -v ON_ERROR_STOP=1 -c 'SELECT current_database();'
printf 'Development PostgreSQL is ready at 127.0.0.1:5432 (database: gigachad). Start the app with ./run.sh\n'
