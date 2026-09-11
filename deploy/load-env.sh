#!/usr/bin/env bash

_gigachad_load_env() {
    local env_file
    local line
    local name
    local value
    local line_number=0
    local -A inherited=()

    if [[ -n ${GIGACHAD_ENV_FILE+x} ]]; then
        env_file="$GIGACHAD_ENV_FILE"
    else
        env_file="${XDG_CONFIG_HOME:-"$HOME/.config"}/gigachad-bot/env"
        [[ ! -e "$env_file" ]] && return 0
    fi

    if [[ ! -e "$env_file" ]]; then
        printf 'error: cannot read environment file: %s\n' "$env_file" >&2
        return 1
    fi

    if [[ ! -f "$env_file" || ! -r "$env_file" ]]; then
        printf 'error: cannot read environment file: %s\n' "$env_file" >&2
        return 1
    fi

    while IFS= read -r name; do
        inherited["$name"]=1
    done < <(compgen -e)

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((++line_number))

        if [[ "$line" =~ ^[[:space:]]*$ || "$line" == \#* ]]; then
            continue
        fi

        if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$ ]]; then
            name="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            if [[ -z ${inherited[$name]+_} ]]; then
                if ! declare -g -x "$name=$value"; then
                    printf 'error: cannot set environment variable %s from %s\n' "$name" "$env_file" >&2
                    return 1
                fi
            fi
        else
            printf 'error: malformed environment record in %s at line %d\n' "$env_file" "$line_number" >&2
            return 2
        fi
    done < "$env_file"
}

if _gigachad_load_env; then
    unset -f _gigachad_load_env
else
    _gigachad_load_env_status=$?
    unset -f _gigachad_load_env
    return "$_gigachad_load_env_status"
fi
