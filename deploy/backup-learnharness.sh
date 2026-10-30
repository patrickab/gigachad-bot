#!/usr/bin/env bash
# Backs up learn mode's LearnHarness data, or restores one backup into an empty home.
#
#   backup             (the default, gigachad-bot-learnharness-backup.timer runs it daily) archives
#                      GIGACHAD_LEARNHARNESS_HOME's data/ (LH_DATA_DIR), auth.json (LH_AUTH_FILE), the credential files
#                      LearnHarness keeps beside it and served-commit (the LearnHarness commit that wrote the data) to
#                      GIGACHAD_LEARNHARNESS_BACKUP_DIR/learnharness-<utc>.tar.gz, keeping the newest
#                      GIGACHAD_LEARNHARNESS_BACKUP_KEEP (default 14). LearnHarness keeps running: each file is copied as it
#                      is at that moment
#   restore <archive>  unpacks a backup into GIGACHAD_LEARNHARNESS_HOME, which must hold no data/ and no auth.json: stop
#                      gigachad-bot-learnharness.service and move the current home aside first
#
# The archives hold the learner's chats and LearnHarness's credentials: mode 600 in a mode 700 directory.
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/load-env.sh"

home="${GIGACHAD_LEARNHARNESS_HOME:-$HOME/.local/share/gigachad-bot/learnharness}"
backup_dir="${GIGACHAD_LEARNHARNESS_BACKUP_DIR:-$HOME/.local/share/gigachad-bot/backups/learnharness}"
keep="${GIGACHAD_LEARNHARNESS_BACKUP_KEEP:-14}"
umask 077

backup() {
    if [[ ! -d "$home/data" ]]; then
        printf 'No LearnHarness data in %s, nothing to back up\n' "$home"
        return 0
    fi
    local items=(data) file
    shopt -s nullglob
    for file in "$home/auth.json" "$home"/*-credentials.json* "$home/served-commit"; do
        items+=("${file##*/}")
    done
    shopt -u nullglob
    mkdir -p "$backup_dir"
    chmod 700 "$backup_dir"
    local archive="$backup_dir/learnharness-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    # Exit 1 means a file changed or went while it was read (a turn wrote a chat, a session was deleted): the archive is
    # complete, with that file as one of its versions. Anything else is a failure, and the partial archive goes.
    local status=0
    tar --create --gzip --file "$archive.partial" --directory "$home" --warning=no-file-changed --warning=no-file-removed \
        -- "${items[@]}" || status=$?
    if ((status > 1)); then
        rm -f "$archive.partial"
        printf 'error: tar failed (%d) archiving %s\n' "$status" "$home" >&2
        return 1
    fi
    mv "$archive.partial" "$archive"
    printf 'Backed up %s to %s (%s)\n' "$home" "$archive" "$(du -h "$archive" | cut -f1)"
    local archives
    mapfile -t archives < <(find "$backup_dir" -maxdepth 1 -name 'learnharness-*.tar.gz' -printf '%f\n' | LC_ALL=C sort -r)
    for file in "${archives[@]:keep}"; do
        rm -f "$backup_dir/$file"
        printf 'Removed the old backup %s\n' "$file"
    done
}

restore() {
    local archive
    archive="$(realpath -m -- "$1")"
    if [[ ! -f "$archive" ]]; then
        printf 'error: no backup at %s\n' "$archive" >&2
        return 1
    fi
    if [[ -e "$home/data" || -e "$home/auth.json" ]]; then
        printf 'error: %s already holds LearnHarness data: stop gigachad-bot-learnharness.service and move it aside first\n' \
            "$home" >&2
        return 1
    fi
    mkdir -p "$home"
    chmod 700 "$home"
    tar --extract --gzip --file "$archive" --directory "$home" --same-permissions
    printf 'Restored %s into %s (written by LearnHarness %s)\n' "$archive" "$home" "$(cat "$home/served-commit" 2>/dev/null || echo unknown)"
}

case "${1:-backup}" in
    backup)
        (($# <= 1)) || { printf 'usage: %s [backup | restore <archive>]\n' "${BASH_SOURCE[0]}" >&2; exit 2; }
        backup
        ;;
    restore)
        (($# == 2)) || { printf 'usage: %s [backup | restore <archive>]\n' "${BASH_SOURCE[0]}" >&2; exit 2; }
        restore "$2"
        ;;
    *)
        printf 'usage: %s [backup | restore <archive>]\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac
