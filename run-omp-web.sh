#!/usr/bin/env bash
# Serves omp-web, the chat and settings surface of the frontend's omp tabs.
#
#   source    patrickab/ompweb fork, branch `custom` (embed mode + postMessage bridge)
#   checkout  a dedicated detached worktree, so its `next build` never touches
#             the fork checkout's `npm run dev`. Every start moves it to the
#             latest commit of `custom`, its run.sh then rebuilds what changed
#   framing   omp-web refuses frames unless OMP_WEB_FRAME_ANCESTORS lists the
#             frontend origin. Its run.sh rebuilds when that value changes
#
# Idempotent: an omp-web already answering on the port is reused rather than
# restarted (one instance per machine, it owns the omp session processes), with
# a warning when it would refuse the frontend's frame.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bind="${GIGACHAD_OMP_WEB_BIND:-127.0.0.1:30177}"
repo="${GIGACHAD_OMP_WEB_REPO:-$HOME/git/ompweb}"
dir="${GIGACHAD_OMP_WEB_DIR:-$HOME/.local/share/gigachad-bot/ompweb}"
ref="${GIGACHAD_OMP_WEB_REF:-custom}"
frame_ancestors="${GIGACHAD_OMP_WEB_FRAME_ANCESTORS:-http://127.0.0.1:2999 http://localhost:2999}"

port_open() {
    local host="${1%%:*}" port="${1##*:}"
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
        exec 3<&- 3>&-
        return 0
    fi
    return 1
}

# The first listed origin must appear in the served CSP, else the frame stays blank.
allows_framing() {
    local first_origin="${frame_ancestors%% *}"
    curl -sIL "http://$bind/" | grep -i '^content-security-policy:' | grep -qF "$first_origin"
}

if port_open "$bind"; then
    printf 'Reusing the omp-web already listening on %s\n' "$bind"
    if ! allows_framing; then
        printf 'omp-web on %s refuses frames from %s: restart it with OMP_WEB_FRAME_ANCESTORS="%s"\n' \
            "$bind" "${frame_ancestors%% *}" "$frame_ancestors" >&2
    fi
    exit 0
fi
if ! command -v omp >/dev/null 2>&1; then
    printf 'omp not on PATH; skipping omp-web\n' >&2
    exit 0
fi
if [[ ! -d "$dir" ]]; then
    if [[ ! -d "$repo/.git" ]]; then
        printf 'No ompweb fork at %s (set GIGACHAD_OMP_WEB_REPO); skipping omp-web\n' "$repo" >&2
        exit 0
    fi
    printf 'Creating the omp-web worktree %s at %s\n' "$dir" "$ref"
    mkdir -p "$(dirname -- "$dir")"
    git -C "$repo" worktree add --detach "$dir" "$ref"
fi
# Detached because `custom` is usually checked out in the fork checkout itself.
git -C "$dir" checkout --quiet --detach "$ref"
printf 'omp-web worktree at %s (%s)\n' "$ref" "$(git -C "$dir" log -1 --format='%h %s')"

# Node >= 24 comes from the project venv, like the frontend's.
export PATH="$root_dir/.venv/bin:$PATH"
export OMP_WEB_FRAME_ANCESTORS="$frame_ancestors"
printf 'Starting omp-web on %s (first start installs and builds, ~2 min)\n' "$bind"
exec "$dir/run.sh" --port "${bind##*:}" --hostname "${bind%%:*}" --no-open
