#!/usr/bin/env bash
# Serves omp-web, the chat and settings surface of the frontend's omp tabs.
#
#   source    upstream kahme247/ompweb (GIGACHAD_OMP_WEB_REPO, ref GIGACHAD_OMP_WEB_REF), never modified there
#   checkout  a dedicated detached worktree (GIGACHAD_OMP_WEB_DIR) owned by this script: every start resets it to
#             the ref, so its `next build` never touches the source checkout's `npm run dev`
#   overlay   ompweb-overlay/ becomes the worktree's custom/ (embed mode, postMessage bridge, module overrides),
#             ompweb-overlay/seams.patch adds the two upstream lines that load it (app/layout.tsx, next.config.ts)
#   chat-ui   @gigachad/chat-ui is packed from src/frontend/packages/chat-ui and installed without touching the
#             worktree's manifest or lock
#   framing   omp-web refuses frames unless OMP_WEB_FRAME_ANCESTORS lists the frontend origin (custom/framing.ts),
#             `next build` bakes the headers in, so a changed value rebuilds
#
# Idempotent: an omp-web already answering on the port is reused rather than
# restarted (one instance per machine, it owns the omp session processes), with
# a warning when it would refuse the frontend's frame.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir/deploy/lib.sh"
bind="${GIGACHAD_OMP_WEB_BIND:-127.0.0.1:30177}"
repo="${GIGACHAD_OMP_WEB_REPO:-$HOME/git/ompweb_original}"
dir="${GIGACHAD_OMP_WEB_DIR:-$HOME/.local/share/gigachad-bot/ompweb-upstream}"
ref="${GIGACHAD_OMP_WEB_REF:-integration_patrick_ab}"
frame_ancestors="${GIGACHAD_OMP_WEB_FRAME_ANCESTORS:-http://127.0.0.1:2999 http://localhost:2999}"
overlay="$root_dir/ompweb-overlay"

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
    if [[ ! -e "$repo" ]]; then
        printf 'Cloning ompweb into %s\n' "$repo"
        mkdir -p "$(dirname -- "$repo")"
        git clone https://github.com/kahme247/ompweb.git "$repo"
    fi
    if [[ ! -e "$repo/.git" ]]; then
        printf 'error: %s is not an ompweb Git checkout (set GIGACHAD_OMP_WEB_REPO)\n' "$repo" >&2
        exit 1
    fi
    printf 'Creating the omp-web worktree %s at %s\n' "$dir" "$ref"
    mkdir -p "$(dirname -- "$dir")"
    git -C "$repo" worktree add --detach "$dir" "$ref"
fi

# Reset to the ref and apply the seams only when the worktree is elsewhere or the patch changed, a needless reset
# would touch the patched files and force a rebuild. rsync keeps unchanged files' mtimes for the same reason.
# Detached because the ref is usually checked out in the source checkout itself.
target="$(git -C "$repo" rev-parse "$ref^{commit}")"
if [[ "$(git -C "$dir" rev-parse HEAD)" != "$target" ]] || ! git -C "$dir" apply --reverse --check "$overlay/seams.patch" 2>/dev/null; then
    git -C "$dir" checkout --quiet --force --detach "$target"
    git -C "$dir" apply "$overlay/seams.patch"
fi
rsync -a --delete "$overlay/" "$dir/custom/"
printf 'omp-web worktree at %s (%s) + ompweb-overlay\n' "$ref" "$(git -C "$dir" log -1 --format='%h %s')"

# Node >= 24 comes from the project venv, like the frontend's.
export PATH="$root_dir/.venv/bin:$PATH"
cd -- "$dir"

if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
    npm ci --no-audit --no-fund
    rm -f node_modules/.gigachad-chat-ui.sha256
fi
# Packed into node_modules, which the rebuild check below skips: the build stamp tracks the tarball's digest instead.
# `npm install --no-save` keeps manifest and lock upstream's, a later `npm ci` removes the package and resets the stamp.
chat_ui_tarball="$(cd -- "$root_dir/src/frontend" && node packages/chat-ui/pack.mjs --out "$dir/node_modules/.gigachad-chat-ui")"
chat_ui_digest="$(sha256sum "$chat_ui_tarball" | cut -d' ' -f1)"
if [[ "$(cat node_modules/.gigachad-chat-ui.sha256 2>/dev/null)" != "$chat_ui_digest" ]]; then
    npm install --no-save --no-audit --no-fund --ignore-scripts "$chat_ui_tarball"
    printf '%s' "$chat_ui_digest" >node_modules/.gigachad-chat-ui.sha256
fi

# Webpack's persistent cache misses module-replacement changes, it then resolves overrides stale.
if [[ -f .next/BUILD_ID && -n "$(find next.config.ts custom/next.config.ts custom/overrides -newer .next/BUILD_ID -print -quit)" ]]; then
    rm -rf .next/cache
fi
export OMP_WEB_FRAME_ANCESTORS="$frame_ancestors"
build_stamp=.next/gigachad-build
build_inputs="$frame_ancestors $chat_ui_digest"
if [[ ! -f .next/BUILD_ID ]] \
    || [[ -n "$(find . \( -path ./.next -o -path ./node_modules -o -path ./.git \) -prune -o -type f -newer .next/BUILD_ID -print -quit)" ]] \
    || [[ "$(cat "$build_stamp" 2>/dev/null)" != "$build_inputs" ]]; then
    printf 'Building omp-web (~2 min)\n'
    npm run build 2>&1 | tee node_modules/.gigachad-build.log
    # Next only warns when an override lacks an export upstream modules import, the page would then crash at runtime.
    if grep -q 'Attempted import error' node_modules/.gigachad-build.log; then
        printf 'error: an overlay module misses an export, see %s/node_modules/.gigachad-build.log\n' "$dir" >&2
        rm -f .next/BUILD_ID
        exit 1
    fi
    printf '%s' "$build_inputs" >"$build_stamp"
fi

printf 'Starting omp-web on %s\n' "$bind"
exec node bin/omp-web.js --port "${bind##*:}" --hostname "${bind%%:*}" --no-open
