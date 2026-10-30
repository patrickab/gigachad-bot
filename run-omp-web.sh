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
#   update    before serving, the ref branch is rebased locally onto origin/main when it lacks it (auto_update below),
#             GIGACHAD_AUTO_UPDATE=0 opts out
#
# Idempotent: an omp-web already answering on the port is reused rather than
# restarted (one instance per machine, it owns the omp session processes), with
# a warning when it would refuse the frontend's frame. Its update waits for the next start.
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

note() {
    printf 'omp-web auto-update: %s\n' "$*" >&2
}

# True when the worktree has no tracked changes: untracked files survive the move, `reset --keep` refuses to clobber them.
clean() {
    [[ -z "$(git -C "$1" status --porcelain --untracked-files=no 2>/dev/null)" ]] && git -C "$1" rev-parse --verify --quiet HEAD >/dev/null
}

# The runbook's overlay checks (.docs/integration-ompweb.md, "Private overlay check checkout") in worktree $1.
# `&&` chains them: set -e is off inside a function called from a condition.
overlay_checks() (
    local tarball
    cd -- "$1" \
        && git apply "$overlay/seams.patch" \
        && rsync -a --delete "$overlay/" custom/ \
        && npm ci --no-audit --no-fund \
        && tarball="$(pack_chat_ui "$1/node_modules/.gigachad-chat-ui")" \
        && npm install --no-save --no-audit --no-fund --ignore-scripts "$tarball" \
        && node_modules/.bin/tsc --noEmit \
        && node --experimental-strip-types --test custom/*.test.mjs
)

# Rebases a copy of the ref branch onto origin/main in a temporary worktree, runs the overlay checks on it, then moves
# the branch (and the clean worktree holding it) there. Local only, never pushes. Every failure keeps the branch and
# notes why: a start is never blocked. A failed rebase or check is remembered per (branch, origin/main, overlay and
# chat-ui sources), so later starts skip the npm ci and checks until one of them changes.
auto_update() {
    local branch="refs/heads/$ref" old main held="" wt="" line key stamp="$dir.auto-update-failed" error
    local log="$dir.auto-update.log" work reason="" new="" seen_key="" seen_reason=""
    if ! git -C "$repo" show-ref --verify --quiet "$branch"; then
        note "skipped, $ref is not a local branch of $repo"
        return 0
    fi
    # stdin closed and no prompts: ssh or credential prompts would otherwise hang a systemd start.
    if ! error="$(GIT_TERMINAL_PROMPT=0 timeout 60 git -C "$repo" fetch --quiet origin 2>&1 </dev/null)"; then
        error="${error%%$'\n'*}"
        note "kept $ref, fetching origin failed (offline?): ${error:-timed out}"
        return 0
    fi
    if ! main="$(git -C "$repo" rev-parse --verify --quiet refs/remotes/origin/main)"; then
        note "kept $ref, origin has no main branch"
        return 0
    fi
    old="$(git -C "$repo" rev-parse "$branch")" || return 0
    if git -C "$repo" merge-base --is-ancestor "$main" "$old"; then
        return 0
    fi
    while IFS= read -r line; do
        case $line in
            "worktree "*) wt="${line#worktree }" ;;
            "branch $branch") held="$wt" ;;
        esac
    done < <(git -C "$repo" worktree list --porcelain)
    if [[ -n $held ]] && ! clean "$held"; then
        note "kept $ref, its worktree $held has uncommitted changes: commit or stash them to update"
        return 0
    fi
    key="$old $main $( { find "$overlay" "$root_dir/src/frontend/packages/chat-ui" -name node_modules -prune -o -type f \
        -printf '%P %s %T@\n' 2>/dev/null || true; } | sort | sha256sum | cut -c1-64)"
    { read -r seen_key && read -r seen_reason; } 2>/dev/null <"$stamp" || true
    if [[ $seen_key == "$key" ]]; then
        note "kept $ref, $seen_reason (as on the last start)"
        return 0
    fi
    if ! work="$(mktemp -d "$dir.update.XXXXXX")"; then
        note "kept $ref, could not create a temporary directory next to $dir"
        return 0
    fi
    if ! git -C "$repo" worktree add --quiet --detach "$work" "$old" >"$log" 2>&1; then
        rm -rf "$work"
        note "kept $ref, could not create a temporary worktree, see $log"
        return 0
    fi
    printf 'omp-web auto-update: rebasing %s onto origin/main %s and checking the overlay\n' "$ref" "${main:0:7}"
    # No updateRefs: other branches at the rebased commits (topic branches) stay where they are.
    if ! git -C "$work" -c rebase.updateRefs=false rebase --no-autosquash --no-autostash "$main" >>"$log" 2>&1; then
        git -C "$work" rebase --abort >>"$log" 2>&1
        reason="rebasing onto origin/main ${main:0:7} conflicts"
    elif ! overlay_checks "$work" >>"$log" 2>&1; then
        reason="the overlay checks fail after rebasing onto origin/main ${main:0:7}, see $log"
    else
        new="$(git -C "$work" rev-parse HEAD)"
    fi
    git -C "$repo" worktree remove --force "$work" >>"$log" 2>&1 || { rm -rf "$work" && git -C "$repo" worktree prune; }
    if [[ -z $new ]]; then
        printf '%s\n%s\n' "$key" "$reason" >"$stamp"
        note "kept $ref, $reason"
        return 0
    fi
    # Compare-and-swap: the branch or its worktree may have moved during the checks.
    if [[ -n $held ]]; then
        if [[ "$(git -C "$held" rev-parse HEAD 2>/dev/null)" != "$old" ]] || ! clean "$held" \
            || ! git -C "$held" reset --quiet --keep "$new" >>"$log" 2>&1; then
            note "kept $ref, its worktree $held changed during the update"
            return 0
        fi
    elif ! git -C "$repo" update-ref -m "gigachad auto-update: rebase onto origin/main" "$branch" "$new" "$old" >>"$log" 2>&1; then
        note "kept $ref, it moved during the update"
        return 0
    fi
    rm -f "$stamp"
    printf 'omp-web auto-update: rebased %s onto origin/main %s (%s -> %s)\n' "$ref" "${main:0:7}" "${old:0:7}" "${new:0:7}"
}

if port_open "$bind"; then
    printf 'Reusing the omp-web already listening on %s\n' "$bind"
    if ! allows_framing; then
        printf 'omp-web on %s refuses frames from %s: restart it with OMP_WEB_FRAME_ANCESTORS="%s"\n' \
            "$bind" "${frame_ancestors%% *}" "$frame_ancestors" >&2
    fi
    # It was built from whatever overlay its launcher copied in: another checkout's, or this one's before an edit.
    if [[ -d "$dir/custom" && -n "$(rsync -rnic --delete "$overlay/" "$dir/custom/")" ]]; then
        printf 'omp-web on %s serves another overlay than %s: stop it (when idle) and rerun to rebuild\n' "$bind" "$overlay" >&2
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

# Node >= 24 comes from the project venv, like the frontend's.
export PATH="$root_dir/.venv/bin:$PATH"
if [[ ${GIGACHAD_AUTO_UPDATE:-1} != 0 ]]; then
    # One update at a time per served checkout, e.g. run.sh and the systemd unit starting together.
    mkdir -p "$(dirname -- "$dir")"
    exec {update_lock}>"$dir.auto-update.lock"
    if flock -n "$update_lock"; then
        auto_update || true
    else
        note "skipped, another launcher is updating $ref"
    fi
    # Closed before exec, else the server would hold the lock for its lifetime.
    exec {update_lock}>&-
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

cd -- "$dir"

if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
    npm ci --no-audit --no-fund
    rm -f node_modules/.gigachad-chat-ui.sha256
fi
# Packed into node_modules, which the rebuild check below skips: the build stamp tracks the tarball's digest instead.
# `npm install --no-save` keeps manifest and lock upstream's, a later `npm ci` removes the package and resets the stamp.
chat_ui_tarball="$(pack_chat_ui "$dir/node_modules/.gigachad-chat-ui")"
chat_ui_digest="$(sha256sum "$chat_ui_tarball" | cut -d' ' -f1)"
if [[ "$(cat node_modules/.gigachad-chat-ui.sha256 2>/dev/null)" != "$chat_ui_digest" ]]; then
    npm install --no-save --no-audit --no-fund --ignore-scripts "$chat_ui_tarball"
    printf '%s' "$chat_ui_digest" >node_modules/.gigachad-chat-ui.sha256
fi

build_stamp=.next/gigachad-build
built_inputs="$(cat "$build_stamp" 2>/dev/null || true)"
# Webpack's persistent cache misses module-replacement changes and a repacked chat-ui (it takes a node_modules package as
# immutable per version), it then builds them stale.
if [[ -f .next/BUILD_ID ]] && { [[ -n "$(find next.config.ts custom/next.config.ts custom/overrides -newer .next/BUILD_ID -print -quit)" ]] \
    || [[ "${built_inputs##* }" != "$chat_ui_digest" ]]; }; then
    rm -rf .next/cache
fi
export OMP_WEB_FRAME_ANCESTORS="$frame_ancestors"
build_inputs="$frame_ancestors $chat_ui_digest"
if [[ ! -f .next/BUILD_ID ]] \
    || [[ -n "$(find . \( -path ./.next -o -path ./node_modules -o -path ./.git \) -prune -o -type f -newer .next/BUILD_ID -print -quit)" ]] \
    || [[ "$built_inputs" != "$build_inputs" ]]; then
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
