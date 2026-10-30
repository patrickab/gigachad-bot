#!/usr/bin/env bash
# Serves LearnHarness, the learn mode's teaching surface.
#
#   source    upstream amosblomqvist/learnharness (GIGACHAD_LEARNHARNESS_REPO), never modified there
#   update    GIGACHAD_LEARNHARNESS_REF unset (the default): every start fetches origin's tags and serves the newest
#             release tag (v*, version order) the overlay checks pass against, else the last working commit. The checks
#             (the markers below, `tsc --noEmit`, `node --test custom/*.test.mjs`) run in a second launcher-owned worktree,
#             GIGACHAD_LEARNHARNESS_DIR-check, which keeps its node_modules. Results are kept per commit and overlay in
#             GIGACHAD_LEARNHARNESS_HOME/update-checks/ (a failure with its log), so an unchanged upstream costs only the
#             fetch. A start is never blocked by an update: offline, a failing check or a failing build of the new commit
#             serve the last working commit (served-commit) and print one line saying why. GIGACHAD_AUTO_UPDATE=0 skips
#             the update. A set GIGACHAD_LEARNHARNESS_REF pins that ref and is never updated: a ref the clone lacks is
#             fetched from its origin once, a failure ends the start. `--select` prints the commit a start would serve
#             and exits (redeploy.sh)
#   checkout  a dedicated detached worktree (GIGACHAD_LEARNHARNESS_DIR) owned by this script: every start resets it to the ref
#   data      GIGACHAD_LEARNHARNESS_HOME (mode 700) holds data/ (LH_DATA_DIR), auth.json (LH_AUTH_FILE) and the credential
#             files LearnHarness keeps beside it. Upstream keeps no backward compatibility for stored data, so moving to
#             another commit first copies them to snapshots/<old-sha>-<utc>/. served-commit names the commit that last
#             served them, written once LearnHarness listens: the last working commit
#   overlay   learnharness-overlay/ becomes the worktree's custom/ (embed mode, postMessage bridge, theme),
#             learnharness-overlay/seams.patch adds the upstream lines that load it (app/layout.tsx, next.config.ts)
#   chat-ui   @gigachad/chat-ui is packed from src/frontend/packages/chat-ui and installed without touching the
#             worktree's manifest or lock
#   build     .next/gigachad-build is the only proof of a complete build: removed before `npm run build`, written after
#             it, holding a digest of the commit, the overlay tree, the chat-ui tarball and the frame ancestors.
#             `npm ci` and the build run from an allowlisted environment, never with this one's provider keys
#   framing   GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS (default the local :2999 shell) may frame LearnHarness (CSP
#             frame-ancestors) and talk to its bridge, baked into the build (custom/framing.ts)
#   sandbox   LearnHarness runs as the transient user service gigachad-learnharness-<port> (systemd-run): its agent reads
#             any absolute path, so the user's credentials, gigachad's .env files and the LiteLLM proxy's home are
#             InaccessiblePaths, and HOME is GIGACHAD_LEARNHARNESS_HOME/home. Stopping this script stops the unit
#   env       LearnHarness starts from an allowlisted environment, never this one or the user manager's: a provider key in
#             it would list that provider's models in every LearnHarness model menu, LH_DESKTOP_* would lock the frame out
#             of its API
#   demos     LearnHarness runs every demo in Chrome (CHROME_PATH) from a file in TMPDIR. Flatpak Chrome has a private
#             /tmp, so TMPDIR is GIGACHAD_LEARNHARNESS_HOME/tmp and CHROME_PATH a wrapper sharing it with the sandbox.
#             flatpak reaches its portal over the session bus (XDG_RUNTIME_DIR, DBUS_SESSION_BUS_ADDRESS)
#   models    the only provider is the custom endpoint, the LiteLLM proxy. run-litellm-proxy.sh writes LearnHarness's model
#             snapshot and tiers when it starts the proxy, so both list the same models. The proxy's master key reaches
#             LearnHarness as CUSTOM_OPENAI_API_KEY (created here when the proxy has not yet) and is never stored in its files
#   markers   the overlay drives upstream's own components, events and URLs, and the backend adapter its chat API: a ref
#             lacking one of the names below is refused rather than served with a silently broken learn mode
#
# Idempotent: a LearnHarness already answering on the port, or its unit still starting, is reused rather than restarted.
set -euo pipefail

mode=serve
case "${1:-}" in
    '') ;;
    --select) mode=select ;;
    *)
        printf 'usage: %s [--select]\n' "${BASH_SOURCE[0]}" >&2
        exit 2
        ;;
esac
# --select: stdout carries only the chosen commit, everything else goes to stderr.
if [[ $mode == select ]]; then
    exec 5>&1 1>&2
fi

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir/deploy/lib.sh"
bind="${GIGACHAD_LEARNHARNESS_BIND:-127.0.0.1:30181}"
repo="${GIGACHAD_LEARNHARNESS_REPO:-$HOME/git/learnharness}"
dir="${GIGACHAD_LEARNHARNESS_DIR:-$HOME/.local/share/gigachad-bot/learnharness-upstream}"
check_dir="$dir-check"
pin="${GIGACHAD_LEARNHARNESS_REF:-}"
auto_update="${GIGACHAD_AUTO_UPDATE:-1}"
home="${GIGACHAD_LEARNHARNESS_HOME:-$HOME/.local/share/gigachad-bot/learnharness}"
frame_ancestors="${GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS:-http://127.0.0.1:2999 http://localhost:2999}"
overlay="$root_dir/learnharness-overlay"
proxy_home="${GIGACHAD_LITELLM_PROXY_HOME:-$HOME/.local/share/gigachad-bot/litellm-proxy}"
chrome_app=com.google.Chrome
unit="gigachad-learnharness-${bind##*:}"

# The first listed origin must appear in the served CSP, else the shell's frame stays blank.
allows_framing() {
    curl -sI --max-time 5 "http://$bind/" | grep -i '^content-security-policy:' | grep -qF "${frame_ancestors%% *}"
}

if [[ $mode == serve ]] && port_open "$bind"; then
    printf 'Reusing the LearnHarness already listening on %s\n' "$bind"
    if ! allows_framing; then
        printf 'LearnHarness on %s refuses frames from %s: stop it and rerun with GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS="%s"\n' \
            "$bind" "${frame_ancestors%% *}" "$frame_ancestors" >&2
    fi
    # It was built from whatever overlay its launcher copied in: another checkout's, or this one's before an edit.
    if [[ -d "$dir/custom" && -n "$(rsync -rnic --delete "$overlay/" "$dir/custom/")" ]]; then
        printf 'LearnHarness on %s serves another overlay than %s: stop it and rerun to rebuild\n' "$bind" "$overlay" >&2
    fi
    exit 0
fi
if [[ $mode == serve ]] && systemctl --user --quiet is-active "$unit"; then
    printf 'Reusing the LearnHarness unit %s, still starting on %s\n' "$unit" "$bind"
    exit 0
fi

if [[ ! -e "$repo" ]]; then
    printf 'Cloning LearnHarness into %s\n' "$repo"
    mkdir -p "$(dirname -- "$repo")"
    git clone https://github.com/amosblomqvist/learnharness.git "$repo"
fi
if [[ ! -e "$repo/.git" ]]; then
    printf 'error: %s is not a LearnHarness Git checkout (set GIGACHAD_LEARNHARNESS_REPO)\n' "$repo" >&2
    exit 1
fi

# Chats and credentials are the learner's: no other local user reads them.
data_dir="$home/data"
demo_tmp="$home/tmp"
mkdir -p "$data_dir" "$demo_tmp" "$home/home"
chmod 700 "$home"
served_commit_file="$home/served-commit"
check_results="$home/update-checks"

# Node >= 24 comes from the project venv when it has one, like the frontend's.
export PATH="$root_dir/.venv/bin:$PATH"

# Installs, checks and the build run upstream's and its dependencies' code: no provider key reaches them.
build_environment=(
    PATH="$PATH"
    HOME="$HOME"
    LANG="${LANG:-C.UTF-8}"
    ELECTRON_SKIP_BINARY_DOWNLOAD=1
    GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS="$frame_ancestors"
)
for name in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [[ -n "${!name:-}" ]]; then
        build_environment+=("$name=${!name}")
    fi
done
isolated() {
    env -i "${build_environment[@]}" "$@"
}

# Upstream names and URLs the overlay (custom/) and the backend adapter (src/lib/learnharness.py) rely on, as file|text.
# Update when an overlay piece changes, drop a marker only with the piece that needed it: tsc and inventory.test.mjs
# catch the same in a check checkout, this list guards every update candidate and the ref actually served.
upstream_markers=(
    # Session model freeze and model bridge (overrides/components/model-picker.tsx, embed/freeze.ts, embed/model.tsx)
    'components/model-picker.tsx|export function ModelControl('
    'components/model-picker.tsx|"/api/settings/models"'
    'components/model-picker.tsx|`/api/chats/${chatId}/settings`'
    'components/settings/agent-section.tsx|export function composeSlotModel('
    'components/setup-cta.tsx|export function selectedModelUnavailable('
    'components/settings-signal.ts|export function notifySettingsSaved('
    'components/settings-signal.ts|export function useSettingsSaved('
    'app/api/settings/[section]/route.ts|export async function PUT('
    'app/api/chats/[id]/settings/route.ts|export async function PUT('
    # Markdown wrapper (overrides/components/ai-elements/message.tsx)
    'components/ai-elements/message.tsx|export const MessageResponse'
    'components/ai-elements/message.tsx|export function ToolContentResponse('
    # Hidden chrome (overrides/components/{sidebar,top-bar}.tsx, embed/chrome.tsx): content markers and the panels
    'components/sidebar.tsx|export function AppShell('
    'components/sidebar.tsx|export function SidebarToggle('
    'components/sidebar.tsx|data-testid="new-session"'
    'components/top-bar.tsx|export function TopBar('
    'components/chat.tsx|data-testid="phase-chip"'
    'components/chat.tsx|data-testid="plan-chip"'
    'components/settings/settings-dialog.tsx|export function useSettingsDialog('
    'components/settings/settings-dialog.tsx|data-testid="settings-nav"'
    'components/activity-dock.tsx|export function ActivitySection('
    'components/activity-dock.tsx|export function useActivity('
    'components/activity-dock.tsx|export function toBlobs('
    'components/extraction-coverage.tsx|export function CoverageMenuBlock('
    'components/extraction-coverage.tsx|export function CoverageRingDetail('
    'components/extraction-coverage.tsx|export async function fetchGraphRead('
    'components/popover-dismiss.ts|export function usePopoverDismiss('
    # Backend adapter: sessions, turns, rename, delete modes, Markdown export
    'app/api/chats/route.ts|export async function GET('
    'app/api/chats/route.ts|export async function POST('
    'app/api/chat/route.ts|export async function POST('
    'app/api/chats/[id]/route.ts|export async function PUT('
    'app/api/chats/[id]/route.ts|export async function DELETE('
    'app/api/chats/[id]/route.ts|"preserve-graph"'
    'app/api/chats/[id]/route.ts|"delete-nodes"'
    'app/api/chats/[id]/markdown/route.ts|export async function GET('
)
# The freeze's keyboard events, read from the freeze itself.
for event in $(grep -o 'lh:kbd-[a-z-]*' "$overlay/embed/freeze.ts" | sort -u); do
    upstream_markers+=("components/model-picker.tsx|$event")
done
# The freeze's keyboard events, read from the freeze itself.
for event in $(grep -o 'lh:kbd-[a-z-]*' "$overlay/embed/freeze.ts" | sort -u); do
    upstream_markers+=("components/model-picker.tsx|$event")
done
# Prints the first file|marker the checkout at $1 lacks, fails when it has them all.
lacking_marker() {
    local entry
    for entry in "${upstream_markers[@]}"; do
        if ! grep -qF -- "${entry#*|}" "$1/${entry%%|*}" 2>/dev/null; then
            printf '%s' "$entry"
            return 0
        fi
    done
    return 1
}

# Upstream's dependencies and @gigachad/chat-ui for the checkout in the working directory, sets chat_ui_digest.
# Electron is a devDependency for the desktop app. Without its binary LearnHarness renders demos in CHROME_PATH.
# chat-ui is packed into node_modules, which the rebuild check skips: the build stamp tracks the tarball's digest instead.
# `npm install --no-save` keeps manifest and lock upstream's, a later `npm ci` removes the package and resets the stamp.
install_dependencies() {
    local checkout="$PWD" chat_ui_tarball
    if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
        isolated npm ci --no-audit --no-fund || return 1
        rm -f node_modules/.gigachad-chat-ui.sha256
    fi
    chat_ui_tarball="$(cd -- "$root_dir/src/frontend" && node packages/chat-ui/pack.mjs --out "$checkout/node_modules/.gigachad-chat-ui")" \
        || return 1
    chat_ui_digest="$(sha256sum "$chat_ui_tarball" | cut -d' ' -f1)"
    if [[ "$(cat node_modules/.gigachad-chat-ui.sha256 2>/dev/null)" != "$chat_ui_digest" ]]; then
        isolated npm install --no-save --no-audit --no-fund --ignore-scripts "$chat_ui_tarball" || return 1
        printf '%s' "$chat_ui_digest" >node_modules/.gigachad-chat-ui.sha256
    fi
}

# A release tag and short sha for commit $1.
describe() {
    local tags
    mapfile -t tags < <(git -C "$repo" tag --points-at "$1" --list 'v*' --sort=-v:refname)
    printf '%s (%s)' "${tags[0]:-commit}" "$(git -C "$repo" rev-parse --short "$1")"
}

# The overlay checks against commit $1 in the check checkout, upstream's dependencies kept there between runs. The
# reason of a failure goes to fd 3: status 1 is the commit's (kept), 2 the environment's (an install, retried next start).
run_checks() {
    local lacking
    if [[ ! -d $check_dir ]]; then
        mkdir -p "$(dirname -- "$check_dir")"
        git -C "$repo" worktree add --quiet --detach "$check_dir" "$1" || { printf 'creating %s' "$check_dir" >&3; return 2; }
    fi
    if [[ ! -e $check_dir/.git ]]; then
        printf '%s is not a worktree of %s (remove it)' "$check_dir" "$repo" >&3
        return 2
    fi
    git -C "$check_dir" checkout --quiet --force --detach "$1" || { printf 'checking it out' >&3; return 2; }
    git -C "$check_dir" apply "$overlay/seams.patch" || { printf 'seams.patch' >&3; return 1; }
    rsync -a --delete "$overlay/" "$check_dir/custom/" || { printf 'copying the overlay' >&3; return 2; }
    if lacking="$(lacking_marker "$check_dir")"; then
        printf 'the launcher markers (%s lacks %s)' "${lacking%%|*}" "${lacking#*|}" >&3
        return 1
    fi
    cd -- "$check_dir" || return 2
    install_dependencies || { printf 'installing its dependencies' >&3; return 2; }
    isolated node_modules/.bin/tsc --noEmit || { printf 'tsc' >&3; return 1; }
    isolated node --test custom/*.test.mjs || { printf 'the overlay tests' >&3; return 1; }
}

# Passes when the overlay checks pass against commit $2 (tag $1), else prints why. A result is kept per commit and
# overlay, so an unchanged upstream costs only the fetch and a fixed overlay checks again.
overlay_digest="$(cd -- "$overlay" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1)"
check_candidate() {
    local result="$check_results/$2-$overlay_digest" reason status lock
    if [[ -f $result ]]; then
        reason="$(<"$result")"
        [[ $reason == pass ]] && return 0
        printf '%s' "$reason"
        return 1
    fi
    mkdir -p "$check_results" "$(dirname -- "$check_dir")"
    # One check at a time per check checkout: a dev start and redeploy.sh's --select may overlap.
    exec {lock}>"$check_dir.lock"
    flock "$lock"
    printf 'Checking LearnHarness %s against learnharness-overlay (a new release takes a few minutes)\n' "$1" >&2
    if reason="$(run_checks "$2" 3>&1 >"$result.log" 2>&1)"; then
        printf 'pass' >"$result"
        return 0
    else
        status=$?
    fi
    reason="${1} failed ${reason:-its checks} (log: $result.log)"
    if [[ $status != 2 ]]; then
        printf '%s' "$reason" >"$result"
    fi
    printf '%s' "$reason"
    return 1
}

# The last commit that listened is the last working one, the fallback of every update.
working="$(cat "$served_commit_file" 2>/dev/null || true)"
if [[ -n $working ]] && ! git -C "$repo" cat-file -e "$working^{commit}" 2>/dev/null; then
    working=""
fi
if [[ -n $pin ]]; then
    ref="$pin"
    if ! target="$(git -C "$repo" rev-parse --verify --quiet "$ref^{commit}")"; then
        printf 'Fetching %s into %s\n' "$ref" "$repo"
        git -C "$repo" fetch --quiet --tags origin || true
        if ! target="$(git -C "$repo" rev-parse --verify --quiet "$ref^{commit}")"; then
            printf 'error: ref %s not found in %s, nor in its origin\n' "$ref" "$repo" >&2
            exit 1
        fi
    fi
else
    skipped=""
    if [[ $auto_update == 0 ]]; then
        skipped="auto-update is off (GIGACHAD_AUTO_UPDATE=0)"
    elif ! fetch_error="$(GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$repo" fetch --quiet --force --tags origin 2>&1)"; then
        skipped="fetching origin's tags failed: ${fetch_error%%$'\n'*}"
        skipped="${skipped%: }"
    fi
    mapfile -t tags < <(git -C "$repo" tag --list 'v*' --sort=-v:refname)
    target=""
    failures=()
    if [[ -z $skipped ]]; then
        # Newest first, down to the last working commit: the first release passing the checks wins.
        for tag in "${tags[@]}"; do
            commit="$(git -C "$repo" rev-parse "$tag^{commit}")"
            if [[ -n $working ]] && git -C "$repo" merge-base --is-ancestor "$commit" "$working"; then
                break
            fi
            if failure="$(check_candidate "$tag" "$commit")"; then
                target="$commit"
                break
            fi
            failures+=("$failure")
        done
    fi
    why="$skipped"
    for failure in "${failures[@]}"; do
        why+="${why:+; }$failure"
    done
    if [[ -z $target && -n $working ]]; then
        target="$working"
    elif [[ -z $target && -n $skipped && ${#tags[@]} -gt 0 ]]; then
        # Nothing served yet and nothing to check against: the newest local release, the markers still guard it.
        target="$(git -C "$repo" rev-parse "${tags[0]}^{commit}")"
    elif [[ -z $target ]]; then
        printf 'error: no release tag (v*) in %s passes the overlay checks%s\n' "$repo" "${why:+: $why}" >&2
        exit 1
    fi
    ref="$(describe "$target")"
    if [[ -n $working && $target != "$working" ]]; then
        printf 'LearnHarness update: %s -> %s%s\n' "$(describe "$working")" "$ref" "${why:+ (skipped: $why)}" >&2
    elif [[ -n $why ]]; then
        printf 'LearnHarness not updated, serving %s: %s\n' "$ref" "$why" >&2
    fi
fi
if [[ $mode == select ]]; then
    printf '%s\n' "$target" >&5
    exit 0
fi

if [[ ! -d "$dir" ]]; then
    printf 'Creating the LearnHarness worktree %s at %s\n' "$dir" "$ref"
    mkdir -p "$(dirname -- "$dir")"
    # Detached because the ref may be checked out in the source checkout itself.
    git -C "$repo" worktree add --detach "$dir" "$target"
fi
if ! git -C "$dir" rev-parse --verify --quiet HEAD >/dev/null; then
    printf 'error: %s is not a worktree of %s, remove it to have it recreated\n' "$dir" "$repo" >&2
    exit 1
fi

# The data belongs to the commit that last served it, which outlives a recreated worktree. A snapshot equal to the
# newest one for that commit is not taken again, so retrying a failing bump keeps one copy.
served="$(cat "$served_commit_file" 2>/dev/null || git -C "$dir" rev-parse HEAD)"
snapshot_items=()
if [[ -n "$(ls -A "$data_dir")" ]]; then
    snapshot_items+=(data)
fi
shopt -s nullglob
# auth.json is no pattern, nullglob keeps it when absent.
for file in "$home/auth.json" "$home"/*-credentials.json*; do
    if [[ -e $file ]]; then
        snapshot_items+=("${file##*/}")
    fi
done
previous_snapshots=("$home/snapshots/$served-"*)
shopt -u nullglob
snapshot_is_current() {
    local item
    for item in "${snapshot_items[@]}"; do
        diff -rq --no-dereference "$home/$item" "$1/$item" >/dev/null 2>&1 || return 1
    done
}
if [[ "$served" != "$target" && ${#snapshot_items[@]} -gt 0 ]] \
    && ! { [[ ${#previous_snapshots[@]} -gt 0 ]] && snapshot_is_current "${previous_snapshots[-1]}"; }; then
    snapshot="$home/snapshots/$served-$(date -u +%Y%m%dT%H%M%SZ)"
    printf 'Moving LearnHarness from %s to %s: snapshotting its data to %s\n' "$served" "$target" "$snapshot"
    mkdir -p "$snapshot"
    for item in "${snapshot_items[@]}"; do
        cp -a "$home/$item" "$snapshot/"
    done
fi

# The worktree at commit $1 with the overlay, installed and built. Runs in a subshell, where a failing step ends it.
prepare() {
    set -e
    local target="$1" lacking
    # Reset to the ref and apply the seams only when the worktree is elsewhere or the patch changed, a needless reset
    # would touch the patched files and force a rebuild. rsync keeps unchanged files' mtimes for the same reason.
    if [[ "$(git -C "$dir" rev-parse HEAD)" != "$target" ]] || ! git -C "$dir" apply --reverse --check "$overlay/seams.patch" 2>/dev/null; then
        git -C "$dir" checkout --quiet --force --detach "$target"
        git -C "$dir" apply "$overlay/seams.patch"
    fi
    rsync -a --delete "$overlay/" "$dir/custom/"
    if lacking="$(lacking_marker "$dir")"; then
        printf 'error: %s at %s lacks %s, the learn-mode overlay needs updating for this ref\n' \
            "$dir/${lacking%%|*}" "$ref" "${lacking#*|}" >&2
        exit 1
    fi
    printf 'LearnHarness worktree at %s (%s) + learnharness-overlay\n' "$ref" "$(git -C "$dir" log -1 --format='%h %s')"

    cd -- "$dir"
    install_dependencies

    # Next writes .next/BUILD_ID before the build ends, a failed build can leave it: only the stamp counts. Its digest
    # covers what mtimes miss (a deleted overlay file, another commit, chat-ui, the frame ancestors), mtimes catch edited
    # sources.
    local build_stamp=.next/gigachad-build build_inputs
    build_inputs="$(
        {
            printf '%s\n' "$target" "$chat_ui_digest" "$frame_ancestors"
            cd -- "$overlay" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
        } | sha256sum | cut -d' ' -f1
    )"
    if [[ ! -f "$build_stamp" || "$(<"$build_stamp")" != "$build_inputs" ]] \
        || [[ -n "$(find . \( -path ./.next -o -path ./node_modules -o -path ./.git \) -prune -o -type f -newer "$build_stamp" -print -quit)" ]]; then
        printf 'Building LearnHarness (~1 min)\n'
        rm -f "$build_stamp"
        isolated npm run build
        printf '%s' "$build_inputs" >"$build_stamp"
    fi
}

# An update whose build fails goes back to the last working commit and is not tried again for this overlay. A pinned
# ref, the last working commit itself and an interrupt (status > 128) end the start instead.
set +e
(prepare "$target")
status=$?
if ((status != 0 && status < 128)) && [[ -z $pin && -n $working && $target != "$working" ]]; then
    mkdir -p "$check_results"
    printf '%s failed to build' "$ref" >"$check_results/$target-$overlay_digest"
    printf 'LearnHarness not updated, serving %s: %s failed to build\n' "$(describe "$working")" "$ref" >&2
    target="$working"
    ref="$(describe "$target")"
    (prepare "$target")
    status=$?
fi
set -e
if ((status != 0)); then
    exit "$status"
fi
cd -- "$dir"

# A ':' would end flatpak's path and start its access mode.
if [[ "$demo_tmp" == *:* ]]; then
    printf 'error: flatpak cannot share %s, its path holds a ":" (move GIGACHAD_LEARNHARNESS_HOME)\n' "$demo_tmp" >&2
    exit 1
fi
# HOME is the private one inside the service: user installations stay where the user keeps them.
chrome="$home/chrome"
printf '#!/usr/bin/env bash\nexport FLATPAK_USER_DIR=%q\nexec flatpak run --filesystem=%q %q "$@"\n' \
    "${XDG_DATA_HOME:-$HOME/.local/share}/flatpak" "$demo_tmp" "$chrome_app" >"$chrome"
chmod 700 "$chrome"

# Created here when the proxy launcher has not yet: both read the same file, which appears whole.
if ! master_key="$(GIGACHAD_LITELLM_PROXY_HOME="$proxy_home" PYTHONPATH="$root_dir/src" \
    "$root_dir/.venv/bin/python" -m lib.learnharness master-key)"; then
    master_key=""
fi

environment=(
    PATH="$PATH"
    HOME="$home/home"
    LANG="${LANG:-C.UTF-8}"
    NODE_ENV=production
    LH_DATA_DIR="$data_dir"
    LH_AUTH_FILE="$home/auth.json"
    TMPDIR="$demo_tmp"
    CHROME_PATH="$chrome"
    GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS="$frame_ancestors"
)
for name in XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS BRAVE_API_KEY; do
    if [[ -n "${!name:-}" ]]; then
        environment+=("$name=${!name}")
    fi
done
if [[ -n "$master_key" ]]; then
    environment+=(CUSTOM_OPENAI_API_KEY="$master_key")
else
    printf 'warning: no LiteLLM proxy master key in %s, LearnHarness has no models\n' "$proxy_home" >&2
fi

# Unreadable inside the unit ("-": skipped when absent): the user's credentials and secrets, the LiteLLM proxy's home
# (master key, gateway token), the logins of other agents and browsers (the unit's own Chrome keeps its profile in the
# private HOME). The main checkout's .env files too when this is a worktree.
hidden_paths=(
    "$HOME/.secrets" "$HOME/.ssh" "$HOME/.omp" "$HOME/.gnupg" "$HOME/.config/gh" "$HOME/.aws" "$HOME/.netrc"
    "$HOME/.git-credentials" "$proxy_home" "$HOME/.local/share/gigachad-bot/litellm-proxy"
    "$HOME/.claude" "$HOME/.claude.json" "$HOME/.codex" "$HOME/.config/opencode" "$HOME/.local/share/opencode"
    "$HOME/.local/share/keyrings" "$HOME/.password-store" "$HOME/.docker" "$HOME/.kube" "$HOME/.config/gcloud"
    "$HOME/.azure" "$HOME/.npmrc" "$HOME/.pypirc" "$HOME/.var/app" "$HOME/.config/google-chrome" "$HOME/.mozilla"
)
checkouts=("$root_dir")
if common_dir="$(git -C "$root_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    checkouts+=("$(dirname -- "$common_dir")")
fi
shopt -s nullglob
for checkout in "${checkouts[@]}"; do
    hidden_paths+=("$checkout"/.env*)
done
shopt -u nullglob
unit_properties=(-p UMask=0077)
for path in "${hidden_paths[@]}"; do
    unit_properties+=(-p "InaccessiblePaths=\"-$path\"")
done

printf 'Starting LearnHarness on %s (unit %s)\n' "$bind" "$unit"
# `env -i`: the unit would otherwise inherit the user manager's environment. The description replaces the default, the
# command line, which holds the master key and would land in the journal.
systemd-run --user --unit="$unit" --description="LearnHarness (gigachad-bot learn mode) on $bind" --collect --quiet --pipe \
    --working-directory="$dir" "${unit_properties[@]}" \
    -- env -i "${environment[@]}" "$dir/node_modules/.bin/next" start --hostname "${bind%%:*}" --port "${bind##*:}" &
runner=$!
# systemd-run leaves the unit running when it is killed: run.sh's cleanup (TERM) and Ctrl-C stop it here.
trap 'systemctl --user stop "$unit" 2>/dev/null || true' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

until port_open "$bind"; do
    kill -0 "$runner" 2>/dev/null || break
    sleep 0.2
done
if port_open "$bind"; then
    printf '%s' "$target" >"$served_commit_file"
fi
wait "$runner"
