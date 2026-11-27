#!/usr/bin/env bash
# Rebases the production worktree onto origin/master and restarts the production
# backend (gigachad-bot.service, which also starts the C4 service) from a clean
# environment, so nothing inherited from the calling shell reaches uv, npm or the unit.
#
# Before the backend, pi-host (gigachad-bot-pi-host.service, M14): restarted only when
# pi-host/ changed (code, its package.json, the lockfile), the installed pi
# is another version than the one it runs, or it is down, never while a Pi session is
# running a turn (M4, `running` in pi-host's /health; idle and pinned sessions resume after it).
# Whatever pi version is installed runs: Pi updates itself (mise), nothing pins it here.
#
# Then learn mode, unless a learn session is busy: restarts LearnHarness when the commit
# its launcher would serve moved (the newest release tag passing the overlay checks, or
# the pinned GIGACHAD_LEARNHARNESS_REF: run-learnharness.sh --select) or
# learnharness-overlay/, the chat-ui package or run-learnharness.sh changed (the launcher
# snapshots its data before moving to another commit, and regenerates learn mode's models
# from the catalog), else regenerates the models alone. Then restarts its Tailnet gate and
# starts the backup timer.
set -euo pipefail

unit=gigachad-bot.service
prod="${GIGACHAD_PROD_DIR:-$HOME/git/gigachad-bot-prod}"
port=8002

pi_host_unit=gigachad-bot-pi-host.service
pi_host_url=http://127.0.0.1:8301

learn_unit=gigachad-bot-learnharness.service
learn_gate_unit=gigachad-bot-learnharness-gate.service

preview_gate_unit=gigachad-bot-preview-gate.service
preview_gate_url=http://127.0.0.1:30185

if [[ -z ${GIGACHAD_REDEPLOY_FRESH:-} ]]; then
    exec env -i \
        GIGACHAD_REDEPLOY_FRESH=1 GIGACHAD_PROD_DIR="$prod" \
        HOME="$HOME" USER="$(id -un)" LANG="${LANG:-C.UTF-8}" \
        PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin" \
        XDG_RUNTIME_DIR="/run/user/$(id -u)" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
        SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/run/user/$(id -u)/ssh-agent.socket}" \
        bash --noprofile --norc "${BASH_SOURCE[0]}" "$@"
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/deploy/lib.sh"

if [[ $# -ne 0 ]]; then
    printf "usage: %s\n" "$0" >&2
    exit 2
fi
if [[ -n $(git -C "$prod" status --porcelain --untracked-files=no) ]]; then
    printf "Production worktree has uncommitted changes: %s\n" "$prod" >&2
    exit 1
fi

backend_up() {
    systemctl --user is-active --quiet "$unit" && curl -s -o /dev/null "http://127.0.0.1:$port/"
}

# A setting from the production env file the units read, else the launcher's default.
prod_setting() {
    (source "$prod/deploy/load-env.sh" && printf '%s' "${!1:-$2}")
}

learn_up() {
    systemctl --user is-active --quiet "$learn_unit" && curl -sf -o /dev/null "$learn_url/"
}

learn_gate_up() {
    systemctl --user is-active --quiet "$learn_gate_unit" && [[ $(curl -s -o /dev/null -w '%{http_code}' "$learn_gate_url/") == 403 ]]
}

preview_gate_up() {
    systemctl --user is-active --quiet "$preview_gate_unit" && [[ $(curl -s -o /dev/null -w '%{http_code}' "$preview_gate_url/") == 403 ]]
}

busy_learn_sessions() {
    curl -sf "$learn_url/api/chats" | python3 -c 'import json, sys; print(sum(1 for chat in json.load(sys.stdin).get("chats", []) if chat.get("busy")))'
}

# The Pi version the running pi-host checked at its start (M13), empty when it does not answer.
pi_host_version() {
    curl -sf -H "Authorization: Bearer $(cat "$pi_host_token_file")" "$pi_host_url/health" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["piVersion"])'
}

pi_host_up() {
    systemctl --user is-active --quiet "$pi_host_unit" && [[ -n $(pi_host_version 2>/dev/null) ]]
}

# Live factory runners, one per task at work: `bun .../agent-factory/src/runner.ts`.
# The factory starts them outside pi-host's control group, but one started without systemd-run would
# end with a restart, so they count as running sessions.
factory_runners() {
    pgrep -cf '/agent-factory/src/runner\.ts' || true
}

# Pi sessions running a turn per pi-host's /health plus the live factory runners; a restart ends
# them. A pi-host that predates the session list in /health falls back to counting its `pi`
# children (every live session).
running_pi_sessions() {
    local answer running cgroup pid count
    count="$(factory_runners)"
    answer="$(curl -sf -H "Authorization: Bearer $(cat "$pi_host_token_file")" "$pi_host_url/health" 2>/dev/null || true)"
    running="$(printf '%s' "$answer" | python3 -c 'import json, sys
sessions = json.load(sys.stdin).get("sessions")
print("" if sessions is None else sum(1 for session in sessions if session.get("running")))' 2>/dev/null || true)"
    if [[ -n $running ]]; then
        printf '%s\n' "$((running + count))"
        return
    fi
    cgroup="$(systemctl --user show -p ControlGroup --value "$pi_host_unit")"
    if [[ -n $cgroup && -r /sys/fs/cgroup$cgroup/cgroup.procs ]]; then
        while read -r pid; do
            if [[ $(cat "/proc/$pid/comm" 2>/dev/null) == pi ]]; then
                count=$((count + 1))
            fi
        done <"/sys/fs/cgroup$cgroup/cgroup.procs"
    fi
    printf '%s\n' "$count"
}

old="$(git -C "$prod" rev-parse HEAD)"
git -C "$prod" fetch origin master
git -C "$prod" rebase origin/master
git -C "$prod" submodule update --init
new="$(git -C "$prod" rev-parse HEAD)"

if [[ $old == "$new" ]]; then
    printf "Already at %s, restarting anyway\n" "$(git -C "$prod" rev-parse --short HEAD)"
else
    git -C "$prod" --no-pager log --oneline "$old..$new"
fi

cd "$prod"
uv sync --python 3.12 --no-dev
if [[ ! -d src/c4/node_modules ]] || ! git diff --quiet "$old" "$new" -- src/c4/package-lock.json; then
    npm ci --prefix src/c4 --no-audit --no-fund
fi
# run-learnharness.sh packs @gigachad/chat-ui with src/frontend's tsc and types. Scripts stay off, only tsc runs.
if [[ ! -d src/frontend/node_modules ]] || ! git diff --quiet "$old" "$new" -- src/frontend/package-lock.json; then
    npm ci --prefix src/frontend --ignore-scripts --no-audit --no-fund
fi
if ! git diff --quiet "$old" "$new" -- deploy; then
    printf "deploy/ changed: run deploy/install-systemd-service.sh to reinstall the units\n" >&2
fi

# pi-host, before the backend that talks to it
if ! systemctl --user cat "$pi_host_unit" >/dev/null 2>&1; then
    printf "%s is not installed: run %s/deploy/install-systemd-service.sh\n" "$pi_host_unit" "$prod" >&2
    exit 1
fi
pi_host_token_file="$(prod_setting PI_HOST_TOKEN_FILE "$HOME/.local/state/gigachad-bot/pi-host.token")"
pi_installed="$(timeout 30 pi --version 2>/dev/null || true)"
pi_served="$(pi_host_version 2>/dev/null || true)"
if [[ -z $pi_served || $pi_served != "$pi_installed" ]] || ! git diff --quiet "$old" "$new" -- pi-host || ! pi_host_up; then
    if [[ -z $pi_installed ]]; then
        printf "pi-host not restarted: no pi on PATH (mise), it keeps serving %s\n" "${pi_served:-nothing}" >&2
        exit 1
    fi
    running="$(running_pi_sessions)"
    if [[ $running != 0 ]]; then
        printf "pi-host not restarted: %s Pi session(s) running, run systemctl --user restart %s when they are idle\n" \
            "$running" "$pi_host_unit" >&2
    else
        printf "Restarting %s with Pi %s\n" "$pi_host_unit" "$pi_installed"
        systemctl --user restart "$pi_host_unit"
        # The first start installs pi-host's dependencies (bun install).
        if ! wait_until 300 1 pi_host_up; then
            journalctl --user -u "$pi_host_unit" -n 30 --no-pager -o cat >&2 || true
            printf "%s did not come up within 5 min, see the log above\n" "$pi_host_unit" >&2
            exit 1
        fi
    fi
fi
if ! pi_host_up; then
    systemctl --user --no-pager status "$pi_host_unit" >&2 || true
    printf "%s is not up on %s\n" "$pi_host_unit" "$pi_host_url" >&2
    exit 1
fi
printf "%s is up on %s with Pi %s\n" "$pi_host_unit" "$pi_host_url" "$(pi_host_version)"

systemctl --user restart "$unit"
if ! wait_until 30 1 backend_up; then
    systemctl --user --no-pager status "$unit" >&2 || true
    printf "%s did not come up within 30s\n" "$unit" >&2
    exit 1
fi
printf "%s is up on 127.0.0.1:%s at %s\n" "$unit" "$port" "$(git rev-parse --short HEAD)"

# Learn mode
if ! systemctl --user cat "$learn_unit" "$learn_gate_unit" gigachad-bot-learnharness-backup.timer >/dev/null 2>&1; then
    printf "learn mode units are not installed: run %s/deploy/install-systemd-service.sh\n" "$prod" >&2
    exit 1
fi
learn_repo="$(prod_setting GIGACHAD_LEARNHARNESS_REPO "$HOME/git/learnharness")"
learn_home="$(prod_setting GIGACHAD_LEARNHARNESS_HOME "$HOME/.local/share/gigachad-bot/learnharness")"
learn_ref="$(prod_setting GIGACHAD_LEARNHARNESS_REF "")"
learn_url="http://$(prod_setting GIGACHAD_LEARNHARNESS_BIND 127.0.0.1:30181)"
learn_gate_url="http://127.0.0.1:$(prod_setting GIGACHAD_LEARNHARNESS_GATE_PORT 30182)"
# The launcher's own choice, from the unit's settings: unpinned, it fetches the release tags and runs the overlay checks
# against a newer one (minutes, results kept) while LearnHarness keeps serving. A pinned remote-tracking ref
# (origin/main) is fetched here, the launcher only fetches a ref it lacks.
if [[ -n $learn_ref && -d $learn_repo/.git ]] \
    && git -C "$learn_repo" rev-parse --verify --quiet "refs/remotes/$learn_ref" >/dev/null; then
    git -C "$learn_repo" fetch "${learn_ref%%/*}"
fi
learn_new="$( (source "$prod/deploy/load-env.sh" && "$prod/run-learnharness.sh" --select) || true)"
learn_old="$(cat "$learn_home/served-commit" 2>/dev/null || true)"

busy="$(busy_learn_sessions 2>/dev/null || echo 0)"
# What the chat-ui tarball is made of: only these change what LearnHarness is built from.
chat_ui=src/frontend/packages/chat-ui
chat_ui_shipped=("$chat_ui/package.json" "$chat_ui/pack.mjs" "$chat_ui/tsconfig.build.json" "$chat_ui/src/learn-protocol.ts"
    "$chat_ui/src/markdown.css" "$chat_ui/src/palette.css" "$chat_ui/src/fonts.css" "$chat_ui/src/fonts")
if [[ $busy != 0 ]]; then
    # A restart ends a running turn.
    printf "Learn mode not restarted: %s learn session(s) busy, run systemctl --user restart %s when they are idle\n" \
        "$busy" "$learn_unit" >&2
else
    if [[ -z $learn_new || $learn_old != "$learn_new" ]] \
        || ! git diff --quiet "$old" "$new" -- learnharness-overlay "${chat_ui_shipped[@]}" run-learnharness.sh || ! learn_up; then
        if [[ -n $learn_old && -n $learn_new && $learn_old != "$learn_new" ]]; then
            git -C "$learn_repo" --no-pager log --oneline "$learn_old..$learn_new" || true
            printf "LearnHarness moves to another commit: its data is snapshotted to %s/snapshots/ first\n" "$learn_home"
        fi
        printf "Restarting %s at %s (a change rebuilds, about 2 min)\n" "$learn_unit" "${learn_ref:-release ${learn_new:0:7}}"
        systemctl --user restart "$learn_unit"
        if ! wait_until 600 1 learn_up; then
            systemctl --user --no-pager status "$learn_unit" >&2 || true
            printf "%s did not come up within 10 min\n" "$learn_unit" >&2
            exit 1
        fi
    # A LearnHarness start regenerates the models, a running one gets them here (a failure keeps the last ones).
    elif ! "$prod/deploy/run-production-learnharness.sh" models; then
        printf "LearnHarness's models not regenerated, it keeps the last ones\n" >&2
    fi
fi
printf "%s is up on %s at %s\n" "$learn_unit" "$learn_url" "$(cut -c1-7 "$learn_home/served-commit" 2>/dev/null || echo unknown)"

systemctl --user restart "$learn_gate_unit"
if ! wait_until 30 1 learn_gate_up; then
    systemctl --user --no-pager status "$learn_gate_unit" >&2 || true
    printf "%s did not come up within 30s\n" "$learn_gate_unit" >&2
    exit 1
fi
printf "%s is up on %s\n" "$learn_gate_unit" "$learn_gate_url"
systemctl --user start gigachad-bot-learnharness-backup.timer

# Factory previews
if ! systemctl --user cat "$preview_gate_unit" >/dev/null 2>&1; then
    printf "%s is not installed: run %s/deploy/install-systemd-service.sh\n" "$preview_gate_unit" "$prod" >&2
    exit 1
fi
systemctl --user restart "$preview_gate_unit"
if ! wait_until 30 1 preview_gate_up; then
    systemctl --user --no-pager status "$preview_gate_unit" >&2 || true
    printf "%s did not come up within 30s\n" "$preview_gate_unit" >&2
    exit 1
fi
printf "%s is up on %s\n" "$preview_gate_unit" "$preview_gate_url"
