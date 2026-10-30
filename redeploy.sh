#!/usr/bin/env bash
# Rebases the production worktree onto origin/master and restarts the production
# backend (gigachad-bot.service, which also starts the C4 service) from a clean
# environment, so nothing inherited from the calling shell reaches uv, npm or the unit.
#
# Then omp-web (omp tabs): restarts gigachad-bot-omp-web.service when the upstream ref
# it serves (GIGACHAD_OMP_WEB_REF) moved, origin/main moved past it (the launcher rebases on start),
# or ompweb-overlay/ or the chat-ui package
# changed, never while an omp session runs, and restarts its Tailnet gate. The unit's
# run-omp-web.sh resets the served worktree to the ref and applies the overlay on start.
#
# Then learn mode, unless a learn session is busy: restarts the LiteLLM proxy (its start
# regenerates learn mode's model list from the catalog), restarts LearnHarness when the
# commit its launcher would serve moved (the newest release tag passing the overlay
# checks, or the pinned GIGACHAD_LEARNHARNESS_REF: run-learnharness.sh --select) or
# learnharness-overlay/, the chat-ui package or run-learnharness.sh changed (the launcher
# snapshots its data before moving to another commit), restarts its Tailnet gate and
# starts the backup timer.
set -euo pipefail

unit=gigachad-bot.service
prod="${GIGACHAD_PROD_DIR:-$HOME/git/gigachad-bot-prod}"
port=8002

omp_web_unit=gigachad-bot-omp-web.service
omp_web_gate_unit=gigachad-bot-omp-web-gate.service
omp_web_remote=https://github.com/kahme247/ompweb.git
omp_web_url=http://127.0.0.1:30177
omp_web_gate_url=http://127.0.0.1:30179

proxy_unit=gigachad-bot-litellm-proxy.service
learn_unit=gigachad-bot-learnharness.service
learn_gate_unit=gigachad-bot-learnharness-gate.service

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

omp_web_up() {
    systemctl --user is-active --quiet "$omp_web_unit" && curl -sf -o /dev/null "$omp_web_url/login"
}

# The gate answers 403 to a request without Tailscale login, that is the healthy answer.
omp_web_gate_up() {
    systemctl --user is-active --quiet "$omp_web_gate_unit" && [[ $(curl -s -o /dev/null -w '%{http_code}' "$omp_web_gate_url/") == 403 ]]
}

# A setting from the production env file the units read, else the launcher's default.
prod_setting() {
    (source "$prod/deploy/load-env.sh" && printf '%s' "${!1:-$2}")
}

running_omp_sessions() {
    curl -sf "$omp_web_url/api/sessions" | python3 -c 'import json, sys; print(len(json.load(sys.stdin).get("runningSessionIds", [])))'
}

proxy_up() {
    systemctl --user is-active --quiet "$proxy_unit" && port_open "$proxy_bind"
}

learn_up() {
    systemctl --user is-active --quiet "$learn_unit" && curl -sf -o /dev/null "$learn_url/"
}

learn_gate_up() {
    systemctl --user is-active --quiet "$learn_gate_unit" && [[ $(curl -s -o /dev/null -w '%{http_code}' "$learn_gate_url/") == 403 ]]
}

busy_learn_sessions() {
    curl -sf "$learn_url/api/chats" | python3 -c 'import json, sys; print(sum(1 for chat in json.load(sys.stdin).get("chats", []) if chat.get("busy")))'
}

old="$(git -C "$prod" rev-parse HEAD)"
git -C "$prod" fetch origin master
git -C "$prod" rebase origin/master
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
# run-omp-web.sh packs @gigachad/chat-ui with src/frontend's tsc and types. Scripts stay off, only tsc runs.
if [[ ! -d src/frontend/node_modules ]] || ! git diff --quiet "$old" "$new" -- src/frontend/package-lock.json; then
    npm ci --prefix src/frontend --ignore-scripts --no-audit --no-fund
fi
if ! git diff --quiet "$old" "$new" -- deploy; then
    printf "deploy/ changed: run deploy/install-systemd-service.sh to reinstall the units\n" >&2
fi

systemctl --user restart "$unit"
if ! wait_until 30 1 backend_up; then
    systemctl --user --no-pager status "$unit" >&2 || true
    printf "%s did not come up within 30s\n" "$unit" >&2
    exit 1
fi
printf "%s is up on 127.0.0.1:%s at %s\n" "$unit" "$port" "$(git rev-parse --short HEAD)"

# omp-web
if ! systemctl --user cat "$omp_web_unit" "$omp_web_gate_unit" >/dev/null 2>&1; then
    printf "omp-web units are not installed: run %s/deploy/install-systemd-service.sh\n" "$prod" >&2
    exit 1
fi
omp_web_repo="$(prod_setting GIGACHAD_OMP_WEB_REPO "$HOME/git/ompweb_original")"
omp_web_dir="$(prod_setting GIGACHAD_OMP_WEB_DIR "$HOME/.local/share/gigachad-bot/ompweb-upstream")"
omp_web_ref="$(prod_setting GIGACHAD_OMP_WEB_REF integration_patrick_ab)"
if [[ ! -d $omp_web_repo/.git ]]; then
    printf "No omp-web checkout at %s: git clone %s %s\n" "$omp_web_repo" "$omp_web_remote" "$omp_web_repo" >&2
    exit 1
fi
if git -C "$omp_web_repo" rev-parse --verify --quiet "refs/remotes/$omp_web_ref" >/dev/null; then
    git -C "$omp_web_repo" fetch "${omp_web_ref%%/*}"
fi
omp_web_new="$(git -C "$omp_web_repo" rev-parse "$omp_web_ref^{commit}")"
omp_web_old="$(git -C "$omp_web_dir" rev-parse HEAD 2>/dev/null || true)"

# A new origin/main alone also restarts: run-omp-web.sh rebases the branch onto it on start. Not when its
# auto-update already failed for this branch and origin/main (the stamp's key starts with both), nor offline.
omp_web_behind=false
if [[ $(prod_setting GIGACHAD_AUTO_UPDATE 1) != 0 ]] \
    && git -C "$omp_web_repo" show-ref --verify --quiet "refs/heads/$omp_web_ref" \
    && GIT_TERMINAL_PROMPT=0 timeout 60 git -C "$omp_web_repo" fetch --quiet origin </dev/null 2>/dev/null \
    && omp_web_main="$(git -C "$omp_web_repo" rev-parse --verify --quiet refs/remotes/origin/main)" \
    && ! git -C "$omp_web_repo" merge-base --is-ancestor "$omp_web_main" "$omp_web_new"; then
    omp_web_failed_key="$(head -n1 "$omp_web_dir.auto-update-failed" 2>/dev/null || true)"
    if [[ $omp_web_failed_key != "$omp_web_new $omp_web_main "* ]]; then
        omp_web_behind=true
    fi
fi

if [[ $omp_web_old != "$omp_web_new" ]] || $omp_web_behind || ! git diff --quiet "$old" "$new" -- ompweb-overlay src/frontend/packages/chat-ui || ! omp_web_up; then
    running="$(running_omp_sessions 2>/dev/null || echo 0)"
    if [[ $running != 0 ]]; then
        printf "omp-web not restarted: %s omp session(s) running, run systemctl --user restart %s when they are idle\n" \
            "$running" "$omp_web_unit" >&2
    else
        if [[ -n $omp_web_old && $omp_web_old != "$omp_web_new" ]]; then
            git -C "$omp_web_repo" --no-pager log --oneline "$omp_web_old..$omp_web_new" || true
        fi
        printf "Restarting %s at %s (a change rebuilds, about 2 min)\n" "$omp_web_unit" "$omp_web_ref"
        systemctl --user restart "$omp_web_unit"
        if ! wait_until 600 1 omp_web_up; then
            systemctl --user --no-pager status "$omp_web_unit" >&2 || true
            printf "%s did not come up within 10 min\n" "$omp_web_unit" >&2
            exit 1
        fi
    fi
fi
printf "%s is up on %s at %s\n" "$omp_web_unit" "$omp_web_url" "$(git -C "$omp_web_dir" rev-parse --short HEAD)"

systemctl --user restart "$omp_web_gate_unit"
if ! wait_until 30 1 omp_web_gate_up; then
    systemctl --user --no-pager status "$omp_web_gate_unit" >&2 || true
    printf "%s did not come up within 30s\n" "$omp_web_gate_unit" >&2
    exit 1
fi
printf "%s is up on %s\n" "$omp_web_gate_unit" "$omp_web_gate_url"

# Learn mode
if ! systemctl --user cat "$proxy_unit" "$learn_unit" "$learn_gate_unit" gigachad-bot-learnharness-backup.timer >/dev/null 2>&1; then
    printf "learn mode units are not installed: run %s/deploy/install-systemd-service.sh\n" "$prod" >&2
    exit 1
fi
learn_repo="$(prod_setting GIGACHAD_LEARNHARNESS_REPO "$HOME/git/learnharness")"
learn_home="$(prod_setting GIGACHAD_LEARNHARNESS_HOME "$HOME/.local/share/gigachad-bot/learnharness")"
learn_ref="$(prod_setting GIGACHAD_LEARNHARNESS_REF "")"
learn_url="http://$(prod_setting GIGACHAD_LEARNHARNESS_BIND 127.0.0.1:30181)"
proxy_bind="$(prod_setting GIGACHAD_LITELLM_PROXY_BIND 127.0.0.1:30183)"
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
if [[ $busy != 0 ]]; then
    # A turn's model calls go through the proxy, a restart of either ends it.
    printf "Learn mode not restarted: %s learn session(s) busy, run systemctl --user restart %s %s when they are idle\n" \
        "$busy" "$proxy_unit" "$learn_unit" >&2
else
    # Its start is the one point that regenerates the model list from the catalog, so it restarts on every redeploy.
    systemctl --user restart "$proxy_unit"
    if ! wait_until 300 1 proxy_up; then
        systemctl --user --no-pager status "$proxy_unit" >&2 || true
        printf "%s did not come up within 5 min\n" "$proxy_unit" >&2
        exit 1
    fi
    printf "%s is up on %s\n" "$proxy_unit" "$proxy_bind"

    if [[ -z $learn_new || $learn_old != "$learn_new" ]] \
        || ! git diff --quiet "$old" "$new" -- learnharness-overlay src/frontend/packages/chat-ui run-learnharness.sh || ! learn_up; then
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
