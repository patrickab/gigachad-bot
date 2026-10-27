#!/usr/bin/env bash
# Rebases the production worktree onto origin/master and restarts the production
# backend (gigachad-bot.service, which also starts the C4 service) from a clean
# environment, so nothing inherited from the calling shell reaches uv, npm or the unit.
#
# Then omp-web (omp tabs): restarts gigachad-bot-omp-web.service when the upstream ref
# it serves (GIGACHAD_OMP_WEB_REF) moved or ompweb-overlay/ or the chat-ui package
# changed, never while an omp session runs, and restarts its Tailnet gate. The unit's
# run-omp-web.sh resets the served worktree to the ref and applies the overlay on start.
set -euo pipefail

unit=gigachad-bot.service
prod="${GIGACHAD_PROD_DIR:-$HOME/git/gigachad-bot-prod}"
port=8002

omp_web_unit=gigachad-bot-omp-web.service
omp_web_gate_unit=gigachad-bot-omp-web-gate.service
omp_web_remote=https://github.com/kahme247/ompweb.git
omp_web_url=http://127.0.0.1:30177
omp_web_gate_url=http://127.0.0.1:30179

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

# A setting from the production env file the omp-web unit reads, else the launcher's default.
omp_web_setting() {
    (source "$prod/deploy/load-env.sh" && printf '%s' "${!1:-$2}")
}

running_omp_sessions() {
    curl -sf "$omp_web_url/api/sessions" | python3 -c 'import json, sys; print(len(json.load(sys.stdin).get("runningSessionIds", [])))'
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
omp_web_repo="$(omp_web_setting GIGACHAD_OMP_WEB_REPO "$HOME/git/ompweb_original")"
omp_web_dir="$(omp_web_setting GIGACHAD_OMP_WEB_DIR "$HOME/.local/share/gigachad-bot/ompweb-upstream")"
omp_web_ref="$(omp_web_setting GIGACHAD_OMP_WEB_REF integration_patrick_ab)"
if [[ ! -d $omp_web_repo/.git ]]; then
    printf "No omp-web checkout at %s: git clone %s %s\n" "$omp_web_repo" "$omp_web_remote" "$omp_web_repo" >&2
    exit 1
fi
if git -C "$omp_web_repo" rev-parse --verify --quiet "refs/remotes/$omp_web_ref" >/dev/null; then
    git -C "$omp_web_repo" fetch "${omp_web_ref%%/*}"
fi
omp_web_new="$(git -C "$omp_web_repo" rev-parse "$omp_web_ref^{commit}")"
omp_web_old="$(git -C "$omp_web_dir" rev-parse HEAD 2>/dev/null || true)"

if [[ $omp_web_old != "$omp_web_new" ]] || ! git diff --quiet "$old" "$new" -- ompweb-overlay src/frontend/packages/chat-ui || ! omp_web_up; then
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
