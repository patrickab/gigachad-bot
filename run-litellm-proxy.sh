#!/usr/bin/env bash
# Serves the LiteLLM proxy, learn mode's model endpoint: LearnHarness's custom OpenAI provider reaches every model of
# gigachad's catalog through it.
#
#   config    `python -m lib.learnharness write-config` generates GIGACHAD_LITELLM_PROXY_HOME/config.yaml and
#             LearnHarness's model settings from one read of gigachad's catalog, right before the proxy starts. This is
#             their only generation point, so the proxy and LearnHarness's menus list the same models. When generation
#             fails (gateway down, unknown login, empty catalog) the last config and settings are served unchanged
#   auth      loopback only (127.0.0.1 or localhost), the master key (master.key, created once) is the bearer.
#             LearnHarness gets it as CUSTOM_OPENAI_API_KEY (run-learnharness.sh)
#   secrets   config.yaml holds none: it names LITELLM_MASTER_KEY and GIGACHAD_OMP_GATEWAY_TOKEN, exported here. Provider
#             keys come from this environment (~/.secrets), OMP models go to the OMP gateway
#   package   litellm[proxy] cannot share the project venv (it needs starlette >= 1, MinerU's gradio < 1, and rich < 14,
#             llm-baseclient >= 14.2), so `uv tool run` serves it from uv's cache: the LiteLLM version uv.lock pins, the
#             venv's Python, and the index as of a day after that LiteLLM's upload (`proxy-tool-pin`)
#
# Idempotent: a proxy already answering on the port is reused rather than restarted. `check-proxy` warns when it serves
# another model list or another master key than this home (started earlier, or from another home): stop it and rerun.
# A model added to gigachad's catalog reaches the proxy and LearnHarness together, on the proxy's next start.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir/deploy/lib.sh"
# run.sh loaded them already, a standalone start needs the provider keys and the database password too.
load_secrets
bind="${GIGACHAD_LITELLM_PROXY_BIND:-127.0.0.1:30183}"
home="${GIGACHAD_LITELLM_PROXY_HOME:-$HOME/.local/share/gigachad-bot/litellm-proxy}"
python="$root_dir/.venv/bin/python"

# The proxy's environment holds every provider key and the OMP gateway token.
case "${bind%:*}" in
    127.0.0.1 | localhost) ;;
    *)
        printf 'error: GIGACHAD_LITELLM_PROXY_BIND=%s, the LiteLLM proxy listens on 127.0.0.1 or localhost only\n' "$bind" >&2
        exit 1
        ;;
esac

generator() {
    GIGACHAD_LITELLM_PROXY_HOME="$home" GIGACHAD_LITELLM_PROXY_BIND="$bind" PYTHONPATH="$root_dir/src" \
        "$python" -m lib.learnharness "$@"
}

if port_open "$bind"; then
    printf 'Reusing the LiteLLM proxy already listening on %s\n' "$bind"
    generator check-proxy || true
    exit 0
fi

if ! generator write-config; then
    if [[ ! -f "$home/config.yaml" ]]; then
        printf 'error: no LiteLLM proxy config in %s, skipping the LiteLLM proxy\n' "$home" >&2
        exit 1
    fi
    printf 'Serving the last LiteLLM proxy config in %s\n' "$home" >&2
fi

LITELLM_MASTER_KEY="$(generator master-key)"
GIGACHAD_OMP_GATEWAY_TOKEN="$(PYTHONPATH="$root_dir/src" "$python" -c 'from lib import omp_source; print(omp_source.gateway_token())')"
export LITELLM_MASTER_KEY GIGACHAD_OMP_GATEWAY_TOKEN
pin="$(generator proxy-tool-pin "$root_dir/uv.lock")"
read -r requirement python_version exclude_newer <<<"$pin"
printf 'Starting the LiteLLM proxy (%s) on %s\n' "$requirement" "$bind"
# Outside the project, so uv resolves the tool alone.
cd -- "$home"
exec uv tool run --python "$python_version" --exclude-newer "$exclude_newer" --from "$requirement" \
    litellm --config "$home/config.yaml" --host "${bind%:*}" --port "${bind##*:}" --telemetry False
