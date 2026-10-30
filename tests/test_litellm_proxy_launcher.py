"""Exercise run-litellm-proxy.sh against a fake project python and uv: config generation, fallback, secrets, reuse."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shutil
import socket
import threading

from tests.shell import REPOSITORY, environment_with, fake_command, run_script

PIN = "litellm[proxy]==1.98.0 3.12 2026-08-23T22:19:21Z"
# The project venv's python: write-config succeeds unless WRITE_CONFIG_FAILS, check-proxy fails when PROXY_STALE.
FAKE_PYTHON = f"""case "$*" in
    *gateway_token*) echo gateway-token; exit ;;
    *master-key*) echo sk-master; exit ;;
    *proxy-tool-pin*) echo "{PIN}"; exit ;;
esac
echo "$* bind=$GIGACHAD_LITELLM_PROXY_BIND home=$GIGACHAD_LITELLM_PROXY_HOME src=$PYTHONPATH secret=${{SECRET_FROM_FILE:-}}" >>"$LOG"
if [[ $* == *check-proxy* ]]; then
    [[ -z ${{PROXY_STALE:-}} ]] || {{ echo 'warning: the LiteLLM proxy is stale or mismatched' >&2; exit 1; }}
    exit
fi
[[ -z ${{WRITE_CONFIG_FAILS:-}} ]] || exit 1
echo 'model_list: []' >"$GIGACHAD_LITELLM_PROXY_HOME/config.yaml"
"""
FAKE_UV = 'echo "uv $* cwd=$PWD key=$LITELLM_MASTER_KEY token=$GIGACHAD_OMP_GATEWAY_TOKEN path=${PYTHONPATH:-}" >>"$LOG"\n'


def launcher_copy(tmp_path: Path) -> Path:
    root = tmp_path / "gigachad-bot"
    (root / "deploy").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPOSITORY / "run-litellm-proxy.sh", root)
    shutil.copy(REPOSITORY / "deploy" / "lib.sh", root / "deploy")
    (root / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    fake_command(root / ".venv" / "bin", "python", FAKE_PYTHON)
    return root


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def launch(tmp_path: Path, bind: str = "127.0.0.1:1", **variables: str):
    root = launcher_copy(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir(exist_ok=True)
    fake_command(binaries, "uv", FAKE_UV)
    home = tmp_path / "proxy"
    home.mkdir(exist_ok=True)
    user = tmp_path / "user"
    user.mkdir(exist_ok=True)
    (user / ".secrets").write_text("export SECRET_FROM_FILE=loaded\n")
    log = tmp_path / "log"
    environment = environment_with(
        binaries, HOME=str(user), GIGACHAD_LITELLM_PROXY_HOME=str(home), GIGACHAD_LITELLM_PROXY_BIND=bind, LOG=str(log), **variables
    )
    for name in ("PYTHONPATH", "LITELLM_MASTER_KEY", "GIGACHAD_OMP_GATEWAY_TOKEN"):
        environment.pop(name, None)
    result = run_script(root / "run-litellm-proxy.sh", environment, timeout=15)
    return result, root, home, log.read_text().splitlines() if log.exists() else []


def test_proxy_serves_a_fresh_config_on_loopback_with_its_secrets_in_the_environment(tmp_path):
    result, root, home, log = launch(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    # ~/.secrets is loaded for a standalone start. config.yaml names the secrets, uv gets them, never the project's src.
    assert log == [
        f"-m lib.learnharness write-config bind=127.0.0.1:1 home={home} src={root / 'src'} secret=loaded",
        "uv tool run --python 3.12 --exclude-newer 2026-08-23T22:19:21Z --from litellm[proxy]==1.98.0 "
        f"litellm --config {home / 'config.yaml'} --host 127.0.0.1 --port 1 --telemetry False "
        f"cwd={home} key=sk-master token=gateway-token path=",
    ]


def test_a_failed_generation_serves_the_last_config_and_without_one_nothing(tmp_path):
    result, _, home, log = launch(tmp_path, WRITE_CONFIG_FAILS="1")
    assert result.returncode == 1
    assert "no LiteLLM proxy config" in result.stderr
    assert not any(line.startswith("uv ") for line in log)

    (home / "config.yaml").write_text("model_list: []\n")
    result, _, _, log = launch(tmp_path, WRITE_CONFIG_FAILS="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Serving the last LiteLLM proxy config" in result.stderr
    assert log[-1].startswith("uv tool run --python 3.12 --exclude-newer 2026-08-23T22:19:21Z --from litellm[proxy]==1.98.0")
    assert (home / "config.yaml").read_text() == "model_list: []\n"


def test_only_a_loopback_bind_is_served(tmp_path):
    result, _, _, log = launch(tmp_path, bind="0.0.0.0:30183")
    assert result.returncode == 1
    assert "listens on 127.0.0.1 or localhost only" in result.stderr
    assert log == []


class _Listening(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_a_running_proxy_is_reused_and_checked_never_regenerated(tmp_path):
    port = free_port()
    server = HTTPServer(("127.0.0.1", port), _Listening)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result, root, home, log = launch(tmp_path, bind=f"127.0.0.1:{port}")
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout == f"Reusing the LiteLLM proxy already listening on 127.0.0.1:{port}\n"
        assert log == [f"-m lib.learnharness check-proxy bind=127.0.0.1:{port} home={home} src={root / 'src'} secret=loaded"]

        # A proxy serving another model list or master key is reported, and still left running.
        result, _, _, log = launch(tmp_path, bind=f"127.0.0.1:{port}", PROXY_STALE="1")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "stale or mismatched" in result.stderr
        assert not any("write-config" in line or line.startswith("uv ") for line in log)
    finally:
        server.shutdown()
