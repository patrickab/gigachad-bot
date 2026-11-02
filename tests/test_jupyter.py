"""Jupyter notebooks on the host: kernelspecs, kernels per notebook, the proxy to the private Jupyter Server and the document hooks.

Most tests share one real Jupyter Server (about two seconds to start) with the app's own `ipykernel`, venvs are
wrappers around the test interpreter.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from backend import websocket_origin
from backend.identity import get_request_identity
from backend.routes import documents as documents_route
from backend.routes import jupyter as jupyter_route
from backend.routes.deps import (
    get_asset_store,
    get_disk_files,
    get_document_store,
    get_file_roots,
    get_project_store,
    get_references,
    get_relocate,
)
import config
from lib import jupyter
from lib.asset_store import AssetStore
from lib.changes import UserScope
from lib.chat_store import ChatStore
from lib.disk_files import DiskFiles
from lib.jupyter import JupyterHost
from lib.postgres_data_store import PostgresDataStore
from lib.project_rename import PostgresProjectRenamer
from lib.project_store import ProjectStore
from lib.references import References
from lib.upstream import UpstreamContractError, UpstreamRequestFailed, forwardable_headers
from tests.conftest import create_user_and_device, mounted_roots, request_identity, route_client
from tests.fakes import MemoryAssetStore, MemoryDataStore, RecordingScope

ORIGIN = "http://localhost:2999"
SERVER = "/api/jupyter/server"


def executable(path: Path, script: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def install_ipykernel(venv: Path) -> None:
    """Where `ipykernel` lives in a venv: the file the host looks for, nothing of it is run."""
    package = venv / "lib" / "python3.13" / "site-packages" / "ipykernel"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()


def fake_venv(folder: Path, *, ipykernel: bool = True) -> Path:
    """`folder/.venv` whose python is the test interpreter, with or without `ipykernel` installed."""
    python = folder / ".venv" / "bin" / "python"
    executable(python, f'exec {sys.executable} "$@"')
    if ipykernel:
        install_ipykernel(folder / ".venv")
    return python


def notebook(path: Path, kernelspec: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"kernelspec": {"name": kernelspec, "display_name": kernelspec}} if kernelspec else {}
    path.write_text(json.dumps({"cells": [], "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    return path


def names(specs: jupyter.KernelSpecList) -> list[str]:
    return [spec.name for spec in specs.specs]


def client_for(host: JupyterHost, *, host_access: bool = True):
    """A client whose requests come from the app's own page (`ORIGIN`), `sent_from` changes that."""
    overrides = {
        get_request_identity: request_identity(host_access),
        get_file_roots: mounted_roots(host_access=host_access),
        jupyter_route.get_jupyter_host: host,
    }
    client = route_client(jupyter_route.router, overrides=overrides)
    client.headers["origin"] = ORIGIN
    return client


@contextmanager
def sent_from(client, origin: str | None):
    """Run requests of *client* as sent by a page of *origin*, or by something that sends no `Origin`."""
    previous = client.headers.get("origin")
    if origin is None:
        client.headers.pop("origin", None)
    else:
        client.headers["origin"] = origin
    try:
        yield client
    finally:
        if previous is None:
            client.headers.pop("origin", None)
        else:
            client.headers["origin"] = previous


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home directory that has no `.venv` anywhere above its projects."""
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def offline_host(tmp_path, home, monkeypatch):
    """A host answering the kernelspec list of a server with only `python3`, without starting one."""
    host = JupyterHost(tmp_path / "jupyter")
    discovered = jupyter._UpstreamSpecs(
        default="python3", kernelspecs={"python3": {"name": "python3", "spec": {"display_name": "Python 3", "language": "python"}}}
    )
    monkeypatch.setattr(host, "_discovered_specs", lambda: discovered)
    return host


def test_venvs_of_the_folder_and_its_ancestors_up_to_home_become_specs_nearest_first(offline_host, home, tmp_path):
    fake_venv(tmp_path.resolve())  # above $HOME, never searched
    fake_venv(home)
    fake_venv(home / "proj")
    nb = notebook(home / "proj" / "sub" / "a.ipynb")

    specs = offline_host.kernelspecs(str(nb))

    assert [spec.display_name for spec in specs.specs] == ["proj/.venv", "home/.venv", "Python 3"]
    assert names(specs)[0].startswith("venv-") and names(specs)[0] != names(specs)[1]
    assert specs.default == names(specs)[0]
    written = json.loads((offline_host.home / "data" / "kernels" / names(specs)[0] / "kernel.json").read_text())
    assert written["argv"] == [str(home / "proj" / ".venv" / "bin" / "python"), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    assert written["language"] == "python"


def test_a_venv_without_ipykernel_is_listed_as_needing_it(offline_host, home):
    fake_venv(home / "bare", ipykernel=False)
    fake_venv(home / "ready")

    bare = offline_host.kernelspecs(str(notebook(home / "bare" / "a.ipynb")))
    ready = offline_host.kernelspecs(str(notebook(home / "ready" / "a.ipynb")))

    assert [spec.needs_ipykernel for spec in bare.specs] == [True, False]
    assert [spec.needs_ipykernel for spec in ready.specs] == [False, False]


def test_the_venv_specs_of_other_notebooks_are_not_offered(offline_host, home):
    fake_venv(home / "proj")
    offline_host.kernelspecs(str(notebook(home / "proj" / "a.ipynb")))

    other = offline_host.kernelspecs(str(notebook(home / "elsewhere" / "b.ipynb")))

    assert names(other) == ["python3"]
    assert other.default == "python3"


def test_the_default_is_the_listed_kernelspec_of_the_notebook_then_the_nearest_venv_then_the_app_env(offline_host, home):
    fake_venv(home / "proj")

    named = offline_host.kernelspecs(str(notebook(home / "proj" / "a.ipynb", "python3")))
    unknown = offline_host.kernelspecs(str(notebook(home / "proj" / "b.ipynb", "gone")))
    unreadable = offline_host.kernelspecs(str(home / "proj" / "missing.ipynb"))
    plain = offline_host.kernelspecs(str(notebook(home / "plain" / "c.ipynb")))

    assert named.default == "python3"
    assert unknown.default == unreadable.default == names(unknown)[0] != "python3"
    assert plain.default == "python3"


@pytest.mark.parametrize("path", ["relative/a.ipynb", "/tmp/a.py"])
def test_only_absolute_notebook_paths_are_accepted(offline_host, path):
    with pytest.raises(UpstreamRequestFailed, match="Not an absolute notebook path"):
        offline_host.kernelspecs(path)


def test_the_folder_of_the_notebook_has_to_exist(offline_host, home):
    with pytest.raises(UpstreamRequestFailed, match="Folder not found"):
        offline_host.kernelspecs(str(home / "nowhere" / "a.ipynb"))


def test_an_unexpected_kernelspec_answer_is_a_bad_response(tmp_path, home, monkeypatch):
    host = JupyterHost(tmp_path / "jupyter")
    answer = httpx.Response(200, json={"unexpected": 1}, request=httpx.Request("GET", "http://x/api/kernelspecs"))
    monkeypatch.setattr(host, "_call", lambda *_args, **_kwargs: answer)

    with pytest.raises(UpstreamContractError):
        host.kernelspecs(str(notebook(home / "a.ipynb")))

    response = client_for(host).get("/api/jupyter/kernelspecs", params={"path": str(home / "a.ipynb")})
    assert response.status_code == 502


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("api/kernels", True),
        ("api/kernels/0e1d/channels", True),
        ("api/kernels/0e1d/restart", True),
        ("api/kernelspecs", True),
        ("api/sessions/abc", True),
        ("api/events/subscribe", True),
        ("api/status", True),
        ("api/me", True),
        ("kernelspecs/python3/logo-svg.svg", True),
        ("api/contents", False),
        ("api/contents/etc/passwd", False),
        ("api/terminals", False),
        ("api/kernelspecsx", False),
        ("api/kernels/../contents", False),
        ("api/kernels/%2e%2e/contents", False),
        ("api/kernels//x", False),
        ("api/kernels/./x", False),
        ("api/kernels\\..\\contents", False),
        ("api", False),
        ("", False),
        ("login", False),
        ("files/etc/passwd", False),
        ("kernelspecs/python3", False),
        ("nbconvert/html", False),
    ],
)
def test_proxy_allowlist(path, allowed):
    assert jupyter.is_proxied_path(path) is allowed


def test_listing_kernelspecs_never_runs_the_python_of_a_venv(offline_host, home, tmp_path):
    ran = tmp_path / "ran"
    executable(home / "proj" / ".venv" / "bin" / "python", f'touch "{ran}"\nexit 1')
    nb = notebook(home / "proj" / "a.ipynb")

    bare = offline_host.kernelspecs(str(nb))
    install_ipykernel(home / "proj" / ".venv")
    installed = offline_host.kernelspecs(str(nb))

    assert not ran.exists()
    assert bare.specs[0].needs_ipykernel is True and installed.specs[0].needs_ipykernel is False


def test_browser_credentials_never_reach_the_server_and_its_cookies_never_reach_the_browser():
    request = [
        ("Authorization", "token browser"), ("Cookie", "_xsrf=1"), ("Origin", "https://x"), ("Referer", "https://x/"),
        ("X-XSRFToken", "1"), ("Host", "app"), ("Content-Type", "application/json"), ("Tailscale-User-Login", "me"),
    ]  # fmt: skip
    response = [("Set-Cookie", "a=b"), ("Content-Type", "application/json"), ("Access-Control-Allow-Origin", "*"), ("Etag", '"1"')]

    assert forwardable_headers(request, jupyter._DROPPED_REQUEST_HEADERS) == [("Content-Type", "application/json")]
    assert [name for name, _ in response if not jupyter.drops_response_header(name.lower())] == ["Content-Type", "Etag"]


@dataclass
class World:
    host: JupyterHost
    home: Path
    project: Path
    client: object


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One real Jupyter Server for the module, `other` as a second discovered kernelspec, `proj/.venv` as a venv."""
    base = tmp_path_factory.mktemp("jupyter").resolve()
    home = base / "home"
    project = home / "proj"
    project.mkdir(parents=True)
    fake_venv(project)
    extra = base / "extra" / "kernels" / "other"
    extra.mkdir(parents=True)
    (extra / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Other",
                "language": "python",
            }
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(home))
        patch.setenv("GIGACHAD_JUPYTER_HOME", str(base / "jupyter"))
        patch.setenv("JUPYTER_PATH", str(base / "extra"))
        patch.setattr(websocket_origin, "CORS_ORIGINS", ["*"])
        host = JupyterHost()
        patch.setattr(jupyter, "_host", host)
        with client_for(host) as client:
            yield World(host, home, project, client)
        host.stop()


def start(world, path, name):
    response = world.client.post("/api/jupyter/kernel", json={"path": str(path), "name": name})
    assert response.status_code == 200, response.text
    return response.json()


def execute(client, kernel_id, code):
    """Run *code* in the kernel over the WebSocket proxy and return what it printed or evaluated to."""
    message_id = uuid4().hex
    request = {
        "header": {"msg_id": message_id, "username": "t", "session": "s", "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
        "content": {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False},
    }  # fmt: skip
    url = f"{SERVER}/api/kernels/{kernel_id}/channels?session_id={uuid4().hex}&token=browser-token"
    output, replied, idle = [], False, False
    with client.websocket_connect(url, headers={"origin": ORIGIN}) as socket:
        socket.send_text(json.dumps(request))
        while not (replied and idle):
            message = json.loads(socket.receive_text())
            if message["parent_header"].get("msg_id") != message_id:
                continue
            match message["msg_type"]:
                case "stream":
                    output.append(message["content"]["text"])
                case "execute_result":
                    output.append(message["content"]["data"]["text/plain"])
                case "execute_reply":
                    replied = True
                case "status":
                    idle = message["content"]["execution_state"] == "idle"
    return "".join(output)


def test_the_kernelspecs_of_a_notebook_list_what_jupyter_finds_and_the_venv(world):
    nb = notebook(world.project / "sub" / "a.ipynb")

    response = world.client.get("/api/jupyter/kernelspecs", params={"path": str(nb)})

    assert response.status_code == 200
    body = response.json()
    by_name = {spec["name"]: spec for spec in body["specs"]}
    assert {"python3", "other"} <= set(by_name)
    assert body["default"].startswith("venv-")
    assert by_name[body["default"]] == {
        "name": body["default"], "display_name": "proj/.venv", "language": "python", "needs_ipykernel": False,
    }  # fmt: skip
    assert by_name["other"]["display_name"] == "Other"


def test_the_kernel_runs_in_the_folder_of_the_notebook_and_is_reused_for_the_same_kernelspec(world):
    nb = notebook(world.project / "sub" / "cwd.ipynb")

    first = start(world, nb, "python3")
    second = start(world, nb, "python3")

    assert first == second and first["name"] == "python3"
    assert execute(world.client, first["kernel_id"], "import os; print(os.getcwd())").strip() == str(nb.parent)
    assert execute(world.client, first["kernel_id"], "1 + 1") == "2"


def test_a_venv_spec_starts_the_kernel_of_that_venv(world):
    nb = notebook(world.project / "venv.ipynb")
    default = world.client.get("/api/jupyter/kernelspecs", params={"path": str(nb)}).json()["default"]

    kernel = start(world, nb, default)

    assert kernel["name"] == default
    assert execute(world.client, kernel["kernel_id"], "import os; print(os.environ['VIRTUAL_ENV'])").strip() == str(
        world.project / ".venv"
    )


def test_another_kernelspec_replaces_the_kernel_of_the_notebook(world):
    nb = notebook(world.project / "switch.ipynb")
    first = start(world, nb, "python3")

    second = start(world, nb, "other")

    assert second["name"] == "other" and second["kernel_id"] != first["kernel_id"]
    assert world.client.get(f"{SERVER}/api/kernels/{first['kernel_id']}").status_code == 404
    assert world.client.get(f"{SERVER}/api/kernels/{second['kernel_id']}").status_code == 200


def test_a_kernel_the_server_lost_is_started_again(world):
    nb = notebook(world.project / "lost.ipynb")
    first = start(world, nb, "python3")
    assert world.client.delete(f"{SERVER}/api/kernels/{first['kernel_id']}").status_code == 204

    second = start(world, nb, "python3")

    assert second["kernel_id"] != first["kernel_id"]


def test_each_notebook_has_its_own_kernel(world):
    first = start(world, notebook(world.project / "one.ipynb"), "python3")
    second = start(world, notebook(world.project / "two.ipynb"), "python3")

    assert first["kernel_id"] != second["kernel_id"]


def test_unknown_kernelspecs_and_notebook_paths_are_not_found(world):
    nb = notebook(world.project / "unknown.ipynb")

    assert world.client.post("/api/jupyter/kernel", json={"path": str(nb), "name": "nope"}).status_code == 404
    assert world.client.post("/api/jupyter/kernel", json={"path": "relative.ipynb", "name": "python3"}).status_code == 404
    assert world.client.get("/api/jupyter/kernelspecs", params={"path": str(world.home / "nowhere" / "a.ipynb")}).status_code == 404


def test_notebooks_outside_the_file_roots_are_not_reachable(world):
    outside = notebook(world.home.parent / "outside" / "a.ipynb")

    assert world.client.get("/api/jupyter/kernelspecs", params={"path": str(outside)}).status_code == 404
    assert world.client.post("/api/jupyter/kernel", json={"path": str(outside), "name": "python3"}).status_code == 404
    assert world.client.delete("/api/jupyter/kernel", params={"path": str(outside)}).status_code == 404


def test_shutting_a_kernel_down_is_idempotent(world):
    nb = notebook(world.project / "shutdown.ipynb")
    kernel = start(world, nb, "python3")

    first = world.client.delete("/api/jupyter/kernel", params={"path": str(nb)})
    second = world.client.delete("/api/jupyter/kernel", params={"path": str(nb)})

    assert (first.status_code, second.status_code) == (204, 204)
    assert world.client.get(f"{SERVER}/api/kernels/{kernel['kernel_id']}").status_code == 404


def test_a_moved_notebook_keeps_its_kernel(world):
    nb = notebook(world.project / "folder" / "a.ipynb")
    kernel = start(world, nb, "python3")

    jupyter.move_kernel(str(nb), str(nb.with_name("renamed.ipynb")))

    assert start(world, nb.with_name("renamed.ipynb"), "python3") == kernel
    assert world.client.get(f"{SERVER}/api/kernels/{kernel['kernel_id']}").status_code == 200


def test_deleting_a_folder_shuts_down_the_kernels_of_its_notebooks_only(world):
    doomed = notebook(world.project / "doomed" / "a.ipynb")
    sibling = notebook(world.project / "doomed-not" / "a.ipynb")
    gone = start(world, doomed, "python3")
    kept = start(world, sibling, "python3")

    jupyter.shutdown_kernels_under(str(doomed.parent))

    assert world.client.get(f"{SERVER}/api/kernels/{gone['kernel_id']}").status_code == 404
    assert world.client.get(f"{SERVER}/api/kernels/{kept['kernel_id']}").status_code == 200


def test_a_venv_without_ipykernel_gets_it_installed_before_its_kernel_starts(world, tmp_path, monkeypatch):
    folder = world.home / "bare"
    python = fake_venv(folder, ipykernel=False)
    nb = notebook(folder / "a.ipynb")
    log = tmp_path / "uv.log"
    install = 'd="$(dirname "$4")/../lib/python3.13/site-packages/ipykernel"\nmkdir -p "$d"\ntouch "$d/__init__.py"'
    uv = executable(tmp_path / "bin" / "uv", f'echo "$@" >> {log}\n{install}')
    monkeypatch.setattr(jupyter.shutil, "which", lambda name: str(uv) if name == "uv" else None)
    before = world.client.get("/api/jupyter/kernelspecs", params={"path": str(nb)}).json()
    venv = before["specs"][0]
    assert venv["needs_ipykernel"] is True

    kernel = start(world, nb, venv["name"])

    assert log.read_text().split() == ["pip", "install", "--python", str(python), "ipykernel"]
    assert execute(world.client, kernel["kernel_id"], "6 * 7") == "42"
    after = world.client.get("/api/jupyter/kernelspecs", params={"path": str(nb)}).json()
    assert after["specs"][0]["needs_ipykernel"] is False


def test_a_failed_ipykernel_install_answers_bad_gateway_with_the_reason(world, tmp_path, monkeypatch):
    folder = world.home / "broken"
    fake_venv(folder, ipykernel=False)
    nb = notebook(folder / "a.ipynb")
    uv = executable(tmp_path / "bin" / "uv", "echo 'no index reachable' >&2\nexit 2")
    monkeypatch.setattr(jupyter.shutil, "which", lambda name: str(uv) if name == "uv" else None)
    name = world.client.get("/api/jupyter/kernelspecs", params={"path": str(nb)}).json()["specs"][0]["name"]

    response = world.client.post("/api/jupyter/kernel", json={"path": str(nb), "name": name})

    assert response.status_code == 502
    assert "no index reachable" in response.json()["detail"]


def test_the_proxy_reaches_the_server_with_its_token_whatever_the_browser_sends(world):
    response = world.client.get(
        f"{SERVER}/api/status?token=bad&_xsrf=bad",
        headers={"Authorization": "token bad", "Cookie": "_xsrf=bad", "X-XSRFToken": "bad"},
    )

    assert response.status_code == 200
    assert "set-cookie" not in response.headers
    assert "kernels" in response.json()


def test_kernels_never_see_the_servers_token_and_it_is_in_no_file_or_command_line(world):
    kernel = start(world, notebook(world.project / "env.ipynb"), "python3")
    token = world.host._server.token
    code = (
        "import glob, os\n"
        f"print([name for name, value in os.environ.items() if {token!r} in value], 'JUPYTER_TOKEN' in os.environ, "
        "glob.glob(os.environ['JUPYTER_RUNTIME_DIR'] + '/jpserver*'))"
    )

    assert execute(world.client, kernel["kernel_id"], code).strip() == "[] False []"
    pid = world.host._server.process.pid
    for name in ("cmdline", "environ"):
        assert token.encode() not in Path(f"/proc/{pid}/{name}").read_bytes(), name
    assert stat.S_IMODE(world.host.home.stat().st_mode) == 0o700
    assert stat.S_IMODE((world.host.home / "server.log").stat().st_mode) == 0o600


def test_idle_kernels_nobody_is_connected_to_are_culled_and_start_again_for_their_notebook(tmp_path, home, monkeypatch):
    monkeypatch.setattr(jupyter, "_KERNEL_CULL_IDLE_SECONDS", 1)
    monkeypatch.setattr(jupyter, "_KERNEL_CULL_INTERVAL_SECONDS", 1)
    host = JupyterHost(tmp_path / "jupyter")
    nb = str(notebook(home / "a.ipynb"))
    try:
        kernel = host.start_kernel(nb, "python3")
        deadline = time.monotonic() + 30
        while host._is_running(kernel) and time.monotonic() < deadline:
            time.sleep(0.5)

        assert not host._is_running(kernel)
        assert host.start_kernel(nb, "python3").kernel_id != kernel.kernel_id
    finally:
        host.stop()


def test_a_notebook_has_one_kernel_whichever_way_its_path_is_spelled(world):
    real = world.project / "real"
    (real / "sub").mkdir(parents=True)
    link = world.project / "link"
    link.symlink_to(real, target_is_directory=True)
    nb = notebook(real / "a.ipynb")

    first = start(world, nb, "python3")
    second = start(world, link / "a.ipynb", "python3")
    third = start(world, real / "sub" / ".." / "a.ipynb", "python3")

    assert first == second == third
    assert [key for key in world.host._kernels if key.endswith("/a.ipynb") and "/real/" in key] == [str(nb)]
    assert execute(world.client, first["kernel_id"], "import os; print(os.getcwd())").strip() == str(real)

    jupyter.move_kernel(str(link / "a.ipynb"), str(link / "b.ipynb"))
    assert start(world, real / "b.ipynb", "python3") == first
    assert world.client.delete("/api/jupyter/kernel", params={"path": str(link / "b.ipynb")}).status_code == 204
    assert world.client.get(f"{SERVER}/api/kernels/{first['kernel_id']}").status_code == 404

    again = start(world, real / "b.ipynb", "python3")
    jupyter.shutdown_kernels_under(str(link))
    assert world.client.get(f"{SERVER}/api/kernels/{again['kernel_id']}").status_code == 404


def test_the_proxy_forwards_request_bodies_and_methods(world):
    nb = notebook(world.project / "restart.ipynb")
    kernel = start(world, nb, "python3")

    restarted = world.client.post(f"{SERVER}/api/kernels/{kernel['kernel_id']}/restart")
    interrupted = world.client.post(f"{SERVER}/api/kernels/{kernel['kernel_id']}/interrupt")
    listed = world.client.get(f"{SERVER}/api/kernels")

    assert restarted.status_code == 200 and restarted.json()["id"] == kernel["kernel_id"]
    assert interrupted.status_code == 204
    assert kernel["kernel_id"] in [item["id"] for item in listed.json()]


def test_the_proxy_serves_kernel_logos_with_the_servers_csp_and_refuses_everything_else(world):
    logo = world.client.get(f"{SERVER}/kernelspecs/python3/logo-svg.svg")

    assert logo.status_code == 200 and b"<svg" in logo.content
    assert "content-security-policy" in logo.headers
    for path in ("api/contents", "api/terminals", "api/kernels/../contents", "api/kernels/%2e%2e/contents", "login", "files/x"):
        assert world.client.get(f"{SERVER}/{path}").status_code == 404, path


def test_requests_another_site_sent_neither_change_kernels_nor_list_specs(world):
    nb = notebook(world.project / "forged.ipynb")
    kernel = start(world, nb, "python3")
    kernels = set(world.host._kernels)
    kernel_url = f"{SERVER}/api/kernels/{kernel['kernel_id']}"
    session = {"path": ".datalayer/forged.ipynb", "type": "notebook", "kernel": {"id": kernel["kernel_id"]}}

    target = {"path": str(nb.with_name("other.ipynb")), "name": "python3"}
    for origin in ("https://evil.example", "null", None):
        with sent_from(world.client, origin):
            client = world.client
            assert client.post("/api/jupyter/kernel", json=target).status_code == 403
            as_text = client.post("/api/jupyter/kernel", content=json.dumps(target), headers={"content-type": "text/plain"})
            assert as_text.status_code == 403
            assert client.delete("/api/jupyter/kernel", params={"path": str(nb)}).status_code == 403
            assert client.post(f"{kernel_url}/restart").status_code == 403
            assert client.post(f"{kernel_url}/interrupt").status_code == 403
            assert client.post(f"{SERVER}/api/sessions", json=session).status_code == 403
            assert client.delete(kernel_url).status_code == 403

    assert world.client.get(kernel_url).status_code == 200
    assert set(world.host._kernels) == kernels
    assert world.client.get(f"{SERVER}/api/sessions").json() == []


def test_reads_another_site_sent_are_refused_and_plain_ones_are_not(world):
    nb = notebook(world.project / "reads.ipynb")
    listing = {"path": str(nb)}

    with sent_from(world.client, "https://evil.example"):
        assert world.client.get("/api/jupyter/kernelspecs", params=listing).status_code == 403
        assert world.client.get(f"{SERVER}/api/status").status_code == 403
    with sent_from(world.client, None):
        for site, status in (("cross-site", 403), ("same-site", 403), ("same-origin", 200), ("none", 200), (None, 200)):
            headers = {"sec-fetch-site": site} if site else {}
            assert world.client.get("/api/jupyter/kernelspecs", params=listing, headers=headers).status_code == status, site
            assert world.client.get(f"{SERVER}/api/status", headers=headers).status_code == status, site


def test_browsers_cannot_start_kernels_through_the_proxy(world):
    known = {item["id"] for item in world.client.get(f"{SERVER}/api/kernels").json()}
    session = {"path": ".datalayer/a.ipynb", "type": "notebook", "name": "a.ipynb"}

    created = world.client.post(f"{SERVER}/api/kernels", json={"name": "python3", "path": "tmp"})
    by_name = world.client.post(f"{SERVER}/api/sessions", json={**session, "kernel": {"name": "python3"}})
    without = world.client.post(f"{SERVER}/api/sessions", json=session)
    by_default = world.client.post(f"{SERVER}/api/sessions", json={**session, "kernel": {}})
    stray = world.host._call("POST", "api/kernels", json={"name": "python3"}).json()["id"]
    foreign = world.client.post(f"{SERVER}/api/sessions", json={**session, "kernel": {"id": stray}})
    unknown = world.client.post(f"{SERVER}/api/sessions", json={**session, "kernel": {"id": "no-such-kernel"}})
    legacy = world.client.post(f"{SERVER}/api/sessions", json={"notebook": {"path": "a.ipynb"}, "kernel": {"name": "python3"}})
    world.host._call("DELETE", f"api/kernels/{stray}")

    assert created.status_code == 405
    statuses = [response.status_code for response in (by_name, without, by_default, foreign, unknown, legacy)]
    assert statuses == [403, 403, 403, 403, 403, 400]
    assert {item["id"] for item in world.client.get(f"{SERVER}/api/kernels").json()} == known
    assert world.client.get(f"{SERVER}/api/sessions").json() == []


def test_a_session_binds_the_kernels_of_notebooks_and_nothing_else(world):
    first = start(world, notebook(world.project / "session-one.ipynb"), "python3")
    second = start(world, notebook(world.project / "session-two.ipynb"), "python3")
    body = {"path": ".datalayer/tab.ipynb", "type": "notebook", "name": "tab.ipynb", "kernel": {"id": first["kernel_id"]}}

    created = world.client.post(f"{SERVER}/api/sessions", json=body)
    session_url = f"{SERVER}/api/sessions/{created.json()['id']}"
    # The library names the session by id in the body as well as in the URL.
    renamed = world.client.patch(session_url, json={"path": ".datalayer/renamed.ipynb", "id": created.json()["id"]})
    by_name = world.client.patch(session_url, json={"kernel": {"name": "python3"}})
    absolute = world.client.patch(session_url, json={"path": "/etc/passwd"})
    traversal = world.client.post(f"{SERVER}/api/sessions", json={**body, "path": "../a.ipynb"})
    switched = world.client.patch(session_url, json={"kernel": {"id": second["kernel_id"]}})
    ended = world.client.delete(session_url)

    assert created.status_code == 201 and created.json()["kernel"]["id"] == first["kernel_id"]
    assert (renamed.status_code, by_name.status_code, absolute.status_code, traversal.status_code) == (200, 403, 400, 400)
    assert switched.status_code == 200 and switched.json()["kernel"]["id"] == second["kernel_id"]
    assert ended.status_code == 204


@pytest.mark.parametrize(
    ("method", "path", "allowed"),
    [
        ("GET", "api/kernels", True),
        ("HEAD", "api/sessions", True),
        ("POST", "api/kernels", False),
        ("PUT", "api/kernels/0e1d", False),
        ("PATCH", "api/kernels/0e1d", False),
        ("DELETE", "api/kernels/0e1d", True),
        ("POST", "api/kernels/0e1d/restart", True),
        ("POST", "api/kernels/0e1d/interrupt", True),
        ("POST", "api/kernels/0e1d/channels", False),
        ("POST", "api/sessions", True),
        ("POST", "api/sessions/abc", False),
        ("PATCH", "api/sessions/abc", True),
        ("DELETE", "api/sessions/abc", True),
        ("POST", "api/events", False),
        ("POST", "api/kernelspecs", False),
        ("DELETE", "api/kernelspecs", False),
    ],
)
def test_the_browser_changes_only_what_a_notebook_editor_needs(method, path, allowed):
    assert jupyter.allows_method(method, path) is allowed


def test_the_websocket_proxy_negotiates_the_kernel_protocol_the_browser_asks_for(world):
    kernel = start(world, notebook(world.project / "protocol.ipynb"), "python3")
    url = f"{SERVER}/api/kernels/{kernel['kernel_id']}/channels?session_id={uuid4().hex}"

    with world.client.websocket_connect(url, headers={"origin": ORIGIN}, subprotocols=["v1.kernel.websocket.jupyter.org"]) as socket:
        assert socket.accepted_subprotocol == "v1.kernel.websocket.jupyter.org"


def test_the_websocket_proxy_refuses_other_paths_untrusted_origins_and_logins_without_host_access(world):
    kernel = start(world, notebook(world.project / "refused.ipynb"), "python3")
    channels = f"{SERVER}/api/kernels/{kernel['kernel_id']}/channels"
    denied = client_for(world.host, host_access=False)

    for client, url, origin in (
        (world.client, f"{SERVER}/api/terminals/websocket/1", ORIGIN),
        (world.client, channels, "https://evil.example"),
        (world.client, channels, None),
        (denied, channels, ORIGIN),
    ):
        with sent_from(client, origin), pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(url):
            pass
        assert refused.value.code == 1008, (url, origin)


def test_logins_without_host_access_get_nothing(world):
    denied = client_for(world.host, host_access=False)
    notebook_path = str(world.project / "a.ipynb")

    assert denied.get(f"{SERVER}/api/status").status_code == 403
    assert denied.get("/api/jupyter/kernelspecs", params={"path": notebook_path}).status_code == 403
    assert denied.post("/api/jupyter/kernel", json={"path": notebook_path, "name": "python3"}).status_code == 403
    assert denied.delete("/api/jupyter/kernel", params={"path": notebook_path}).status_code == 403


def test_refused_requests_do_not_start_a_server(tmp_path):
    host = JupyterHost(tmp_path / "jupyter")

    assert client_for(host).get(f"{SERVER}/api/contents").status_code == 404
    assert host._server is None


def test_a_server_that_cannot_start_is_unavailable(tmp_path, home, monkeypatch):
    monkeypatch.setattr(jupyter, "_BOOTSTRAP", "raise SystemExit(3)")
    client = client_for(JupyterHost(tmp_path / "jupyter"))

    assert client.get(f"{SERVER}/api/status").status_code == 503
    assert client.get("/api/jupyter/kernelspecs", params={"path": str(notebook(home / "a.ipynb"))}).status_code == 503


def test_a_server_that_died_is_started_again_without_the_kernels_it_took_along(world):
    nb = notebook(world.project / "died.ipynb")
    first = start(world, nb, "python3")
    world.host._server.process.kill()
    world.host._server.process.wait()

    second = start(world, nb, "python3")

    assert second["kernel_id"] != first["kernel_id"]
    assert execute(world.client, second["kernel_id"], "1 + 2") == "3"


def test_stopping_the_host_ends_the_server_and_its_files(tmp_path, home):
    host = JupyterHost(tmp_path / "jupyter")
    host.start_kernel(str(notebook(home / "a.ipynb")), "python3")
    process = host._server.process
    scratch = Path(host._server.scratch.name)
    assert scratch.is_dir()

    host.stop()

    assert process.poll() is not None
    assert host._server is None
    assert not scratch.exists()


@pytest.fixture
def documents(world):
    """The documents routes over memory stores and a mount `world.home / "mount"`."""
    mount = world.home / "mount"
    mount.mkdir(exist_ok=True)
    roots = mounted_roots(mount)
    docs = MemoryDataStore()
    assets = MemoryAssetStore()
    store = ProjectStore(ChatStore(data_store=docs), rename=lambda *_: None)
    overrides = {
        get_file_roots: roots,
        get_project_store: store,
        get_document_store: docs,
        get_asset_store: assets,
        get_relocate: lambda old, new: None,
        get_references: References(docs, assets),
        get_disk_files: DiskFiles(roots, RecordingScope()),
    }
    return SimpleNamespace(client=route_client(documents_route.router, overrides=overrides), mount=mount, store=store)


def test_renaming_a_notebook_moves_its_kernel_with_it(world, documents):
    nb = notebook(documents.mount / "analysis" / "a.ipynb")
    kernel = start(world, nb, "python3")

    renamed = documents.client.post("/api/documents/rename", json={"path": str(nb), "slug": "", "name": "b.ipynb"})

    assert renamed.status_code == 200
    assert start(world, documents.mount / "analysis" / "b.ipynb", "python3") == kernel


def test_deleting_a_document_folder_shuts_down_the_kernels_of_its_notebooks(world, postgres_pool, clean_database, monkeypatch):
    monkeypatch.setattr(config, "WORKBENCH", world.home / "workbench")
    user_id, device_id = create_user_and_device(postgres_pool)
    scope = UserScope(postgres_pool, user_id, device_id)
    docs, assets = PostgresDataStore(scope), AssetStore(scope)
    projects = ProjectStore(ChatStore(data_store=docs), rename=PostgresProjectRenamer(scope).rename)
    slug = projects.create_project("Thesis")["slug"]
    nb = notebook(config.WORKBENCH / "latex" / slug / "paper" / "a.ipynb")
    projects.add_file(slug, str(nb))
    kernel = start(world, nb, "python3")

    removed = documents_route.remove_document(slug, str(nb), projects, References(docs, assets), DiskFiles(mounted_roots(), scope))[
        "removed"
    ]

    assert removed == [str(nb.parent)]
    assert world.client.get(f"{SERVER}/api/kernels/{kernel['kernel_id']}").status_code == 404
