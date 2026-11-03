"""Jupyter notebooks on the host: a private Jupyter Server, kernelspecs, one kernel per notebook, and the proxy to reach it.

The backend is the only client of the server, so everything that knows Jupyter lives here.

- **Server**: `python -m jupyter_server` from the app venv on a free loopback port, started on first use, stopped with the backend
  (also when the backend is killed, kernels with it, see `_BOOTSTRAP`, and marked for `lib.process_owner.reap_orphans`) and
  started again after it died. The token exists only in the backend and the server's memory (never in its environment, its
  argv or a file its kernels could read), requests the browser sends through the proxy get it injected. Its kernelspecs and log
  live in `jupyter_home()` (private to the user), its runtime files in a temporary folder, and config files are ignored. Kernels
  nobody is connected to are culled after `_KERNEL_CULL_IDLE_SECONDS`, so a reload reattaches by path and a closed browser frees them.
- **Kernelspecs**: what Jupyter discovers, plus one generated spec per `.venv` in the notebook's folder or an ancestor up to `$HOME`
  (`venv-<hash of the venv>`, written below `jupyter_home()`). A venv without `ipykernel` (looked for as a file, listing never runs
  a venv's python) is listed and gets it on first use.
- **Kernels**: one per notebook path, keyed by the resolved path (`notebook_path`), its working directory is the notebook's folder.
  The map from path to kernel follows renames (`move_kernel`) and ends with deletes (`shutdown_kernels_under`), the documents
  routes call both. The browser starts kernels only here, never through the proxy.
- **Proxy**: the browser reaches the server through `/api/jupyter/server/`, restricted to `is_proxied_path` and `allows_method`
  (sessions only bind a kernel of this host, `check_session`) and spoken by `open_http` and `open_websocket`, so it never sees the
  token, cookies or the server's port.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel

from lib.process_owner import owned_environment
from lib.upstream import (
    UpstreamContractError,
    UpstreamRequestFailed,
    UpstreamUnavailable,
    forwardable_headers,
    unreachable,
    validate_answer,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from websockets.asyncio.client import ClientConnection

log = logging.getLogger(__name__)

BASE_URL = "/api/jupyter/server/"
APP_KERNELSPEC = "python3"
_VENV_PREFIX = "venv-"
_SERVER_START_SECONDS = 40
_SERVER_STOP_SECONDS = 10
_REQUEST_SECONDS = 10
_IPYKERNEL_INSTALL_SECONDS = 180
_LOG_TAIL_BYTES = 2000
# A kernel without a connected browser tab and without activity this long is shut down, checked every interval.
_KERNEL_CULL_IDLE_SECONDS = 2 * 60 * 60
_KERNEL_CULL_INTERVAL_SECONDS = 60

# Runs in the server process, which ends with the backend even when that dies without stopping it (`kill -9`, reload): a thread
# watches the parent, stops the server like SIGTERM does (kernels end) and exits once the grace period is over. Kernels get SIGKILL
# whenever the server ends (Linux parent-death signal): ipykernel's own parent check misses a server that died while it started.
# The token arrives on stdin: not in argv or the environment (every kernel inherits that, and `/proc/<pid>/environ` stays readable).
_BOOTSTRAP = """
import os, signal, sys, threading, time
parent, grace = int(sys.argv[1]), float(sys.argv[2])
server = os.getpid()
token = sys.stdin.readline().strip()
def stop_with_parent():
    while os.getppid() == parent:
        time.sleep(2)
    os.kill(server, signal.SIGTERM)
    time.sleep(grace)
    os._exit(1)
threading.Thread(target=stop_with_parent, daemon=True).start()
if sys.platform == "linux":
    import ctypes
    from jupyter_client.provisioning import local_provisioner
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    def die_with_server():
        prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG, kept across exec
        if os.getppid() != server:
            os._exit(1)
    launch = local_provisioner.launch_kernel
    local_provisioner.launch_kernel = lambda cmd, **kwargs: launch(cmd, preexec_fn=die_with_server, **kwargs)
from jupyter_server.serverapp import ServerApp
from traitlets.config import Config
config = Config()
config.IdentityProvider.token = token
ServerApp.launch_instance(sys.argv[3:], config=config)
"""

# What the browser may reach upstream: kernels, kernelspecs and their logos, sessions, events and the status calls of the client.
_PROXIED_PATH = re.compile(
    r"api/(?:kernels|kernelspecs|sessions|events)(?:/[^/]+)*|api/(?:status|me)|kernelspecs/[^/]+/[^/]+",
)
_READ_METHODS = frozenset({"GET", "HEAD"})
# What the browser may change: a kernel's signals and end, and the sessions that bind a notebook's kernel (`check_session`).
# Starting a kernel is `POST /api/jupyter/kernel`, never `POST api/kernels`.
_WRITE_PATHS = (
    (re.compile(r"api/kernels/[^/]+"), frozenset({"DELETE"})),
    (re.compile(r"api/kernels/[^/]+/(?:restart|interrupt)"), frozenset({"POST"})),
    (re.compile(r"api/sessions"), frozenset({"POST"})),
    (re.compile(r"api/sessions/[^/]+"), frozenset({"PATCH", "DELETE"})),
)
_SESSION_KEYS = frozenset({"id", "path", "name", "type", "kernel"})
_SESSION_KERNEL_KEYS = frozenset({"id", "name"})
# Credentials of the browser's own session: the proxy replaces them with the server's token.
_DROPPED_QUERY = frozenset({"token", "_xsrf"})
_DROPPED_REQUEST_HEADERS = frozenset({
    "host", "origin", "referer", "cookie", "authorization", "x-xsrftoken", "content-length", "accept-encoding",
    "x-device-id", "tailscale-user-login", "tailscale-app-capabilities",
})  # fmt: skip


class KernelSpecInfo(BaseModel):
    name: str
    display_name: str
    language: str
    needs_ipykernel: bool


class KernelSpecList(BaseModel):
    default: str
    specs: list[KernelSpecInfo]


class KernelRef(BaseModel):
    kernel_id: str
    name: str


class _UpstreamSpecBody(BaseModel):
    display_name: str
    language: str = ""


class _UpstreamSpec(BaseModel):
    name: str
    spec: _UpstreamSpecBody


class _UpstreamSpecs(BaseModel):
    default: str = ""
    kernelspecs: dict[str, _UpstreamSpec]


class _UpstreamKernel(BaseModel):
    id: str
    name: str


@dataclass(frozen=True)
class _Venv:
    directory: Path
    spec: KernelSpecInfo

    @property
    def python(self) -> Path:
        return self.directory / "bin" / "python"


@dataclass(frozen=True)
class _Server:
    process: subprocess.Popen
    port: int
    token: str
    client: httpx.Client
    scratch: tempfile.TemporaryDirectory

    @property
    def http_base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}


def jupyter_home() -> Path:
    """Where the server keeps its files, `GIGACHAD_JUPYTER_HOME` or below `~/.local/share/gigachad-bot`."""
    override = os.environ.get("GIGACHAD_JUPYTER_HOME")
    return Path(override) if override else Path.home() / ".local" / "share" / "gigachad-bot" / "jupyter"


def notebook_path(path: str) -> Path:
    """*path* as the key of a notebook: absolute with symlinks resolved, a 404 `UpstreamRequestFailed` for anything else."""
    if not os.path.isabs(path):
        raise UpstreamRequestFailed(404, f"Not an absolute notebook path: {path}")
    resolved = Path(path).resolve()
    if resolved.suffix != ".ipynb":
        raise UpstreamRequestFailed(404, f"Not an absolute notebook path: {path}")
    return resolved


def is_proxied_path(path: str) -> bool:
    """Whether the upstream *path* (below the server's base URL, decoded) is one the browser may reach."""
    if any(character in path for character in "%\\\x00") or any(segment in {"", ".", ".."} for segment in path.split("/")):
        return False
    return _PROXIED_PATH.fullmatch(path) is not None


def _upstream_target(path: str, query: Iterable[tuple[str, str]]) -> str:
    kept = [(name, value) for name, value in query if name not in _DROPPED_QUERY]
    return f"{BASE_URL}{quote(path, safe='/')}" + (f"?{urlencode(kept)}" if kept else "")


def allows_method(method: str, path: str) -> bool:
    """Whether the browser may send *method* to the proxied upstream *path*: reads anywhere, changes only where listed."""
    return method in _READ_METHODS or any(method in methods and pattern.fullmatch(path) for pattern, methods in _WRITE_PATHS)


def drops_response_header(name: str) -> bool:
    """Whether the server's response header *name* (lowercase) stays with the proxy: cookies and CORS fields, not its CSP."""
    return name == "set-cookie" or name.startswith("access-control-")


def _shorten(directory: Path) -> str:
    return hashlib.sha256(str(directory).encode()).hexdigest()[:8]


def _find_venvs(notebook: Path) -> list[_Venv]:
    """The `.venv` of the notebook's folder and of each ancestor up to `$HOME`, nearest first."""
    home = Path.home().resolve()
    folders = [notebook.parent]
    if folders[0].is_relative_to(home):
        folders += [parent for parent in folders[0].parents if parent.is_relative_to(home)]
    venvs = []
    for folder in folders:
        directory = folder / ".venv"
        if (directory / "bin" / "python").exists():
            directory = directory.resolve()
            spec = KernelSpecInfo(
                name=f"{_VENV_PREFIX}{_shorten(directory)}",
                display_name=f"{folder.name}/.venv",
                language="python",
                needs_ipykernel=False,
            )
            venvs.append(_Venv(directory, spec))
    return venvs


def _has_ipykernel(venv: Path) -> bool:
    """Whether the venv at *venv* has `ipykernel`, looked up as a file: the venv's python is not run to find out."""
    return next(venv.glob("lib*/python*/site-packages/ipykernel/__init__.py"), None) is not None


def _notebook_kernelspec(notebook: Path) -> str | None:
    """`metadata.kernelspec.name` of the notebook on disk, None when unreadable or missing."""
    try:
        name = json.loads(notebook.read_text(encoding="utf-8"))["metadata"]["kernelspec"]["name"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return name if isinstance(name, str) else None


class JupyterHost:
    """The server process and the kernels it runs for notebooks, safe to call from any thread."""

    def __init__(self, home: Path | None = None) -> None:
        self._home = home
        self._server: _Server | None = None
        self._server_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._kernels: dict[str, KernelRef] = {}
        self._notebook_locks: dict[str, threading.Lock] = {}
        self._async_client: httpx.AsyncClient | None = None

    @property
    def home(self) -> Path:
        return self._home or jupyter_home()

    # --- server -----------------------------------------------------------------------------------------------------------

    def ensure_running(self) -> _Server:
        """The running server, started (or started again after it died, without the kernels it took along) when needed."""
        with self._server_lock:
            if self._server is not None and self._server.process.poll() is None:
                return self._server
            self._drop_server()
            self._server = self._start_server()
            return self._server

    def _drop_server(self) -> None:
        server, self._server = self._server, None
        with self._state_lock:
            self._kernels.clear()
        if server is not None:
            server.client.close()
            self._terminate(server.process)
            server.scratch.cleanup()

    def _start_server(self) -> _Server:
        home = self.home
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        home.chmod(0o700)
        (home / "data" / "kernels").mkdir(parents=True, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(prefix="gigachad-jupyter-")
        runtime = Path(scratch.name)
        token = secrets.token_urlsafe(32)
        jupyter_path = os.pathsep.join(filter(None, [str(home / "data"), os.environ.get("JUPYTER_PATH")]))
        environment = owned_environment({
            **os.environ,
            "JUPYTER_RUNTIME_DIR": str(runtime),
            "JUPYTER_PATH": jupyter_path,
            "JUPYTER_NO_CONFIG": "1",
        })  # fmt: skip
        arguments = [
            "--no-browser", "--ServerApp.ip=127.0.0.1", "--ServerApp.port=0", "--ServerApp.port_retries=0",
            f"--ServerApp.base_url={BASE_URL}", "--ServerApp.root_dir=/", "--ServerApp.terminals_enabled=False",
            f"--MappingKernelManager.cull_idle_timeout={_KERNEL_CULL_IDLE_SECONDS}",
            f"--MappingKernelManager.cull_interval={_KERNEL_CULL_INTERVAL_SECONDS}", "--MappingKernelManager.cull_connected=False",
        ]  # fmt: skip
        log_descriptor = os.open(home / "server.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(log_descriptor, 0o600)
        log_file = os.fdopen(log_descriptor, "wb")
        try:
            process = subprocess.Popen(  # noqa: S603
                [sys.executable, "-c", _BOOTSTRAP, str(os.getpid()), str(_SERVER_STOP_SECONDS), *arguments],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            scratch.cleanup()
            raise UpstreamUnavailable(f"Jupyter server could not start: {exc}") from exc
        finally:
            log_file.close()
        try:
            with contextlib.suppress(OSError):  # a server that died already is reported by the wait below
                process.stdin.write(f"{token}\n".encode())
                process.stdin.close()
            port = self._wait_for_port(process, runtime)
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"token {token}"}, trust_env=False,
                                  timeout=_REQUEST_SECONDS)  # fmt: skip
            self._wait_until_serving(process, client)
        except BaseException:
            self._terminate(process)
            scratch.cleanup()
            raise
        return _Server(process, port, token, client, scratch)

    def _log_tail(self) -> str:
        try:
            return (self.home / "server.log").read_bytes()[-_LOG_TAIL_BYTES:].decode(errors="replace")
        except OSError:
            return ""

    def _wait_for_port(self, process: subprocess.Popen, runtime: Path) -> int:
        info = runtime / f"jpserver-{process.pid}.json"
        deadline = time.monotonic() + _SERVER_START_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise UpstreamUnavailable(f"Jupyter server exited with {process.returncode}:\n{self._log_tail()}")
            try:
                port = int(json.loads(info.read_text())["port"])
            except (OSError, ValueError, KeyError):
                time.sleep(0.1)
                continue
            # The info file holds the token and the kernels run as this user with this runtime folder: it is read once and gone.
            for leftover in runtime.glob(f"jpserver-{process.pid}*"):
                leftover.unlink(missing_ok=True)
            return port
        raise UpstreamUnavailable(f"Jupyter server did not start within {_SERVER_START_SECONDS} s:\n{self._log_tail()}")

    def _wait_until_serving(self, process: subprocess.Popen, client: httpx.Client) -> None:
        deadline = time.monotonic() + _SERVER_START_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise UpstreamUnavailable(f"Jupyter server exited with {process.returncode}:\n{self._log_tail()}")
            try:
                if client.get(f"{BASE_URL}api/status").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise UpstreamUnavailable(f"Jupyter server does not answer:\n{self._log_tail()}")

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(_SERVER_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def stop(self) -> None:
        """Stop the server and with it every kernel."""
        with self._server_lock:
            self._drop_server()

    async def aclose(self) -> None:
        """Close the proxy's HTTP client."""
        client, self._async_client = self._async_client, None
        if client is not None:
            await client.aclose()

    # --- upstream calls ---------------------------------------------------------------------------------------------------

    def _call(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        """A request to the running server, `UpstreamUnavailable` when it cannot be reached."""
        server = self.ensure_running()
        with unreachable("Jupyter server"):
            return server.client.request(method, f"{BASE_URL}{path}", **kwargs)

    @staticmethod
    def _parse[T: BaseModel](model: type[T], response: httpx.Response) -> T:
        return validate_answer(model, response.content, label="Jupyter server", path=response.request.url.path)

    def _discovered_specs(self) -> _UpstreamSpecs:
        response = self._call("GET", "api/kernelspecs")
        if response.status_code != 200:
            raise UpstreamContractError(f"Jupyter server answered {response.status_code} to the kernelspec list")
        return self._parse(_UpstreamSpecs, response)

    # --- kernelspecs ------------------------------------------------------------------------------------------------------

    def _write_venv_spec(self, venv: _Venv) -> None:
        directory = self.home / "data" / "kernels" / venv.spec.name
        directory.mkdir(parents=True, exist_ok=True)
        path = os.pathsep.join([str(venv.directory / "bin"), os.environ.get("PATH", "")])
        spec = {
            "argv": [str(venv.python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": venv.spec.display_name,
            "language": "python",
            "env": {"VIRTUAL_ENV": str(venv.directory), "PATH": path},
            "metadata": {"debugger": True},
        }
        text = json.dumps(spec, indent=1) + "\n"
        file = directory / "kernel.json"
        if not file.exists() or file.read_text() != text:
            staged = directory / ".kernel.json.tmp"
            staged.write_text(text)
            staged.replace(file)

    def _venv_specs(self, notebook: Path) -> tuple[list[_Venv], list[KernelSpecInfo]]:
        venvs = _find_venvs(notebook)
        specs = []
        for venv in venvs:
            self._write_venv_spec(venv)
            specs.append(venv.spec.model_copy(update={"needs_ipykernel": not _has_ipykernel(venv.directory)}))
        return venvs, specs

    def kernelspecs(self, path: str) -> KernelSpecList:
        """The kernelspecs to pick from for the notebook at *path*, with the one it opens with."""
        notebook = notebook_path(path)
        if not notebook.parent.is_dir():
            raise UpstreamRequestFailed(404, f"Folder not found: {notebook.parent}")
        _, venv_specs = self._venv_specs(notebook)
        discovered = self._discovered_specs()
        managed = {entry.name for entry in (self.home / "data" / "kernels").iterdir()}
        specs = [
            KernelSpecInfo(name=name, display_name=item.spec.display_name, language=item.spec.language, needs_ipykernel=False)
            for name, item in discovered.kernelspecs.items()
            if name not in managed
        ]
        names = {spec.name for spec in specs} | {spec.name for spec in venv_specs}
        app_default = APP_KERNELSPEC if APP_KERNELSPEC in names else discovered.default
        chosen = _notebook_kernelspec(notebook)
        default = chosen if chosen in names else venv_specs[0].name if venv_specs else app_default
        return KernelSpecList(default=default, specs=[*venv_specs, *specs])

    # --- kernels ----------------------------------------------------------------------------------------------------------

    def _notebook_lock(self, key: str) -> threading.Lock:
        with self._state_lock:
            return self._notebook_locks.setdefault(key, threading.Lock())

    def _is_running(self, kernel: KernelRef) -> bool:
        response = self._call("GET", f"api/kernels/{quote(kernel.kernel_id, safe='')}")
        return response.status_code == 200 and self._parse(_UpstreamKernel, response).id == kernel.kernel_id

    def _shutdown_upstream(self, kernel: KernelRef) -> None:
        with self._server_lock:
            server = self._server
        if server is None or server.process.poll() is not None:
            return
        try:
            server.client.delete(f"{BASE_URL}api/kernels/{quote(kernel.kernel_id, safe='')}")
        except httpx.HTTPError as exc:
            log.warning("Could not shut down kernel %s: %s", kernel.kernel_id, exc)

    def _install_ipykernel(self, venv: _Venv) -> None:
        uv = shutil.which("uv")
        if uv is None:
            raise UpstreamRequestFailed(502, "ipykernel is missing in the environment and `uv` is not installed to add it")
        installing = f"Installing ipykernel into {venv.directory}"
        try:
            result = subprocess.run(  # noqa: S603
                [uv, "pip", "install", "--python", str(venv.python), "ipykernel"],
                capture_output=True, text=True, timeout=_IPYKERNEL_INSTALL_SECONDS, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            raise UpstreamRequestFailed(502, f"{installing} took longer than {_IPYKERNEL_INSTALL_SECONDS} s") from exc
        except OSError as exc:
            raise UpstreamRequestFailed(502, f"{installing} failed: {exc}") from exc
        if result.returncode != 0:
            raise UpstreamRequestFailed(502, f"{installing} failed: {result.stderr.strip()[-_LOG_TAIL_BYTES:]}")

    def start_kernel(self, path: str, name: str, directory: str) -> KernelRef:
        """The kernel of the notebook at *path* running kernelspec *name*: the running one, or a new one in the absolute *directory* replacing another."""
        notebook = notebook_path(path)
        listed = self.kernelspecs(str(notebook))
        if name not in {spec.name for spec in listed.specs}:
            raise UpstreamRequestFailed(404, f"Unknown kernelspec: {name}")
        key = str(notebook)
        with self._notebook_lock(key):
            with self._state_lock:
                current = self._kernels.get(key)
            if current is not None:
                if current.name == name and self._is_running(current):
                    return current
                self._shutdown_upstream(current)
                with self._state_lock:
                    self._kernels.pop(key, None)
            venv = next((venv for venv in _find_venvs(notebook) if venv.spec.name == name), None)
            if venv is not None and not _has_ipykernel(venv.directory):
                self._install_ipykernel(venv)
            cwd = Path(directory).resolve().relative_to("/")
            response = self._call("POST", "api/kernels", json={"name": name, "path": str(cwd)})
            if response.status_code != 201:
                raise UpstreamContractError(
                    f"Jupyter server answered {response.status_code} to the kernel start: {response.text[:300]}"
                )
            upstream = self._parse(_UpstreamKernel, response)
            kernel = KernelRef(kernel_id=upstream.id, name=upstream.name)
            with self._state_lock:
                self._kernels[key] = kernel
            return kernel

    def shutdown_kernel(self, path: str) -> None:
        """Shut down the kernel of the notebook at *path*, nothing when it has none."""
        self.shutdown_kernels_under(str(notebook_path(path)))

    def shutdown_kernels_under(self, path: str) -> None:
        """Shut down the kernels of the notebook at *path* or below the folder *path*."""
        for kernel in self._take_kernels(path):
            self._shutdown_upstream(kernel)

    def _take_kernels(self, path: str) -> list[KernelRef]:
        prefix = str(Path(path).resolve())
        with self._state_lock:
            keys = [key for key in self._kernels if key == prefix or key.startswith(f"{prefix}/")]
            return [self._kernels.pop(key) for key in keys]

    def move_kernel(self, old: str, new: str) -> None:
        """Keep the kernel of the notebook *old* under its new path *new*."""
        source, target = str(Path(old).resolve()), str(Path(new).resolve())
        with self._state_lock:
            if source in self._kernels:
                self._kernels[target] = self._kernels.pop(source)

    def check_session(self, method: str, body: bytes) -> None:
        """Refuse a browser's session request that would start a kernel or bind one this host did not start."""
        # A session only ties a kernel this host started to the library's session path: its body names that kernel by id and nothing else.
        try:
            model = json.loads(body)
        except ValueError as exc:
            raise UpstreamRequestFailed(400, "The session request is not JSON") from exc
        if not isinstance(model, dict) or not model.keys() <= _SESSION_KEYS:
            raise UpstreamRequestFailed(400, "The session request has fields a session does not take")
        path = model.get("path")
        if path is not None and (not isinstance(path, str) or path.startswith("/") or ".." in path.split("/") or "\x00" in path):
            raise UpstreamRequestFailed(400, "The session path is not a relative path")
        kernel = model.get("kernel")
        if kernel is None and method == "POST":
            raise UpstreamRequestFailed(403, "A session needs the kernel of its notebook, start it with POST /api/jupyter/kernel")
        if kernel is not None and not self._runs_kernel(kernel):
            raise UpstreamRequestFailed(403, "A session may only use the kernel this host started for a notebook")

    def _runs_kernel(self, kernel: object) -> bool:
        """Whether *kernel* (a session request's `kernel` field) is `{"id": <kernel of a notebook>}`."""
        if not isinstance(kernel, dict) or not kernel.keys() <= _SESSION_KERNEL_KEYS:
            return False
        kernel_id = kernel.get("id")
        with self._state_lock:
            return isinstance(kernel_id, str) and any(known.kernel_id == kernel_id for known in self._kernels.values())

    # --- proxy ------------------------------------------------------------------------------------------------------------

    async def open_http(
        self,
        method: str,
        path: str,
        query: Iterable[tuple[str, str]],
        headers: Iterable[tuple[str, str]],
        body: AsyncIterator[bytes] | None,
    ) -> httpx.Response:
        """Send the browser's request to upstream *path* and return the streaming response, the caller closes it, `UpstreamUnavailable` when the server cannot be reached."""
        server = await asyncio.to_thread(self.ensure_running)
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(_REQUEST_SECONDS, read=None))
        request = self._async_client.build_request(
            method,
            f"{server.http_base}{_upstream_target(path, query)}",
            headers=[*forwardable_headers(headers, _DROPPED_REQUEST_HEADERS), *server.headers.items()],
            content=body,
        )
        with unreachable("Jupyter server"):
            return await self._async_client.send(request, stream=True)

    async def open_websocket(self, path: str, query: Iterable[tuple[str, str]], subprotocols: Sequence[str]) -> ClientConnection:
        """Connect to the upstream WebSocket *path*, raises `UpstreamUnavailable` when the server refuses or cannot be reached."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import WebSocketException

        server = await asyncio.to_thread(self.ensure_running)
        url = f"ws://127.0.0.1:{server.port}{_upstream_target(path, query)}"
        try:
            return await connect(
                url,
                additional_headers=server.headers,
                subprotocols=[*subprotocols] or None,  # type: ignore[list-item]
                max_size=None,
                compression=None,
                ping_interval=None,
                proxy=None,
            )
        except (OSError, WebSocketException, TimeoutError) as exc:
            raise UpstreamUnavailable(f"Jupyter server refused the connection: {exc}") from exc


_host: JupyterHost | None = None
_host_lock = threading.Lock()


def host() -> JupyterHost:
    """The process-wide `JupyterHost`."""
    global _host
    with _host_lock:
        if _host is None:
            _host = JupyterHost()
            atexit.register(_host.stop)
        return _host


def move_kernel(old: str, new: str) -> None:
    """The notebook *old* moved to *new*: its kernel follows."""
    with _host_lock:
        current = _host
    if current is not None:
        current.move_kernel(old, new)


def shutdown_kernels_under(path: str) -> None:
    """The notebook *path* (or the folder of notebooks) is gone: shut down its kernels."""
    with _host_lock:
        current = _host
    if current is not None:
        current.shutdown_kernels_under(path)


async def shutdown() -> None:
    """Stop the server (and the kernels) and close the proxy client, for the end of the backend."""
    with _host_lock:
        current = _host
    if current is not None:
        await asyncio.to_thread(current.stop)
        await current.aclose()
