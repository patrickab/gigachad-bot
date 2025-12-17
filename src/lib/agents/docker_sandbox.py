from dataclasses import dataclass
import os
import subprocess
import tarfile
import tempfile
import time
from typing import List, Optional

import docker
from docker.errors import DockerException, NotFound

from lib.utils.logger import get_logger

logger = get_logger()


@dataclass
class ExecutionLogs:
    """Captured stdout/stderr from an execution."""

    stdout: str = ""
    stderr: str = ""


@dataclass
class Artifacts:
    """Optional structured results (e.g. images, binary data)."""

    png: Optional[bytes] = None


@dataclass
class CodeExecution:
    """High-level outcome of a code execution request."""

    logs: ExecutionLogs
    results: List[Artifacts]


@dataclass
class CommandResult:
    """Low-level result from a shell command."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class SandboxFileSystem:
    """Handles file I/O strictly scoped to the running container."""

    def __init__(self, sandbox: "DockerSandbox") -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> str:
        """Reads a text file from the container."""
        logger.debug("FileSystem: Reading '%s'", path)
        result = self._sandbox.exec_run(f"cat {path}")

        if result.exit_code != 0:
            msg = f"Read failed for {path}: {result.stderr}"
            logger.error(msg)
            raise IOError(msg)

        return result.stdout

    def write(self, path: str, content: str) -> None:
        """Writes text content to a file in the container."""
        logger.debug("FileSystem: Writing to '%s'", path)
        container = self._sandbox.get_container()

        # Ensure parent directory exists
        directory = os.path.dirname(path)
        if directory and directory != "/":
            self._sandbox.exec_run(f"mkdir -p {directory}")

        # Use tar archive to safely inject file into container
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = os.path.basename(path)
                local_path = os.path.join(temp_dir, filename)

                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)

                tar_path = os.path.join(temp_dir, "payload.tar")
                with tarfile.open(tar_path, "w") as tar:
                    tar.add(local_path, arcname=filename)

                with open(tar_path, "rb") as f:
                    tar_data = f.read()

                target_dir = directory if directory else "/"
                if not container.put_archive(target_dir, tar_data):
                    raise IOError("Docker put_archive failed")

        except Exception as e:
            logger.error("FileSystem: Write error: %s", e)
            raise IOError(f"Failed to write {path}: {e}")


class SandboxShell:
    """Facade for executing shell commands in the sandbox."""

    def __init__(self, sandbox: "DockerSandbox") -> None:
        self._sandbox = sandbox

    def run(self, command: str) -> CommandResult:
        """Executes a shell command and logs the outcome."""
        logger.debug("Shell: Executing '%s'", command)
        result = self._sandbox.exec_run(command)

        if result.exit_code != 0:
            logger.warning("Shell: Command failed (code %d): %s", result.exit_code, command)
        else:
            logger.debug("Shell: Command success")

        return result


class DockerSandbox:
    """Orchestrates the Docker environment, repository cloning, and code execution."""

    DEFAULT_IMAGE = "aci-docker-sandbox:latest"
    DOCKERFILE_TEMPLATE = """
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_CACHE_DIR="/uv_cache"
ENV UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y git curl grep procps build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/workspace
CMD ["tail", "-f", "/dev/null"]
"""

    def __init__(self, repo_url: str, branch: str, image_name: str = DEFAULT_IMAGE) -> None:
        logger.info("Sandbox: Initializing for %s (branch: %s)", repo_url, branch)

        self.image_name = image_name
        self.container_name = f"aci-sandbox-{os.urandom(4).hex()}"
        self.container_id: Optional[str] = None

        # Normalize Repo URL
        if repo_url and not repo_url.endswith(".git"):
            repo_url += ".git"
        if repo_url and not repo_url.startswith("https://"):
            repo_url = "https://" + repo_url

        # Setup Workspace Paths
        self.workspace_root = os.path.abspath(os.path.expanduser("~/agent_workspaces"))
        self.repo_name = repo_url.split("/")[-1].replace(".git", "") if repo_url else "scratchpad"
        self.host_repo_path = os.path.join(self.workspace_root, self.repo_name)

        # Initialize Docker Client
        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as e:
            logger.critical("Docker unavailable: %s", e)
            raise RuntimeError("Docker is not accessible. Please ensure the daemon is running.")

        # Setup Environment
        if repo_url:
            self._clone_repository(repo_url, branch)
        else:
            os.makedirs(self.host_repo_path, exist_ok=True)

        self._provision_image()
        self._start_container()

        # Initialize Helpers
        self.files = SandboxFileSystem(self)
        self.commands = SandboxShell(self)

        if repo_url:
            self._install_dependencies()

        logger.info("Sandbox: Ready (Container: %s)", self.container_name)

    def get_container(self) -> docker.models.containers.Container:
        """Retrieves the active container object."""
        if not self.container_id:
            raise RuntimeError("Container not running")
        return self.client.containers.get(self.container_id)

    def exec_run(self, command: str) -> CommandResult:
        """Executes a command directly in the container."""
        try:
            container = self.get_container()
            exec_res = container.exec_run(cmd=["/bin/bash", "-c", command], workdir="/app/workspace", demux=True)

            stdout = exec_res.output[0].decode() if exec_res.output[0] else ""
            stderr = exec_res.output[1].decode() if exec_res.output[1] else ""
            return CommandResult(stdout, stderr, exec_res.exit_code)

        except Exception as e:
            logger.error("Exec failed: %s", e)
            return CommandResult(stderr=str(e), exit_code=1)

    def run_code(self, code: str) -> CodeExecution:
        """Injects and executes a Python script."""
        script_name = f"script_{os.urandom(4).hex()}.py"
        script_path = f"/app/workspace/{script_name}"

        try:
            self.files.write(script_path, code)
            result = self.exec_run(f"python3 {script_name}")
            self.exec_run(f"rm {script_name}")

            logs = ExecutionLogs(result.stdout, result.stderr)
            return CodeExecution(logs, [])

        except Exception as e:
            logger.error("Code execution failed: %s", e)
            return CodeExecution(ExecutionLogs(stderr=str(e)), [])

    def stop(self) -> None:
        """Terminates and removes the container."""
        if not self.container_id:
            return

        try:
            container = self.client.containers.get(self.container_id)
            container.stop()
            container.remove()
            logger.info("Sandbox: Container %s removed", self.container_name)
        except NotFound:
            pass
        finally:
            self.container_id = None

    def _clone_repository(self, url: str, branch: str) -> None:
        if os.path.exists(self.host_repo_path):
            return

        logger.info("Sandbox: Cloning %s...", url)
        try:
            subprocess.run(["git", "clone", "-b", branch, url, self.host_repo_path], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone failed: {e.stderr.decode()}")

    def _provision_image(self) -> None:
        """Ensures the Docker image exists, building it if necessary."""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            logger.info("Sandbox: Building image %s...", self.image_name)
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "Dockerfile"), "w") as f:
                    f.write(self.DOCKERFILE_TEMPLATE)
                self.client.images.build(path=td, tag=self.image_name, rm=True)

    def _start_container(self) -> None:
        """Starts the sandbox container with necessary volume mounts."""
        # Ensure cache volume
        try:
            self.client.volumes.get("agent_uv_cache")
        except NotFound:
            self.client.volumes.create("agent_uv_cache")

        # Check for existing container
        try:
            container = self.client.containers.get(self.container_name)
            if container.status != "running":
                container.start()
            self.container_id = container.id
            return
        except NotFound:
            pass

        # Start new container
        logger.info("Sandbox: Launching container %s", self.container_name)
        container = self.client.containers.run(
            self.image_name,
            name=self.container_name,
            detach=True,
            extra_hosts={"host.docker.internal": "host-gateway"},
            environment={"LLM_API_BASE": "http://host.docker.internal:4000"},
            volumes={
                self.host_repo_path: {"bind": "/app/workspace", "mode": "rw"},
                "agent_uv_cache": {"bind": "/uv_cache", "mode": "rw"},
                "/dummy_venv_mount": {"bind": "/app/workspace/.venv", "mode": "rw"},
                "/dummy_venv_mount_2": {"bind": "/app/workspace/venv", "mode": "rw"},
            },
            remove=False,
        )
        self.container_id = container.id
        time.sleep(1)  # Allow process startup

    def _install_dependencies(self) -> None:
        if os.path.exists(os.path.join(self.host_repo_path, "requirements.txt")):
            logger.info("Sandbox: Installing dependencies...")
            self.exec_run("uv pip install --system -r requirements.txt")
