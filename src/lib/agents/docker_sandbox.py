"""
DockerSandbox: Foundation for Agentic, Git-Native, Test-Driven Development in Isolated Environments

# Core Behavior:
================
- Initializes from a git repository or scratchpad with automatic cloning/pulling.
- Runs all commands inside an isolated Docker container built with `uv` for fast dependency resolution and caching.
- Exposes file I/O and command execution APIs that enforce container encapsulation.
- Automatically mounts the host workspace for seamless local-dev parity.
- Supports test-driven workflows via programmatic script execution and result introspection.

## What It Does
===============
* Create an ephemeral Docker container for code execution.
* Initialize a git workspace by cloning or creating a repository.
* Run all commands **only inside the container**.
* Expose controlled APIs for file I/O and command execution.
* Mount the host workspace for local parity and persistence.
* Support test-driven development with executable scripts and captured results.
* Enforce clear authority boundaries between host and container.

--- Overview

### Initialization
==================
* Clone or pull a git repository, or start from an empty workspace.
* Build a Docker image using `python:3.11-slim`.
* Install `uv`, `git`, and common tooling.
* Start a container tied to the sandbox instance lifecycle.

### Execution
=============
* Execute shell commands inside the container.
* Run Python code and scripts programmatically.
* Capture stdout, stderr, exit codes, and logs.
* Read and write files inside the container via streamed I/O.

### Isolation
=============
* Mount the host repo at `/app/workspace` (read/write).
* Hide host virtual environments with dummy mounts.
* Share a persistent `uv` cache volume for faster installs.
* Allow container access to host services when needed.

### Lifecycle
=============
* Treat the container as disposable.
* Automatically clean up the container on sandbox termination.

---

## Key Components
=================
* **DockerSandbox** — Primary interface; manages lifecycle and orchestration.
* **_DockerFiles** — Read and write files inside the container.
* **_DockerCommands** — Execute shell commands inside the container.
* **DockerExecution** — Collect logs and execution results.
* **DockerProcessResult** — Represent stdout, stderr, and exit codes.
"""

import os
import subprocess
import tarfile
import tempfile
import time
from typing import List, Optional

import docker
from docker.errors import NotFound

from lib.utils.logger import get_logger

logger = get_logger()


class DockerExecutionLogs:
    """Captured stdout/stderr from an execution."""

    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class DockerExecutionResult:
    """Optional structured results (e.g. images, artifacts)."""

    def __init__(self, png: Optional[bytes] = None) -> None:
        self.png = png


class DockerExecution:
    """High-level execution outcome."""

    def __init__(self, logs: DockerExecutionLogs, results: List[DockerExecutionResult]) -> None:
        self.logs = logs
        self.results = results


class DockerProcessResult:
    """Low-level process result from container exec."""

    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _DockerFiles:
    """File I/O strictly scoped to the running container."""

    def __init__(self, sandbox_instance: "DockerSandbox") -> None:
        self._sandbox = sandbox_instance

    def read(self, path: str) -> str:
        logger.debug("Sandbox: Reading file '%s'", path)
        # Simple read via container shell
        proc = self._sandbox._run_in_container(f"cat {path}")
        if proc.exit_code != 0:
            error_msg = f"Failed to read file {path}: {proc.stderr}"
            logger.error("Sandbox: %s", error_msg)
            raise Exception(error_msg)
        logger.debug("Sandbox: Successfully read file '%s'", path)
        return proc.stdout

    def write(self, path: str, content: str) -> None:
        logger.debug("Sandbox: Writing to file '%s'", path)
        # Writes via tar+put_archive (Docker-safe file injection)
        if not self._sandbox.container_id:
            error_msg = "Docker sandbox container is not running."
            logger.error("Sandbox: %s", error_msg)
            raise RuntimeError(error_msg)

        container = self._sandbox.docker_client.containers.get(self._sandbox.container_id)
        container_dir = os.path.dirname(path)
        container_filename = os.path.basename(path)

        if container_dir and container_dir != "/":
            proc = self._sandbox._run_in_container(f"mkdir -p {container_dir}")
            if proc.exit_code != 0:
                error_msg = f"Failed to create directory {container_dir}: {proc.stderr}"
                logger.error("Sandbox: %s", error_msg)
                raise Exception(error_msg)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, container_filename)
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(content)

                tar_path = os.path.join(temp_dir, "temp.tar")
                with tarfile.open(tar_path, "w") as tar:
                    tar.add(temp_file, arcname=container_filename)

                with open(tar_path, "rb") as f:
                    tar_data = f.read()

                target = container_dir if container_dir else "/"
                if not container.put_archive(target, tar_data):
                    error_msg = "put_archive failed"
                    logger.error("Sandbox: %s", error_msg)
                    raise Exception(error_msg)
            logger.debug("Sandbox: Successfully wrote to file '%s'", path)
        except Exception as e:
            error_msg = f"Docker error during file write: {e}"
            logger.error("Sandbox: %s", error_msg)
            raise Exception(error_msg)


class _DockerCommands:
    """Command execution facade."""

    def __init__(self, sandbox_instance: "DockerSandbox") -> None:
        self._sandbox = sandbox_instance

    def run(self, command: str) -> DockerProcessResult:
        logger.debug("Sandbox: Running command: %s", command)
        result = self._sandbox._run_in_container(command)
        if result.exit_code != 0:
            logger.warning("Sandbox: Command exited with code %d: %s", result.exit_code, command)
        else:
            logger.debug("Sandbox: Command executed successfully: %s", command)
        return result


class DockerSandbox:
    """Manages repo prep, image/container lifecycle, and execution APIs."""

    def __init__(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        image_name: str = "agent-worker-uv",
    ) -> None:
        logger.info("Sandbox: Initializing DockerSandbox with repo_url=%s, branch=%s", repo_url, branch)
        self.docker_client = docker.from_env()
        self.image_name = image_name
        self.container_id: Optional[str] = None
        self.container_name = f"aci-sandbox-{os.urandom(4).hex()}"

        # Host-side workspace (for auth, visibility, persistence)
        self.host_workspace_root = os.path.abspath(os.path.expanduser("~/agent_workspaces"))
        self.repo_name = repo_url.split("/")[-1].replace(".git", "") if repo_url else "scratchpad"
        self.host_repo_path = os.path.join(self.host_workspace_root, self.repo_name)

        if repo_url:
            self._prepare_repository(repo_url, branch)
        else:
            logger.debug("Sandbox: Creating scratchpad workspace at %s", self.host_repo_path)
            os.makedirs(self.host_repo_path, exist_ok=True)

        self.files = _DockerFiles(self)
        self.commands = _DockerCommands(self)

        self._ensure_image_and_container()

        # Auto-install deps if conventional requirements.txt exists
        if repo_url:
            self._install_dependencies()
        logger.info("Sandbox: DockerSandbox initialized successfully")

    def _prepare_repository(self, repo_url: str, branch: str) -> None:
        """Clone once; reuse if already present."""
        logger.debug("Sandbox: Preparing repository %s (branch: %s)", repo_url, branch)
        if os.path.exists(self.host_repo_path):
            logger.debug("Sandbox: Repository already exists at %s", self.host_repo_path)
            return
        try:
            logger.info("Sandbox: Cloning repository %s to %s", repo_url, self.host_repo_path)
            subprocess.run(
                ["git", "clone", "-b", branch, repo_url, self.host_repo_path],
                check=True,
                capture_output=True,
            )
            logger.debug("Sandbox: Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"Git clone failed: {e.stderr.decode()}"
            logger.error("Sandbox: %s", error_msg)
            raise Exception(error_msg)

    def _ensure_image_and_container(self) -> None:
        logger.debug("Sandbox: Ensuring Docker image and container")
        # Build image if missing
        try:
            self.docker_client.images.get(self.image_name)
            logger.debug("Sandbox: Docker image %s found", self.image_name)
        except docker.errors.ImageNotFound:
            logger.info("Sandbox: Building Docker image %s", self.image_name)
            dockerfile = """
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_CACHE_DIR="/uv_cache"
ENV UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y git curl grep procps build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/workspace
CMD ["tail", "-f", "/dev/null"]
"""
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "Dockerfile"), "w") as f:
                    f.write(dockerfile)
                self.docker_client.images.build(path=td, tag=self.image_name, rm=True)
            logger.debug("Sandbox: Docker image built successfully")

        # Shared uv cache volume
        try:
            self.docker_client.volumes.get("agent_uv_cache")
            logger.debug("Sandbox: UV cache volume found")
        except NotFound:
            logger.info("Sandbox: Creating UV cache volume")
            self.docker_client.volumes.create("agent_uv_cache")

        # Start (or reuse) container
        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status != "running":
                logger.info("Sandbox: Starting existing container %s", self.container_name)
                container.start()
            self.container_id = container.id
            logger.debug("Sandbox: Using existing container %s", self.container_name)
        except NotFound:
            logger.info("Sandbox: Creating new container %s", self.container_name)
            container = self.docker_client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                extra_hosts={"host.docker.internal": "host-gateway"},
                environment={"LLM_API_BASE": "http://host.docker.internal:4000"},
                volumes={
                    self.host_repo_path: {"bind": "/app/workspace", "mode": "rw"},
                    "agent_uv_cache": {"bind": "/uv_cache", "mode": "rw"},
                    # Hide host venvs to avoid leakage
                    "/dummy_venv_mount": {"bind": "/app/workspace/.venv", "mode": "rw"},
                    "/dummy_venv_mount_2": {"bind": "/app/workspace/venv", "mode": "rw"},
                },
                remove=False,
            )
            self.container_id = container.id
            time.sleep(1)
            logger.debug("Sandbox: New container started with ID %s", self.container_id)

    def _install_dependencies(self) -> None:
        """Installs requirements.txt using uv into system Python."""
        req_path = os.path.join(self.host_repo_path, "requirements.txt")
        if os.path.exists(req_path):
            logger.info("Sandbox: Installing dependencies from requirements.txt")
            self._run_in_container("uv pip install --system -r requirements.txt")
            logger.debug("Sandbox: Dependencies installed successfully")

    def _run_in_container(self, command: str) -> DockerProcessResult:
        # Single-shot exec with stdout/stderr capture
        logger.debug("Sandbox: Executing in container: %s", command)
        if not self.container_id:
            error_msg = "Docker sandbox container is not running."
            logger.error("Sandbox: %s", error_msg)
            raise RuntimeError(error_msg)

        container = self.docker_client.containers.get(self.container_id)
        try:
            exec_result = container.exec_run(
                cmd=["/bin/bash", "-c", command], 
                workdir="/app/workspace",
                stream=False,
                demux=True,
            )
            stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode() if exec_result.output[1] else ""
            logger.debug("Sandbox: Container execution completed with exit code %d", exec_result.exit_code)
            return DockerProcessResult(stdout, stderr, exec_result.exit_code)
        except Exception as e:
            error_msg = f"Docker exec error: {e}"
            logger.error("Sandbox: %s", error_msg)
            return DockerProcessResult(stderr=error_msg, exit_code=1)

    def run_code(self, code: str) -> DockerExecution:
        """Writes and executes a temporary Python script in-container."""
        logger.debug("Sandbox: Running Python code snippet")
        name = f"temp_script_{os.urandom(4).hex()}.py"
        path = f"/app/workspace/{name}"
        try:
            self.files.write(path, code)
            proc = self._run_in_container(f"python3 {name}")
            self._run_in_container(f"rm {name}")
            logger.debug("Sandbox: Python code executed successfully")
            return DockerExecution(DockerExecutionLogs(proc.stdout, proc.stderr), [])
        except Exception as e:
            error_msg = f"Error running Python code: {e}"
            logger.error("Sandbox: %s", error_msg, exc_info=True)
            return DockerExecution(DockerExecutionLogs(stderr=error_msg), [])

    def stop(self) -> None:
        """Stop and remove the container."""
        logger.info("Sandbox: Stopping container")
        if self.container_id:
            try:
                c = self.docker_client.containers.get(self.container_id)
                c.stop()
                c.remove()
                logger.debug("Sandbox: Container stopped and removed")
            except NotFound:
                logger.warning("Sandbox: Container not found during stop operation")
                pass
            self.container_id = None
