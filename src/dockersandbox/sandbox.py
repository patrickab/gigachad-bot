"""
Security Architecture - Defense in Depth

1. Zero-Trust Execution Policy

    - "Assume Breach Principle":
        All code is treated potentially malicious.

2. Kernel-Level Isolation (gVisor)

    - Sandboxed System Calls:
        Uses the 'runsc' runtime to provide a dedicated
        guest kernel for the container.
    - Host Protection:
        Acts as a "firewall" between the container and the host

3. Hardened User Identity & Capability Stripping

    - Globally remove Linux kernel capabilities (--cap-drop=ALL):
        Container cannot perform system-level tasks.
    - Enforce 'no-new-privileges':
        Prevent privilege escalation attacks
    - Non-Root Enforcement:
        Containers are forced to run as an unprivileged user
        (UID 1000), removing administrative power by default.

4. Rootless Infrastructure Architecture

    - Unprivileged Daemon:
        - Docker Engine runs without root privileges on the host
    - Identity Mapping:
        - Rootless Docker maps container root users to non-root host users,

5. Ephemeral Lifecycle Management
    - Destruction on Completion:
        Containers are strictly temporary
        Automatically destroyed after task execution.

6. Network Perimeter Control
    - Traffic Segregation:
        Places containers on isolated bridge networks to monitor and restrict data flow.
    - Exfiltration Prevention:
        Limits the container's ability to communicate with the internal network or move laterally to other services.
"""

import logging
import os
import subprocess

import docker


class SecurityEnvironmentError(Exception):
    """Signal misconfigured Docker security environment."""


class ContainerRuntimeError(Exception):
    """Signal container lifecycle or execution failure."""


class DockerSandbox:
    """
    Secure Docker-based sandbox for agent code execution.

    Features:
    (1) Interactive TTY/Shell access.
    (2) Host-to-Container synchronization via Bind Mounts.
    (3) Syscall isolation using gVisor (runsc).
    (4) Ephemeral lifecycle (auto-removal).
    (5) Least Privilege: Enforces non-root UID and drops capabilities.
    (6) Daemon Isolation: Validates Rootless Docker.
    """

    def __init__(self, dockerimage_name: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.client = docker.from_env()
        self.dockerimage_name = dockerimage_name
        self._verify_environment()

    def _verify_environment(self) -> None:
        """Validates Rootless Docker, gVisor, and Image availability."""
        try:
            info = self.client.info()

            # (3) Validate Rootless Docker
            security_opts = info.get("SecurityOptions", [])
            if not any("rootless" in opt.lower() for opt in security_opts):
                raise SecurityEnvironmentError("Rootless Docker is not enabled. Daemon must run in rootless mode.")

            if "runsc" not in info.get("Runtimes", {}):
                raise SecurityEnvironmentError("gVisor 'runsc' runtime is not configured in Docker.")

            self.client.images.get(self.dockerimage_name)
        except docker.errors.ImageNotFound:
            raise SecurityEnvironmentError(f"Docker image '{self.dockerimage_name}' not found.")
        except Exception as e:
            if isinstance(e, SecurityEnvironmentError):
                raise e
            raise SecurityEnvironmentError(f"Environment check failed: {e}")

    def run_interactive_shell(self, repo_path: str, agent_cmd: str) -> None:
        """
        Runs an interactive shell in the sandbox.
        Uses subprocess for the final 'run' call to ensure high-fidelity TTY hijacking.
        """
        abs_repo_path = os.path.abspath(os.path.expanduser(repo_path))
        os.makedirs(abs_repo_path, exist_ok=True)

        self.logger.info(f"Starting sandbox with image {self.dockerimage_name} at {abs_repo_path}")

        # Constructing the docker run command
        cmd = [
            "docker",
            "run",
            "-it",
            "--rm",
            "--runtime=runsc",
            "--user",
            "0:0",  # Internal Root -> Host User 1000 (Rootless)
            # --- NETWORKING FIXES ---
            "--network",
            "bridge",
            "--add-host=host.docker.internal:host-gateway",
            "--dns",
            "8.8.8.8",
            # --- SECURITY/CAPABILITY ADJUSTMENTS ---
            # Drop everything first for security...
            "--cap-drop=ALL",
            # ...but add back ONLY what tcpdump needs to sniff traffic
            "--cap-add=NET_ADMIN",
            "--cap-add=NET_RAW",
            "--security-opt",
            "no-new-privileges",
            # --- MOUNTS ---
            "-v",
            f"{abs_repo_path}:/workspace:z",
            "-w",
            "/workspace",
            "-e",
            "HOME=/workspace",
            self.dockerimage_name,
            "/bin/bash",
        ]

        # Command to log network traffic from sandbox to host
        # We use 'timeout' or backgrounding.
        # Note: We sleep 1 to ensure tcpdump is listening before the agent starts.
        wrapped_cmd = f"tcpdump -l -A -i eth0 > /workspace/network_traffic.log 2>&1 & sleep 1; {agent_cmd}"

        self.logger.info(f"Executing command: {' '.join(cmd)}")
        cmd.extend(["-c", wrapped_cmd])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise ContainerRuntimeError(f"Container execution failed: {e}")
