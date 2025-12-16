import os
import tarfile
import tempfile
import time
from typing import List, Optional

import docker


class DockerExecutionLogs:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class DockerExecutionResult:
    def __init__(self, png: Optional[bytes] = None) -> None:
        self.png = png


class DockerExecution:
    def __init__(self, logs: DockerExecutionLogs, results: List[DockerExecutionResult]) -> None:
        self.logs = logs
        self.results = results


class DockerProcess:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _DockerFiles:
    def __init__(self, sandbox_instance: "DockerSandbox") -> None:
        self._sandbox = sandbox_instance

    def read(self, path: str) -> str:
        cmd = f"cat {path}"
        proc = self._sandbox._run_in_container(cmd)
        if proc.exit_code != 0:
            raise Exception(f"Failed to read file {path}: {proc.stderr}")
        return proc.stdout

    def write(self, path: str, content: str) -> None:
        if not self._sandbox.container_id:
            raise RuntimeError("Docker sandbox container is not running.")

        container = self._sandbox.docker_client.containers.get(self._sandbox.container_id)

        container_dir = os.path.dirname(path)
        container_filename = os.path.basename(path)

        # 1. Ensure the directory exists in the container
        if container_dir and container_dir != "/":
            mkdir_proc = self._sandbox._run_in_container(f"mkdir -p {container_dir}")
            if mkdir_proc.exit_code != 0:
                raise Exception(f"Failed to create directory {container_dir}: {mkdir_proc.stderr}")

        # 2. Use put_archive to transfer the file content, bypassing the 'input' argument issue
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, container_filename)
                
                # Write content to the temporary file
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Create a tar archive
                tar_path = os.path.join(temp_dir, 'temp.tar')
                with tarfile.open(tar_path, 'w') as tar:
                    # Add the file to the tarball, ensuring its name is just the filename 
                    # for extraction relative to the target directory.
                    tar.add(temp_file_path, arcname=container_filename)

                # Read the tarball data
                with open(tar_path, 'rb') as f:
                    tar_data = f.read()
                
                # Use container_dir as the target path for extraction
                target_path = container_dir if container_dir else '/'
                success = container.put_archive(target_path, tar_data)
                
                if not success:
                    raise Exception(f"Docker put_archive failed to write file to {path}.")
                    
        except Exception as e:
            raise Exception(f"Docker error during file write using put_archive: {e}")


class _DockerCommands:
    def __init__(self, sandbox_instance: "DockerSandbox") -> None:
        self._sandbox = sandbox_instance

    def run(self, command: str) -> DockerProcess:
        return self._sandbox._run_in_container(command)


class DockerSandbox:
    def __init__(self, image_name: str = "python-sandbox-agent") -> None:
        self.docker_client = docker.from_env()
        self.image_name = image_name
        self.container_id: Optional[str] = None
        self.container_name = f"aci-sandbox-{os.urandom(4).hex()}"
        self.workspace_path = os.path.abspath("workspace")

        self.files = _DockerFiles(self)
        self.commands = _DockerCommands(self)

        self._ensure_image_and_container()

    def _ensure_image_and_container(self) -> None:
        try:
            self.docker_client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Docker image '{self.image_name}' not found. Building a default Python sandbox image.")
            dockerfile_content = """
FROM python:3.11-slim
WORKDIR /app
# Install system tools
RUN apt-get update && apt-get install -y git curl grep procps && rm -rf /var/lib/apt/lists/*
# Install Agent dependencies
RUN pip install requests openai pydantic
# Create workspace
RUN mkdir -p /app/workspace
CMD ["tail", "-f", "/dev/null"]
"""
            with tempfile.TemporaryDirectory() as temp_dir:
                dockerfile_path = os.path.join(temp_dir, "Dockerfile")
                with open(dockerfile_path, "w") as f:
                    f.write(dockerfile_content)

                print(f"Building image {self.image_name} from temporary Dockerfile...")
                self.docker_client.images.build(path=temp_dir, tag=self.image_name, rm=True)
                print(f"Image {self.image_name} built successfully.")

        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status != "running":
                container.start()
            self.container_id = container.id
        except docker.errors.NotFound:
            print(f"Container '{self.container_name}' not found. Creating and starting a new one.")
            
            # Ensure host workspace exists
            os.makedirs(self.workspace_path, exist_ok=True)

            container = self.docker_client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                mem_limit="2g",
                cpu_period=100000,
                cpu_quota=100000,
                # Networking: Allow access to Host LLM
                extra_hosts={"host.docker.internal": "host-gateway"},
                environment={
                    "LLM_API_BASE": "http://host.docker.internal:4000",
                },
                # Persistence: Shared Workspace
                volumes={self.workspace_path: {"bind": "/app/workspace", "mode": "rw"}},
                remove=False,
            )
            self.container_id = container.id
            print(f"Container '{self.container_name}' started with ID: {self.container_id}")
            time.sleep(1)

    def _run_in_container(self, command: str) -> DockerProcess:
        if not self.container_id:
            raise RuntimeError("Docker sandbox container is not running.")

        container = self.docker_client.containers.get(self.container_id)

        try:
            exec_result = container.exec_run(
                cmd=command,
                stream=False,
                demux=True,
            )

            stdout = exec_result.output[0].decode("utf-8") if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode("utf-8") if exec_result.output[1] else ""
            exit_code = exec_result.exit_code

            return DockerProcess(stdout=stdout, stderr=stderr, exit_code=exit_code)
        except Exception as e:
            return DockerProcess(stderr=f"Docker exec error: {e}", exit_code=1)

    def run_code(self, code: str) -> DockerExecution:
        script_name = f"temp_script_{os.urandom(4).hex()}.py"
        script_path = f"/app/{script_name}"

        try:
            self.files.write(script_path, code)

            proc = self._run_in_container(f"python3 {script_path}")

            self._run_in_container(f"rm {script_path}")

            results = []
            if "matplotlib" in code or "plt.show()" in code:
                results.append(DockerExecutionResult(png=b""))

            logs = DockerExecutionLogs(stdout=proc.stdout, stderr=proc.stderr)
            return DockerExecution(logs=logs, results=results)
        except Exception as e:
            return DockerExecution(logs=DockerExecutionLogs(stderr=f"Error executing code: {e}"), results=[])

    def stop(self) -> None:
        if self.container_id:
            try:
                container = self.docker_client.containers.get(self.container_id)
                container.stop()
                container.remove()
                print(f"Container '{self.container_name}' stopped and removed.")
            except docker.errors.NotFound:
                print(f"Container '{self.container_name}' not found, likely already removed.")
            self.container_id = None

    def __del__(self) -> None:
        self.stop()