"""Run the repository's shell scripts against fake commands on PATH."""

import os
from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[1]


def fake_command(binaries: Path, name: str, body: str) -> None:
    """Install an executable bash script *name* running *body* in *binaries*."""
    command = binaries / name
    command.write_text(f"#!/usr/bin/env bash\n{body}")
    command.chmod(0o755)


def environment_with(binaries: Path, **variables: str) -> dict[str, str]:
    """This process's environment with *binaries* first on PATH and *variables* set."""
    environment = os.environ.copy()
    environment.update(variables, PATH=f"{binaries}:{environment['PATH']}")
    return environment


def run_script(
    script: Path, environment: dict[str, str], *, input: str | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run *script* with bash, capturing its output without raising on failure."""
    return subprocess.run(
        ["bash", str(script)], input=input, text=True, capture_output=True, env=environment, timeout=timeout, check=False
    )


def bash_output(command: str, environment: dict[str, str]) -> str:
    """Stdout of a bash *command* that must succeed."""
    return subprocess.run(["bash", "-c", command], env=environment, text=True, capture_output=True, check=True).stdout
