import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Pinned so a fresh venv gets the same Node as every other checkout.
NODE_VERSION = "26.10.0"


def _fail_unless_repairable(result: subprocess.CompletedProcess[str], is_lock_failure: bool) -> None:
    """Print a failed command's output, then raise unless this is a stale lock outside production."""
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    if os.environ.get("GIGACHAD_ENV") == "production" or not is_lock_failure:
        result.check_returncode()


def sync_python_lock(project_dir: Path) -> None:
    """Refresh stale development locks without upgrading unchanged dependencies."""
    result = subprocess.run(["uv", "lock", "--check"], cwd=project_dir, capture_output=True, text=True)
    if result.returncode == 0:
        return
    _fail_unless_repairable(
        result,
        "error: The lockfile at `uv.lock` needs to be updated, but `--check` was provided." in result.stderr
        or "error: Unable to find lockfile at `uv.lock`, but `--check` was provided." in result.stderr,
    )
    print(f"Refreshing development lock: {project_dir / 'uv.lock'}. Review and commit the changes.", flush=True)
    subprocess.run(["uv", "lock"], cwd=project_dir, check=True)
    subprocess.run(["uv", "lock", "--check"], cwd=project_dir, check=True)


def sync_npm_dependencies(project_dir: Path, npm_path: str = "npm") -> None:
    """Repair only missing or stale development locks, then retry a strict install once."""
    install_options = ["--ignore-scripts", "--prefer-offline", "--no-audit", "--no-fund"]
    result = subprocess.run(
        [npm_path, "ci", "--json", *install_options], cwd=project_dir, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Installed locked npm dependencies: {project_dir}", flush=True)
        return
    try:
        error = json.loads(result.stdout).get("error", {})
    except json.JSONDecodeError:
        error = {}
    _fail_unless_repairable(result, error.get("code") == "EUSAGE" and error.get("summary", "").startswith((
        "`npm ci` can only install packages when your package.json and package-lock.json or npm-shrinkwrap.json are in sync.",
        "The `npm ci` command can only install with an existing package-lock.json or",
    )))
    print(f"Refreshing development npm lock in {project_dir}. Review and commit the changes.", flush=True)
    subprocess.run([npm_path, "install", "--package-lock-only", *install_options], cwd=project_dir, check=True)
    subprocess.run([npm_path, "ci", *install_options], cwd=project_dir, check=True)


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent.parent
    venv_dir = root_dir / ".venv"
    frontend_dir = root_dir / "src" / "frontend"

    # `nodeenv -p` installs Node.js into the active venv.
    if not (venv_dir / "bin" / "node").exists():
        print(f"Installing Node.js {NODE_VERSION} into the Python virtual environment...")
        subprocess.run(["nodeenv", "-p", f"--node={NODE_VERSION}"], check=True)

    print("Installing frontend dependencies...")
    npm_path = venv_dir / "bin" / "npm"

    if not npm_path.exists():
        print("Error: npm not found in virtual environment.")
        return

    sync_npm_dependencies(frontend_dir, str(npm_path))

    print("All dependencies installed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install locked dependencies with development-only lock recovery.")
    parser.add_argument("operation", choices=["frontend", "python-lock", "npm"], nargs="?", default="frontend")
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parents[2])
    arguments = parser.parse_args()
    if arguments.operation == "python-lock":
        sync_python_lock(arguments.directory)
    elif arguments.operation == "npm":
        sync_npm_dependencies(arguments.directory)
    else:
        main()
