import os
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    venv_dir = root_dir / ".venv"

    # Step 1: Ensure Node.js is installed in the virtual environment
    if not (venv_dir / "bin" / "node").exists():
        print("Installing Node.js into the Python virtual environment...")
        subprocess.run(["nodeenv", "-p"], check=True)

    # Step 2: Install frontend and C4 service dependencies with the venv's npm
    npm_path = venv_dir / "bin" / "npm"
    if not npm_path.exists():
        print("Error: npm not found in virtual environment.")
        return

    for name in ("frontend", "c4"):
        print(f"Installing {name} dependencies...")
        subprocess.run([str(npm_path), "install"], cwd=root_dir / "src" / name, check=True)

    print("All dependencies installed successfully!")


if __name__ == "__main__":
    main()
