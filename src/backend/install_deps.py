import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    venv_dir = root_dir / ".venv"
    frontend_dir = root_dir / "src" / "frontend"

    # `nodeenv -p` installs Node.js into the active venv.
    if not (venv_dir / "bin" / "node").exists():
        print("Installing Node.js into the Python virtual environment...")
        subprocess.run(["nodeenv", "-p"], check=True)

    print("Installing frontend dependencies...")
    npm_path = venv_dir / "bin" / "npm"

    if not npm_path.exists():
        print("Error: npm not found in virtual environment.")
        return

    subprocess.run([str(npm_path), "install"], cwd=frontend_dir, check=True)

    print("All dependencies installed successfully!")


if __name__ == "__main__":
    main()
