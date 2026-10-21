"""Check consent and credential persistence without starting Docker services."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup-postgres.sh"


@pytest.fixture
def setup_environment(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == volume ]]; then\n'
        '    printf "%s\\n" "${SETUP_TEST_VOLUME:-}"\n'
        'fi\n'
    )
    docker.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("POSTGRES_PASSWORD_DEV", None)
    environment.pop("SETUP_TEST_VOLUME", None)
    environment.update(HOME=str(home), PATH=f"{binaries}:{environment['PATH']}")
    return home, environment


def run_setup(environment, answer):
    return subprocess.run(
        ["bash", str(SCRIPT)], input=answer, text=True, capture_output=True,
        env=environment, check=False,
    )


@pytest.mark.parametrize("answer", ["n\n", "\n", ""])
def test_declining_or_eof_leaves_secrets_untouched(setup_environment, answer):
    home, environment = setup_environment
    secrets_file = home / ".secrets"
    original = "export OTHER_CREDENTIAL='keep-me'\n"
    secrets_file.write_text(original)

    result = run_setup(environment, answer)

    assert result.returncode == 0, result.stderr
    assert secrets_file.read_text() == original


def test_confirming_generates_private_credentials_once(setup_environment):
    home, environment = setup_environment
    secrets_file = home / ".secrets"

    first = run_setup(environment, "yes\n")
    assert first.returncode == 0, first.stderr
    credentials = secrets_file.read_text()
    assert secrets_file.stat().st_mode & 0o777 == 0o600
    loaded = subprocess.run(
        ["bash", "-c", 'source "$HOME/.secrets" && printf %s "$POSTGRES_PASSWORD_DEV"'],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout
    assert len(bytes.fromhex(loaded)) == 32
    assert loaded not in first.stdout + first.stderr

    second = run_setup(environment, "")
    assert second.returncode == 0, second.stderr
    assert secrets_file.read_text() == credentials


def test_existing_volume_without_password_is_not_rekeyed(setup_environment):
    home, environment = setup_environment
    environment["SETUP_TEST_VOLUME"] = "gigachad-dev_gigachad-postgres"

    result = run_setup(environment, "y\n")

    assert result.returncode != 0
    assert not (home / ".secrets").exists()


def test_inherited_password_is_saved_without_prompting(setup_environment):
    _home, environment = setup_environment
    password = "quotes' spaces $and\\slashes"
    environment["POSTGRES_PASSWORD_DEV"] = password

    result = run_setup(environment, "")

    assert result.returncode == 0, result.stderr
    loaded = subprocess.run(
        ["bash", "-c", 'unset POSTGRES_PASSWORD_DEV; source "$HOME/.secrets"; printf %s "$POSTGRES_PASSWORD_DEV"'],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout
    assert loaded == password
    assert password not in result.stdout + result.stderr


def test_app_startup_stops_when_database_setup_is_declined(setup_environment):
    home, environment = setup_environment
    launcher = SCRIPT.parents[1] / "run.sh"

    result = subprocess.run(
        ["bash", str(launcher)], input="n\n", text=True, capture_output=True,
        env=environment, timeout=10, check=False,
    )

    assert result.returncode == 1
    assert "Startup cancelled" in result.stderr
    assert not (home / ".secrets").exists()


@pytest.mark.parametrize("credentials", ["environment", "secrets"])
def test_app_startup_reuses_database_credentials_without_setup(setup_environment, credentials):
    home, environment = setup_environment
    if credentials == "environment":
        environment["POSTGRES_PASSWORD_DEV"] = "existing-password"
    else:
        (home / ".secrets").write_text("export POSTGRES_PASSWORD_DEV=existing-password\n")

    # Stop at dependency installation, before any application service can launch.
    binaries = Path(environment["PATH"].split(os.pathsep)[0])
    python = binaries / "python3"
    python.write_text("#!/usr/bin/env bash\nexit 79\n")
    python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT.parents[1] / "run.sh")], input="", text=True,
        capture_output=True, env=environment, timeout=10, check=False,
    )

    assert result.returncode == 79
    assert "Setup cancelled" not in result.stdout
    assert "Create a local development database" not in result.stdout


def test_app_startup_creates_missing_database_credentials(setup_environment):
    home, environment = setup_environment
    binaries = Path(environment["PATH"].split(os.pathsep)[0])
    python = binaries / "python3"
    python.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == */src/backend/install_deps.py ]]; then\n'
        '    exit 79\n'
        'fi\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT.parents[1] / "run.sh")], input="y\n", text=True,
        capture_output=True, env=environment, timeout=10, check=False,
    )

    assert result.returncode == 79, result.stdout + result.stderr
    secrets_file = home / ".secrets"
    assert secrets_file.stat().st_mode & 0o777 == 0o600
    loaded = subprocess.run(
        ["bash", "-c", 'source "$HOME/.secrets"; printf %s "$POSTGRES_PASSWORD_DEV"'],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout
    assert len(bytes.fromhex(loaded)) == 32
    assert loaded not in result.stdout + result.stderr
