"""`sync_npm_dependencies` reinstalls only when its inputs changed or node_modules is gone."""

from pathlib import Path
import shutil

import pytest

from backend.install_deps import sync_npm_dependencies
from tests.shell import fake_command


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path]:
    """A project directory and the log of the `npm` invocations a fake npm records."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "npm.log"
    fake_command(binaries, "npm", f'echo "$1" >>{log}\nmkdir -p node_modules\n')
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / "package.json").write_text("{}")
    (directory / "package-lock.json").write_text("{}")
    return directory, binaries / "npm"


def npm_calls(project_directory: Path) -> list[str]:
    log = project_directory.parent / "npm.log"
    return log.read_text().split() if log.exists() else []


def test_an_unchanged_install_is_kept(project):
    directory, npm = project
    sync_npm_dependencies(directory, str(npm))
    sync_npm_dependencies(directory, str(npm))
    assert npm_calls(directory) == ["ci"]


@pytest.mark.parametrize("changed", ["package.json", "package-lock.json", ".npmrc"])
def test_a_changed_input_reinstalls(project, changed):
    directory, npm = project
    sync_npm_dependencies(directory, str(npm))
    (directory / changed).write_text('{"changed": true}')
    sync_npm_dependencies(directory, str(npm))
    assert npm_calls(directory) == ["ci", "ci"]


def test_a_removed_node_modules_reinstalls(project):
    directory, npm = project
    sync_npm_dependencies(directory, str(npm))
    shutil.rmtree(directory / "node_modules")
    sync_npm_dependencies(directory, str(npm))
    assert npm_calls(directory) == ["ci", "ci"]
