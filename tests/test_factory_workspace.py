"""A factory task's workspace: `factory.yaml` parsing, worktrees made and removed through `agent-worktree`, the busy rule.

`agent-worktree` is a stub on PATH that only runs `git worktree add` and `remove` and logs its calls. One test runs the
real tool, where it is installed.
"""

import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from lib import factory_workspace
from lib.factory_workspace import (
    FactoryConfig,
    Workspace,
    WorkspaceBusy,
    WorkspaceError,
    create_workspace,
    ensure_checkout_free,
    load_factory_config,
    remove_workspace,
)
from tests.conftest import git

STUB = """#!/bin/sh
set -e
printf '%s|%s\\n' "$PWD" "$*" >> "$(dirname "$0")/calls"
primary="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
case "$1" in
    create)
        # create <name> --base <branch> --path <dir> --background
        git -C "$primary" worktree add --quiet -b "$2" "$6" "$4" >&2
        echo "$6" ;;
    remove)
        branch="$(git -C "$2" symbolic-ref --short HEAD)"
        git -C "$primary" worktree remove --force "$2" >&2
        git -C "$primary" branch --quiet -D "$branch" >&2 ;;
esac
"""


def _repository(tmp_path: Path) -> Path:
    repo = (tmp_path / "repo").resolve()
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    (repo / "readme.md").write_text("hi\n")
    git(repo, "add", "readme.md")
    git(repo, "commit", "-q", "-m", "readme")
    return repo


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository on `main`, its factory worktrees going to `tmp_path/worktrees`."""
    monkeypatch.setenv("GIGACHAD_FACTORY_WORKTREES", str(tmp_path / "worktrees"))
    return _repository(tmp_path)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """The stub `agent-worktree` first on PATH. `calls()` lists `<cwd>|<arguments>` of each run."""
    directory = tmp_path / "bin"
    directory.mkdir()
    script = directory / "agent-worktree"
    script.write_text(STUB)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}:{os.environ['PATH']}")
    log = directory / "calls"
    return SimpleNamespace(calls=lambda: log.read_text().splitlines() if log.exists() else [])


def _branches(repo: Path) -> list[str]:
    return git(repo, "branch", "--format=%(refname:short)").splitlines()


def test_missing_factory_yaml_is_an_empty_config(tmp_path):
    config = load_factory_config(tmp_path)
    assert config == FactoryConfig()
    assert (config.hooks.after_create, config.hooks.after_code, config.hooks.after_merge, config.gates) == ([], [], [], [])
    assert config.protected == ["factory.yaml", ".factory/**"]
    assert config.preview == {}


def test_factory_yaml_is_parsed_with_repo_path_filled_in_and_unknown_keys_ignored(repo):
    (repo / "factory.yaml").write_text(
        "profile: old\n"
        "hooks:\n"
        "  after_create: [[touch, created]]\n"
        "  after_merge: [[uv, sync, --directory, '{repo_path}']]\n"
        "  before_remove: [[gone]]\n"
        "gates: [['{repo_path}/.venv/bin/python', -m, pytest]]\n"
        "protected: [secrets/**]\n"
        "preview:\n"
        "  processes: [{name: backend, argv: ['{repo_path}/run', '{port.backend}']}]\n"
    )
    config = load_factory_config(repo)
    assert config.hooks.after_create == [["touch", "created"]]
    assert config.hooks.after_code == []
    assert config.hooks.after_merge == [["uv", "sync", "--directory", str(repo)]]
    assert config.gates == [[f"{repo}/.venv/bin/python", "-m", "pytest"]]
    assert config.protected == ["secrets/**"]
    assert config.preview == {"processes": [{"name": "backend", "argv": ["{repo_path}/run", "{port.backend}"]}]}
    assert not hasattr(config, "profile")
    assert not hasattr(config.hooks, "before_remove")


def test_repo_path_is_the_main_checkout_for_a_linked_worktree(repo, tmp_path):
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-q", "-b", "linked", str(linked))
    (linked / "factory.yaml").write_text("gates: [[ls, '{repo_path}']]\n")
    assert load_factory_config(linked).gates == [["ls", str(repo)]]


@pytest.mark.parametrize("text", ["hooks: {after_create: [[]]}\n", "gates: [ls]\n", "- a list\n", "gates: [unclosed\n"])
def test_invalid_factory_yaml_is_a_workspace_error(repo, text):
    (repo / "factory.yaml").write_text(text)
    with pytest.raises(WorkspaceError, match=r"factory\.yaml is invalid"):
        load_factory_config(repo)


def test_this_repos_factory_yaml_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_factory_config(root)
    assert config.hooks.after_create == []
    assert [argv[:2] for argv in config.hooks.after_merge] == [["uv", "sync"], ["npm", "install"], ["bun", "install"]]
    assert config.gates and all(argv[:2] == ["bash", "factory/gates.sh"] for argv in config.gates)
    assert config.protected == ["factory.yaml", ".factory/**"]
    assert [process["name"] for process in config.preview["processes"]] == ["backend", "frontend"]


def test_create_without_worktree_is_the_repo_itself_and_is_never_removed(repo, stub):
    workspace = create_workspace(repo, "task-1", worktree=False, base="main")
    assert workspace == Workspace(path=repo, branch=None, worktree=False)
    remove_workspace(workspace)
    assert (repo / "readme.md").exists()
    assert stub.calls() == []


def test_create_with_worktree_makes_branch_and_worktree_then_runs_after_create_in_it(repo, stub, tmp_path):
    (repo / "factory.yaml").write_text("hooks:\n  after_create:\n    - [sh, -c, 'pwd > hook-cwd; echo {repo_path} > hook-repo']\n")
    target = tmp_path / "worktrees" / "task-1"
    workspace = create_workspace(repo, "task-1", worktree=True, base="main")
    assert workspace == Workspace(path=target, branch="task-1", worktree=True)
    assert stub.calls() == [f"{repo}|create task-1 --base main --path {target} --background"]
    assert (target / "readme.md").read_text() == "hi\n"
    assert git(target, "symbolic-ref", "--short", "HEAD") == "task-1"
    assert (target / "hook-cwd").read_text().strip() == str(target)
    assert (target / "hook-repo").read_text().strip() == str(repo)


def test_worktree_starts_at_the_tip_of_its_base_branch_without_the_primarys_uncommitted_work(repo, stub, tmp_path):
    main_tip = git(repo, "rev-parse", "main")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "readme.md").write_text("side\n")
    git(repo, "commit", "-q", "-am", "side")
    (repo / "readme.md").write_text("dirty\n")
    (repo / "untracked.md").write_text("new\n")

    workspace = create_workspace(repo, "task-1", worktree=True, base="main")

    assert git(workspace.path, "rev-parse", "HEAD") == main_tip
    assert (workspace.path / "readme.md").read_text() == "hi\n"
    assert not (workspace.path / "untracked.md").exists()
    assert git(workspace.path, "status", "--porcelain") == ""


def test_worktree_root_defaults_to_the_repos_factory_folder_under_home(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repository(tmp_path)
    workspace = create_workspace(repo, "task-1", worktree=True, base="main")
    assert workspace.path == tmp_path / "home" / "git" / "worktrees" / "repo" / "factory" / "task-1"
    assert workspace.path.is_dir()


def test_a_taken_name_is_refused_and_the_existing_worktree_stays(repo, stub):
    first = create_workspace(repo, "task-1", worktree=True, base="main")
    with pytest.raises(WorkspaceError, match="agent-worktree create task-1"):
        create_workspace(repo, "task-1", worktree=True, base="main")
    assert first.path.is_dir()
    assert "task-1" in _branches(repo)


def test_failing_after_create_hook_removes_the_worktree_and_reports_command_and_output(repo, stub, tmp_path):
    (repo / "factory.yaml").write_text("hooks:\n  after_create:\n    - [sh, -c, 'echo boom-out; echo boom-err >&2; exit 3']\n")
    with pytest.raises(WorkspaceError) as raised:
        create_workspace(repo, "task-1", worktree=True, base="main")
    message = str(raised.value)
    assert message.startswith("sh -c 'echo boom-out; echo boom-err >&2; exit 3' failed with exit code 3\n")
    assert message.endswith("boom-out\nboom-err")
    assert not (tmp_path / "worktrees" / "task-1").exists()
    assert _branches(repo) == ["main"]


def test_after_create_hook_running_past_its_timeout_is_killed_and_the_worktree_removed(repo, stub, tmp_path, monkeypatch):
    monkeypatch.setattr(factory_workspace, "HOOK_TIMEOUT_SECONDS", 0.2)
    (repo / "factory.yaml").write_text("hooks:\n  after_create:\n    - [sh, -c, 'echo started; sleep 30 & wait']\n")
    with pytest.raises(WorkspaceError, match=r"timed out after 0\.2 s\nstarted"):
        create_workspace(repo, "task-1", worktree=True, base="main")
    assert not (tmp_path / "worktrees" / "task-1").exists()
    assert _branches(repo) == ["main"]


def test_hook_that_does_not_exist_is_a_workspace_error(repo, stub):
    (repo / "factory.yaml").write_text("hooks:\n  after_create: [[no-such-factory-command]]\n")
    with pytest.raises(WorkspaceError, match="no-such-factory-command did not start"):
        create_workspace(repo, "task-1", worktree=True, base="main")
    assert _branches(repo) == ["main"]


def test_remove_deletes_worktree_and_branch_with_uncommitted_work(repo, stub):
    workspace = create_workspace(repo, "task-1", worktree=True, base="main")
    (workspace.path / "draft.md").write_text("unsaved\n")
    remove_workspace(workspace)
    assert not workspace.path.exists()
    assert _branches(repo) == ["main"]
    assert stub.calls()[-1] == f"{repo}|remove {workspace.path} --force --delete-branch"


def test_remove_of_a_worktree_that_is_already_gone_is_not_an_error(repo, stub):
    workspace = create_workspace(repo, "task-1", worktree=True, base="main")
    remove_workspace(workspace)
    remove_workspace(workspace)
    assert len(stub.calls()) == 2


def test_ensure_checkout_free_refuses_a_checkout_with_an_active_task(repo, tmp_path):
    other = tmp_path / "worktrees" / "task-1"
    ensure_checkout_free(repo, [])
    ensure_checkout_free(repo, [other])
    with pytest.raises(WorkspaceBusy, match="already has a task"):
        ensure_checkout_free(repo, [other, repo / "."])


@pytest.mark.skipif(shutil.which("agent-worktree") is None, reason="agent-worktree is not installed")
def test_real_agent_worktree_creates_and_removes_a_workspace(repo, tmp_path):
    workspace = create_workspace(repo, "task-real", worktree=True, base="main")
    assert workspace == Workspace(path=(tmp_path / "worktrees" / "task-real").resolve(), branch="task-real", worktree=True)
    assert (workspace.path / "readme.md").exists()
    assert git(workspace.path, "symbolic-ref", "--short", "HEAD") == "task-real"
    (workspace.path / "draft.md").write_text("unsaved\n")
    remove_workspace(workspace)
    assert not workspace.path.exists()
    assert _branches(repo) == ["main"]
    assert str(workspace.path) not in git(repo, "worktree", "list")
