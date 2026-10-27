"""Worktrees branch off a mounted clone, update their base first, and only lose work on explicit force."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.identity import get_request_identity
from backend.routes import file_git as file_git_route
from backend.routes import file_roots as file_roots_route
from lib import git_worktree
from lib.file_roots import FileRoots
from lib.git import GitError
from tests.conftest import git, mounted_roots, request_identity, route_client


def _commit(directory, name, content):
    (Path(directory) / name).write_text(content)
    git(directory, "add", name)
    git(directory, "commit", "-q", "-m", name)


@pytest.fixture
def clone(tmp_path, monkeypatch):
    """A clone of `origin` on `dev`, worktrees going to `tmp_path/worktrees`."""
    monkeypatch.setattr(git_worktree, "WORKTREES_HOME", tmp_path / "worktrees")
    upstream = tmp_path / "upstream"
    git(tmp_path, "init", "-q", "-b", "main", str(upstream))
    _commit(upstream, "readme.md", "hi\n")
    git(upstream, "branch", "dev")
    git(tmp_path, "clone", "-q", "-b", "dev", str(upstream), str(tmp_path / "repo"))
    return SimpleNamespace(upstream=upstream, repository=(tmp_path / "repo").resolve(), home=tmp_path / "worktrees")


def test_default_branch_falls_back_to_main_checkout_not_an_arbitrary_branch(clone):
    git(clone.repository, "symbolic-ref", "--delete", "refs/remotes/origin/HEAD")
    assert git_worktree.default_branch(clone.repository) == "dev"
    git(clone.repository, "checkout", "--detach")
    assert git_worktree.default_branch(clone.repository) is None
    with pytest.raises(git_worktree.WorktreeError, match="Cannot infer"):
        git_worktree.default_worktree(clone.repository)


@pytest.mark.parametrize("local_branch", [False, True])
def test_default_checkout_creates_and_reuses_without_pulling_or_changing_main(clone, local_branch):
    assert git_worktree.default_branch(clone.repository) == "main"
    assert git_worktree.default_base_branch(clone.repository) == "dev"
    if local_branch:
        git(clone.repository, "branch", "main", "origin/main")
    before = git(clone.repository, "rev-parse", "HEAD")
    (clone.repository / "readme.md").write_text("unsaved changes\n")
    (clone.repository / "draft.md").write_text("untracked\n")
    _commit(clone.upstream, "later.md", "not fetched\n")
    git(clone.upstream, "branch", "-f", "dev", "main")

    worktree = git_worktree.default_worktree(clone.repository)

    assert worktree.branch == "main"
    assert git_worktree.default_worktree(clone.repository) == worktree
    assert git(worktree.path, "rev-parse", "HEAD") == before
    assert not (Path(worktree.path) / "later.md").exists()
    assert git(clone.repository, "symbolic-ref", "--short", "HEAD") == "dev"
    assert git(clone.repository, "rev-parse", "HEAD") == before
    assert (clone.repository / "readme.md").read_text() == "unsaved changes\n"
    assert (clone.repository / "draft.md").read_text() == "untracked\n"
    if not local_branch:
        assert git(clone.repository, "rev-parse", "--abbrev-ref", "main@{upstream}") == "origin/main"


def test_default_checkout_reuses_an_existing_checkout_outside_standard_home(clone, tmp_path):
    path = tmp_path / "elsewhere"
    git(clone.repository, "worktree", "add", "-b", "main", str(path), "origin/main")
    (path / "draft.md").write_text("keep me\n")

    assert git_worktree.default_worktree(clone.repository) == git_worktree.Worktree(str(path.resolve()), "main", False)
    assert (path / "draft.md").read_text() == "keep me\n"
    assert not clone.home.exists()


def test_default_checkout_reuses_the_main_checkout_when_it_has_the_default_branch(clone):
    git(clone.repository, "checkout", "-b", "main", "origin/main")

    assert git_worktree.default_worktree(clone.repository) == git_worktree.Worktree(str(clone.repository), "main", True)
    assert not clone.home.exists()


@pytest.mark.parametrize("collision", ["directory", "broken_symlink", "worktree"])
def test_default_checkout_refuses_existing_destination_without_creating_a_branch(clone, tmp_path, collision):
    path = clone.home / "repo" / "main"
    path.parent.mkdir(parents=True)
    if collision == "broken_symlink":
        path.symlink_to(tmp_path / "missing")
    elif collision == "worktree":
        git(clone.repository, "worktree", "add", "-b", "other", str(path), "dev")
    else:
        path.mkdir()
        (path / "keep.md").write_text("keep me\n")

    with pytest.raises(git_worktree.WorktreeError, match="Directory already exists"):
        git_worktree.default_worktree(clone.repository)

    assert git(clone.repository, "branch", "--list", "main") == ""
    assert git(clone.repository, "symbolic-ref", "--short", "HEAD") == "dev"
    if collision == "directory":
        assert (path / "keep.md").read_text() == "keep me\n"


def test_a_new_worktree_branches_off_the_freshly_pulled_base(clone):
    _commit(clone.upstream, "later.md", "pushed after cloning\n")
    git(clone.upstream, "branch", "-f", "dev", "main")

    worktree = git_worktree.create_worktree(clone.repository, "42 Fix Usage!", "dev")

    assert (worktree.branch, Path(worktree.path)) == ("42-fix-usage", (clone.home / "repo" / "42-fix-usage").resolve())
    assert (Path(worktree.path) / "later.md").is_file()
    assert (clone.repository / "later.md").is_file(), "the clone on dev is fast-forwarded too"
    assert [entry.branch for entry in git_worktree.list_worktrees(clone.repository)] == ["dev", "42-fix-usage"]


def test_a_base_not_checked_out_anywhere_is_fetched_from_origin(clone):
    worktree = git_worktree.create_worktree(clone.repository, "feature", "main")

    assert git(worktree.path, "rev-parse", "HEAD") == git(clone.upstream, "rev-parse", "main")


def test_an_existing_branch_or_folder_is_refused_not_reused(clone):
    git_worktree.create_worktree(clone.repository, "feature", "dev")
    git(clone.repository, "branch", "taken")

    with pytest.raises(git_worktree.WorktreeError, match="Branch already exists"):
        git_worktree.create_worktree(clone.repository, "taken", "dev")
    with pytest.raises(git_worktree.WorktreeError, match="already exists"):
        git_worktree.create_worktree(clone.repository, "feature", "dev")


def test_status_counts_uncommitted_files_and_commits_against_the_base(clone):
    worktree = git_worktree.create_worktree(clone.repository, "feature", "dev")
    _commit(worktree.path, "one.md", "1\n")
    _commit(worktree.path, "two.md", "2\n")
    (Path(worktree.path) / "draft.md").write_text("wip\n")
    _commit(clone.repository, "base.md", "moved on\n")

    assert git_worktree.status(Path(worktree.path), "dev") == git_worktree.WorktreeStatus(changed_files=1, ahead=2, behind=1)


def test_removal_needs_force_for_uncommitted_work_and_keeps_the_branch(clone):
    worktree = git_worktree.create_worktree(clone.repository, "feature", "dev")
    (Path(worktree.path) / "draft.md").write_text("wip\n")

    with pytest.raises(git_worktree.WorktreeDirty):
        git_worktree.remove_worktree(clone.repository, worktree.path)
    git_worktree.remove_worktree(clone.repository, worktree.path, force=True)

    assert not Path(worktree.path).exists()
    assert "feature" in git_worktree.branches(clone.repository)
    with pytest.raises(git_worktree.WorktreeError, match="cannot be removed"):
        git_worktree.remove_worktree(clone.repository, str(clone.repository))


def test_deleting_an_unmerged_branch_needs_force(clone):
    worktree = git_worktree.create_worktree(clone.repository, "feature", "dev")
    _commit(worktree.path, "work.md", "done\n")
    git_worktree.remove_worktree(clone.repository, worktree.path)

    with pytest.raises(git_worktree.BranchNotMerged):
        git_worktree.delete_branch(clone.repository, "feature")
    git_worktree.delete_branch(clone.repository, "feature", force=True)

    assert "feature" not in git_worktree.branches(clone.repository)


def test_a_mounted_clone_admits_its_worktrees_and_shows_the_active_one(clone, tmp_path):
    worktree = git_worktree.create_worktree(clone.repository, "feature", "dev")
    _commit(worktree.path, "only-here.md", "x\n")
    roots = mounted_roots(clone.repository)

    [main] = roots.tree()
    [switched] = roots.tree([worktree.path])

    assert (main["path"], main["mount"]) == (str(clone.repository), str(clone.repository))
    assert (switched["path"], switched["mount"]) == (worktree.path, str(clone.repository))
    assert [child["name"] for child in switched["children"]] == ["only-here.md", "readme.md"]
    assert roots.read(str(Path(worktree.path) / "only-here.md")) == "x\n"
    assert roots.root_of(str(Path(worktree.path) / "only-here.md")) == Path(worktree.path)
    assert not roots.contains(tmp_path / "upstream" / "readme.md")


def test_worktree_routes_refuse_unmounted_roots_and_report_conflicts(clone):
    client = route_client(file_git_route.router, overrides={file_git_route.get_file_roots: mounted_roots(clone.repository)})

    created = client.post("/api/fileroots/worktrees", json={"root": str(clone.repository), "name": "feature", "base": "dev"}).json()
    (Path(created["path"]) / "draft.md").write_text("wip\n")
    listing = client.get("/api/fileroots/worktrees", params={"root": str(clone.repository)}).json()
    dirty = client.delete("/api/fileroots/worktrees", params={"root": str(clone.repository), "path": created["path"]})
    unmounted = client.get("/api/fileroots/worktrees", params={"root": str(clone.upstream)})

    assert (listing["default_base"], [entry["branch"] for entry in listing["worktrees"]]) == ("dev", ["dev", "feature"])
    assert listing["default_branch"] == "main"
    assert listing["worktrees"][1]["changed_files"] == 1
    assert dirty.status_code == 409
    assert unmounted.status_code == 403


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """Clones go to `tmp_path/git`, git rewrites the `owner/repo` links below to a local upstream."""
    monkeypatch.setattr(git_worktree, "CLONES_HOME", tmp_path / "git")
    upstream = tmp_path / "upstream"
    git(tmp_path, "init", "-q", "-b", "main", str(upstream))
    _commit(upstream, "readme.md", "hi\n")
    rewrites = {"https://example.com/owner/repo.git": upstream, "git@example.com:owner/repo.git": upstream, "ssh://git@example.com/owner/repo": upstream}
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(rewrites)))
    for index, (link, target) in enumerate(rewrites.items()):
        monkeypatch.setenv(f"GIT_CONFIG_KEY_{index}", f"url.{target}.insteadOf")
        monkeypatch.setenv(f"GIT_CONFIG_VALUE_{index}", link)
    return SimpleNamespace(links=list(rewrites), home=tmp_path / "git")


@pytest.mark.parametrize("link_index", [0, 1, 2])
def test_clone_accepts_https_and_both_ssh_spellings_into_the_repository_folder(remote, link_index):
    cloned = git_worktree.clone_repository(f"  {remote.links[link_index]}\n")

    assert cloned == (remote.home / "repo").resolve()
    assert (cloned / "readme.md").read_text() == "hi\n"
    assert git_worktree.is_clone(cloned)


@pytest.mark.parametrize("link", [
    "", "repo", "/tmp/repo", "file:///tmp/repo", "http://example.com/owner/repo", "ext::sh -c touch% /tmp/pwned",
    "--upload-pack=touch /tmp/pwned", "-oProxyCommand=x:owner/repo", "https://example.com/owner/..", "https://example.com/owner/re po",
])
def test_clone_refuses_links_that_are_not_https_or_ssh_repositories(remote, link):
    with pytest.raises(git_worktree.WorktreeError, match="Not an https or ssh"):
        git_worktree.clone_repository(link)
    assert not remote.home.exists()


def test_clone_refuses_existing_folder_and_leaves_no_folder_after_failure(remote, tmp_path):
    (remote.home / "taken").mkdir(parents=True)
    with pytest.raises(git_worktree.WorktreeError, match="already exists"):
        git_worktree.clone_repository("https://example.com/owner/taken.git")

    with pytest.raises(GitError):
        git_worktree.clone_repository("https://127.0.0.1:1/owner/unreachable.git")
    assert not (remote.home / "unreachable").exists()


def test_clone_route_mounts_the_clone_and_reports_refusals(remote):
    saved = []
    roots = FileRoots(repository=SimpleNamespace(load=lambda: [], save=saved.append))
    client = route_client(
        file_roots_route.router, overrides={file_roots_route.get_file_roots: roots, get_request_identity: request_identity()}
    )

    cloned = client.post("/api/fileroots/clone", json={"url": remote.links[1]})
    again = client.post("/api/fileroots/clone", json={"url": remote.links[1]})
    invalid = client.post("/api/fileroots/clone", json={"url": "nonsense"})

    assert cloned.status_code == 200
    assert saved == [[str((remote.home / "repo").resolve())]]
    assert [node["name"] for node in cloned.json()["tree"]] == ["repo"]
    assert (again.status_code, invalid.status_code) == (400, 400)
