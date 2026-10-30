"""Exercise checkout bootstrap and overlay application using local Git repositories, without installs or servers."""

from pathlib import Path
import re
import shutil

import pytest

from tests.conftest import git
from tests.shell import REPOSITORY, environment_with, fake_command, run_script

REF = "integration_patrick_ab"
# npm leaves the stamps the launcher's install and build checks read, node packs an empty chat-ui and serves nothing.
# The installed tsc is the auto-update's overlay check: it fails in a checkout holding `break-check`.
FAKE_NPM = (
    'mkdir -p node_modules\nif [[ $1 == run ]]; then mkdir -p .next && touch .next/BUILD_ID; fi\n'
    'if [[ $1 == ci ]]; then mkdir -p node_modules/.bin && printf \'#!/bin/sh\\n! test -e break-check\\n\' >node_modules/.bin/tsc'
    ' && chmod +x node_modules/.bin/tsc; fi\n'
)
FAKE_NODE = 'if [[ $1 == packages/chat-ui/pack.mjs ]]; then mkdir -p "$3" && : >"$3/chat-ui.tgz" && echo "$3/chat-ui.tgz"; fi\n'
IDENTITY = {f"GIT_{role}_{field}": value for role in ("AUTHOR", "COMMITTER") for field, value in (("NAME", "t"), ("EMAIL", "t@t"))}


def seams_preimages(patch: str) -> dict[str, str]:
    """Upstream files the seams patch applies to: each hunk's context and removed lines at its line, filler elsewhere."""
    files: dict[str, list[str]] = {}
    lines: list[str] = []
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("--- "):
            in_hunk = False
        elif line.startswith("+++ "):
            lines = files.setdefault(line.split(" ", 1)[1].split("/", 1)[1], [])
        elif match := re.match(r"@@ -(\d+)", line):
            lines.extend("// upstream" for _ in range(int(match[1]) - 1 - len(lines)))
            in_hunk = True
        elif in_hunk and line[:1] in (" ", "-", ""):
            lines.append(line[1:])
    return {path: "\n".join(lines) + "\n" for path, lines in files.items()}


def launcher_root(tmp_path: Path) -> Path:
    """A copy of the launcher's inputs: the repository's .venv/bin would otherwise shadow the fake node."""
    root = tmp_path / "gigachad-bot"
    (root / "deploy").mkdir(parents=True)
    (root / "src" / "frontend").mkdir(parents=True)
    shutil.copy(REPOSITORY / "run-omp-web.sh", root)
    shutil.copy(REPOSITORY / "deploy" / "lib.sh", root / "deploy")
    shutil.copytree(REPOSITORY / "ompweb-overlay", root / "ompweb-overlay")
    return root


def fake_binaries(tmp_path: Path) -> Path:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_command(binaries, "omp", "exit 0\n")
    fake_command(binaries, "npm", FAKE_NPM)
    fake_command(binaries, "node", FAKE_NODE)
    return binaries


def commit(repository: Path, path: str, content: str, message: str) -> None:
    (repository / path).write_text(content)
    git(repository, "add", path)
    git(repository, "commit", "-m", message)


def test_bootstrap_applies_overlay_preserves_checkout_and_rejects_unrelated_directory(tmp_path):
    root = launcher_root(tmp_path)
    source = tmp_path / "source"
    git(tmp_path, "init", f"--initial-branch={REF}", str(source))
    for path, content in seams_preimages((root / "ompweb-overlay" / "seams.patch").read_text()).items():
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_text(content)
    (source / "package-lock.json").write_text("{}\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "Fixture")

    binaries = fake_binaries(tmp_path)
    checkout = tmp_path / "nested" / "checkout"
    worktree = tmp_path / "worktree"
    environment = environment_with(
        binaries,
        GIGACHAD_OMP_WEB_REPO=str(checkout),
        GIGACHAD_OMP_WEB_DIR=str(worktree),
        GIGACHAD_OMP_WEB_REF=REF,
        GIGACHAD_OMP_WEB_BIND="127.0.0.1:0",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{source.as_uri()}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/kahme247/ompweb.git",
    )

    def launch():
        return run_script(root / "run-omp-web.sh", environment, timeout=15)

    initial = launch()
    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert git(checkout, "branch", "--show-current") == REF
    assert git(worktree, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")
    assert "CustomRoot" in (worktree / "app" / "layout.tsx").read_text()
    assert "withCustom(nextConfig)" in (worktree / "next.config.ts").read_text()
    assert (worktree / "custom" / "root.tsx").read_bytes() == (root / "ompweb-overlay" / "root.tsx").read_bytes()

    local_file = checkout / "local-file"
    local_file.write_text("preserve local changes")
    repeated = launch()
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert local_file.read_text() == "preserve local changes"
    assert "CustomRoot" in (worktree / "app" / "layout.tsx").read_text()

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("untouched")
    environment["GIGACHAD_OMP_WEB_REPO"] = str(unrelated)
    environment["GIGACHAD_OMP_WEB_DIR"] = str(tmp_path / "unused-worktree")
    rejected = launch()
    assert rejected.returncode != 0
    assert (unrelated / "keep").read_text() == "untouched"
    assert not (tmp_path / "unused-worktree").exists()


def advance_upstream(upstream: Path, branch_worktree: Path) -> None:
    commit(upstream, "upstream.txt", "new upstream\n", "Upstream change")


def conflict_upstream(upstream: Path, branch_worktree: Path) -> None:
    commit(upstream, "ours.txt", "upstream's version\n", "Upstream adds ours.txt")


def break_check_upstream(upstream: Path, branch_worktree: Path) -> None:
    commit(upstream, "break-check", "tsc fails\n", "Upstream breaks the overlay")


def dirty_branch_worktree(upstream: Path, branch_worktree: Path) -> None:
    advance_upstream(upstream, branch_worktree)
    (branch_worktree / "ours.txt").write_text("uncommitted edit\n")


def offline_upstream(upstream: Path, branch_worktree: Path) -> None:
    advance_upstream(upstream, branch_worktree)
    upstream.rename(upstream.with_name("upstream-unreachable"))


@pytest.mark.parametrize(
    ("change", "updated", "message"),
    [
        (None, False, None),
        (advance_upstream, True, "omp-web auto-update: rebased"),
        (conflict_upstream, False, "conflicts"),
        (break_check_upstream, False, "the overlay checks fail"),
        (dirty_branch_worktree, False, "has uncommitted changes"),
        (offline_upstream, False, "fetching origin failed"),
    ],
    ids=["up-to-date", "clean-rebase", "conflict", "failed-check", "dirty-worktree", "offline"],
)
def test_auto_update_rebases_branch_onto_upstream_main_or_keeps_it(tmp_path, change, updated, message):
    root = launcher_root(tmp_path)
    upstream = tmp_path / "upstream"
    git(tmp_path, "init", "--initial-branch=main", str(upstream))
    for path, content in seams_preimages((root / "ompweb-overlay" / "seams.patch").read_text()).items():
        (upstream / path).parent.mkdir(parents=True, exist_ok=True)
        (upstream / path).write_text(content)
    (upstream / "package-lock.json").write_text("{}\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "Upstream")

    # The user's clone: main checked out, the integration branch (upstream main + a local commit) in its own worktree.
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", "--quiet", str(upstream), str(checkout))
    branch_worktree = tmp_path / "branch-worktree"
    git(checkout, "worktree", "add", "--quiet", "-b", REF, str(branch_worktree), "origin/main")
    commit(branch_worktree, "ours.txt", "our pending upstream PR\n", "Our change")
    before = git(checkout, "rev-parse", REF)
    if change:
        change(upstream, branch_worktree)

    served = tmp_path / "served"
    environment = environment_with(
        fake_binaries(tmp_path),
        GIGACHAD_OMP_WEB_REPO=str(checkout),
        GIGACHAD_OMP_WEB_DIR=str(served),
        GIGACHAD_OMP_WEB_REF=REF,
        GIGACHAD_OMP_WEB_BIND="127.0.0.1:0",
        **IDENTITY,
    )
    environment.pop("GIGACHAD_AUTO_UPDATE", None)
    result = run_script(root / "run-omp-web.sh", environment, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr

    after = git(checkout, "rev-parse", REF)
    assert git(served, "rev-parse", "HEAD") == after
    assert git(branch_worktree, "rev-parse", "HEAD") == after
    if updated:
        assert after != before
        assert git(checkout, "rev-parse", f"{REF}~1") == git(checkout, "rev-parse", "origin/main")
        assert (branch_worktree / "upstream.txt").read_text() == "new upstream\n"
        assert (branch_worktree / "ours.txt").read_text() == "our pending upstream PR\n"
    else:
        assert after == before
    if message:
        assert message in output
    else:
        assert "auto-update" not in output
    if change is dirty_branch_worktree:
        assert (branch_worktree / "ours.txt").read_text() == "uncommitted edit\n"
    # The temporary worktree is gone, from disk and from Git's list.
    assert not list(tmp_path.glob("served.update.*"))
    assert len(git(checkout, "worktree", "list", "--porcelain").split("\n\n")) == 3

    # A failed rebase or check is remembered: the next start skips it with the same reason.
    if change in (conflict_upstream, break_check_upstream):
        repeated = run_script(root / "run-omp-web.sh", environment, timeout=30)
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        assert "(as on the last start)" in repeated.stderr
        assert "rebasing" not in repeated.stdout
        assert git(checkout, "rev-parse", REF) == before
