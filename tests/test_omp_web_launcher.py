"""Exercise checkout bootstrap and overlay application using local Git repositories, without installs or servers."""

import re
import shutil

from tests.conftest import git
from tests.shell import REPOSITORY, environment_with, fake_command, run_script

REF = "integration_patrick_ab"
# npm leaves the stamps the launcher's install and build checks read, node packs an empty chat-ui and serves nothing.
FAKE_NPM = 'mkdir -p node_modules\nif [[ $1 == run ]]; then mkdir -p .next && touch .next/BUILD_ID; fi\n'
FAKE_NODE = 'if [[ $1 == packages/chat-ui/pack.mjs ]]; then mkdir -p "$3" && : >"$3/chat-ui.tgz" && echo "$3/chat-ui.tgz"; fi\n'


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


def test_bootstrap_applies_overlay_preserves_checkout_and_rejects_unrelated_directory(tmp_path):
    # A copy of the launcher's inputs: the repository's .venv/bin would otherwise shadow the fake node.
    root = tmp_path / "gigachad-bot"
    (root / "deploy").mkdir(parents=True)
    (root / "src" / "frontend").mkdir(parents=True)
    shutil.copy(REPOSITORY / "run-omp-web.sh", root)
    shutil.copy(REPOSITORY / "deploy" / "lib.sh", root / "deploy")
    shutil.copytree(REPOSITORY / "ompweb-overlay", root / "ompweb-overlay")

    source = tmp_path / "source"
    git(tmp_path, "init", f"--initial-branch={REF}", str(source))
    for path, content in seams_preimages((root / "ompweb-overlay" / "seams.patch").read_text()).items():
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_text(content)
    (source / "package-lock.json").write_text("{}\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "Fixture")

    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_command(binaries, "omp", "exit 0\n")
    fake_command(binaries, "npm", FAKE_NPM)
    fake_command(binaries, "node", FAKE_NODE)
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
