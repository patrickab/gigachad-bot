"""Exercise checkout bootstrap using local Git repositories, without servers."""

from tests.conftest import git
from tests.shell import REPOSITORY, environment_with, fake_command, run_script

LAUNCHER = REPOSITORY / "run-omp-web.sh"


def test_bootstrap_preserves_existing_checkout_and_rejects_unrelated_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    git(tmp_path, "init", "--initial-branch=custom", str(source))
    # Replace the external app with an entrypoint that starts no services.
    fake_command(source, "run.sh", "exit 0\n")
    git(source, "add", "run.sh")
    git(source, "commit", "-m", "Fixture")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_command(binaries, "omp", "exit 1\n")
    checkout = tmp_path / "nested" / "checkout"
    worktree = tmp_path / "worktree"
    environment = environment_with(
        binaries,
        GIGACHAD_OMP_WEB_REPO=str(checkout),
        GIGACHAD_OMP_WEB_DIR=str(worktree),
        GIGACHAD_OMP_WEB_REF="custom",
        GIGACHAD_OMP_WEB_BIND="127.0.0.1:0",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{source.as_uri()}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/patrickab/ompweb.git",
    )

    def launch():
        return run_script(LAUNCHER, environment, timeout=15)

    initial = launch()
    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert git(checkout, "branch", "--show-current") == "custom"
    assert git(worktree, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")

    local_file = checkout / "local-file"
    local_file.write_text("preserve local changes")
    repeated = launch()
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert local_file.read_text() == "preserve local changes"

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("untouched")
    environment["GIGACHAD_OMP_WEB_REPO"] = str(unrelated)
    environment["GIGACHAD_OMP_WEB_DIR"] = str(tmp_path / "unused-worktree")
    rejected = launch()
    assert rejected.returncode != 0
    assert (unrelated / "keep").read_text() == "untouched"
    assert not (tmp_path / "unused-worktree").exists()
