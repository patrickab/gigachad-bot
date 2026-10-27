"""The diff viewer lists a repository's changes and builds one patch per file."""

import pytest

from lib.git_diff import UnsupportedDiffContent, changed_files, file_contents, file_patch, line_counts
from tests.conftest import git


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "kept.txt").write_text("one\ntwo\n")
    (tmp_path / "gone.txt").write_text("bye\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path.resolve()


def test_changes_are_classified_and_sorted(repository):
    (repository / "kept.txt").write_text("one\nTWO\n")
    (repository / "gone.txt").unlink()
    (repository / "new.txt").write_text("hello")

    files = {file.path: file.status for file in changed_files(repository)}

    assert files == {"gone.txt": "deleted", "kept.txt": "modified", "new.txt": "untracked"}


def test_line_counts_cover_tracked_untracked_renamed_and_binary_files(repository):
    (repository / "kept.txt").write_text("one\nTWO\nthree\n")
    (repository / "gone.txt").unlink()
    (repository / "new.txt").write_text("a\nb")
    (repository / "blob.bin").write_bytes(b"\0\1")
    (repository / "old.txt").write_text("1\n2\n3\n4\n")
    git(repository, "add", "old.txt")
    git(repository, "commit", "-q", "-m", "old")
    git(repository, "mv", "old.txt", "moved.txt")

    counts = line_counts(repository, changed_files(repository))

    assert counts == {"kept.txt": (2, 1), "gone.txt": (0, 1), "new.txt": (2, 0), "moved.txt": (0, 0)}


def test_only_changes_below_the_inspected_directory_are_listed(repository):
    (repository / "sub").mkdir()
    (repository / "sub" / "inner.txt").write_text("x\n")
    (repository / "outer.txt").write_text("y\n")

    assert [file.path for file in changed_files(repository / "sub")] == ["inner.txt"]


def test_directory_outside_a_repository_has_no_file_list(tmp_path):
    assert changed_files(tmp_path.resolve()) is None


def test_patches_cover_modified_untracked_and_deleted_files(repository):
    (repository / "kept.txt").write_text("one\nTWO\n")
    (repository / "gone.txt").unlink()
    (repository / "new.txt").write_text("hello")

    assert "-two\n+TWO" in file_patch(repository, "kept.txt")[1]
    assert "-bye" in file_patch(repository, "gone.txt")[1]
    status, patch = file_patch(repository, "new.txt")
    assert status == "untracked"
    assert "+hello\n\\ No newline at end of file" in patch


def test_binary_and_unchanged_files_have_no_patch(repository):
    (repository / "image.bin").write_bytes(b"\0\1\2")

    assert file_patch(repository, "image.bin") is None
    assert file_patch(repository, "kept.txt") is None


def test_contents_pair_head_with_working_tree(repository):
    (repository / "kept.txt").write_text("one\nTWO\n")
    (repository / "new.txt").write_text("hello")

    assert file_contents(repository, "kept.txt") == ("one\ntwo\n", "one\nTWO\n")
    assert file_contents(repository, "new.txt") == ("", "hello")


def test_contents_follow_renames_to_the_original_head_file(repository):
    git(repository, "mv", "kept.txt", "moved.txt")
    (repository / "moved.txt").write_text("one\ntwo\nthree\n")

    assert file_contents(repository, "moved.txt") == ("one\ntwo\n", "one\ntwo\nthree\n")


def test_contents_of_unborn_repository_have_empty_base(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a\n")

    assert file_contents(tmp_path, "a.txt") == ("", "a\n")


def test_contents_reject_unsupported_files(repository, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.txt").write_text("s")
    (repository / "link.txt").symlink_to(outside / "secret.txt")
    (repository / "linkdir").symlink_to(outside, target_is_directory=True)
    (repository / "bin.dat").write_bytes(b"a\0b")
    (repository / "big.txt").write_text("x" * 1_000_001)
    (repository / "gone.txt").unlink()

    for path in ["../x", "/etc/passwd", "link.txt", "linkdir/secret.txt", "bin.dat", "big.txt", "gone.txt", "missing.txt", ""]:
        with pytest.raises(UnsupportedDiffContent):
            file_contents(repository, path)


def test_contents_reject_renamed_baseline_outside_mount(repository):
    mount = repository / "mount"
    mount.mkdir()
    git(repository, "mv", "kept.txt", "mount/kept.txt")

    with pytest.raises(UnsupportedDiffContent, match="Original path is outside the mount"):
        file_contents(mount, "kept.txt")
