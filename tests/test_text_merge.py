import pytest

from lib import text_merge
from lib.text_merge import TextMergeError, merge_theirs


def test_clean_merge_keeps_both_sides():
    assert merge_theirs("a\nb\nc\n", "A\nb\nc\n", "a\nb\nC\n") == "A\nb\nC\n"


def test_conflict_takes_theirs_and_keeps_our_other_edits():
    assert merge_theirs("a\nb\nc\nd\ne\n", "a\nb\nc\nd\nE\n", "X\nb\nc\nd\ne\n") == "X\nb\nc\nd\nE\n"
    assert merge_theirs("a\nb\nc\n", "ours\nb\nc\n", "theirs\nb\nc\n") == "theirs\nb\nc\n"


def test_identical_sides_merge_to_that_text():
    assert merge_theirs("a\n", "b\n", "b\n") == "b\n"
    assert merge_theirs("a\n", "a\n", "a\n") == "a\n"


def test_empty_base_takes_theirs_on_conflict_and_the_only_change_otherwise():
    assert merge_theirs("", "ours\n", "theirs\n") == "theirs\n"
    assert merge_theirs("", "", "theirs\n") == "theirs\n"
    assert merge_theirs("", "ours\n", "") == "ours\n"


def test_a_final_newline_only_one_side_has_is_not_an_edit():
    # Base without a final newline (the buffer always has one): the user's edit next to the end survives an append.
    assert merge_theirs("a\nb\nc\nd", "a\nb\nC\nd\n", "a\nb\nc\nd\ne") == "a\nb\nC\nd\ne\n"
    # The agent dropped the final newline and changed nothing else.
    assert merge_theirs("a\nb\nc\nd\n", "a\nb\nC\nd\n", "a\nb\nc\nd") == "a\nb\nC\nd\n"


def test_git_refusing_the_merge_raises_and_removes_its_temporary_files(monkeypatch, tmp_path):
    monkeypatch.setattr(text_merge.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(TextMergeError, match="git merge-file failed"):
        merge_theirs("a\n", "a\x00\n", "b\n")  # NUL bytes: git refuses to merge binary files
    assert list(tmp_path.iterdir()) == []


def test_missing_git_raises_merge_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(text_merge.subprocess, "run", missing)
    with pytest.raises(TextMergeError):
        merge_theirs("a\n", "b\n", "c\n")
