from lib.text_merge import merge_theirs


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
