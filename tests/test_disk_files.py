import pytest

from lib.disk_files import DiskFiles
from lib.naming import segment
from tests.conftest import mounted_roots
from tests.fakes import RecordingScope


@pytest.fixture
def vault(tmp_path):
    directory = tmp_path / "vault"
    directory.mkdir()
    return directory


def test_creating_is_exclusive_and_records_one_change_keyed_by_the_resolved_path(vault):
    scope = RecordingScope()
    files = DiskFiles(mounted_roots(vault), scope)

    created = files.create(vault / "sub" / ".." / "a.tex", "one")
    with pytest.raises(FileExistsError):
        files.create(vault / "a.tex", "two")
    image = files.create(vault / "a.png", b"\x89PNG")

    assert created == vault.resolve() / "a.tex"
    assert created.read_text() == "one"
    assert image.read_bytes() == b"\x89PNG"
    assert scope.changes == [
        {"kind": "document", "key": str(created), "operation": "write"},
        {"kind": "document", "key": str(image), "operation": "write"},
    ]


def test_writing_overwrites_and_records_one_change(vault):
    note = vault / "note.md"
    note.write_text("v1")
    scope = RecordingScope()

    written = DiskFiles(mounted_roots(vault), scope).write(str(note), "v2")

    assert note.read_text() == "v2"
    assert scope.changes == [{"kind": "document", "key": str(written), "operation": "write"}]


def test_creating_outside_the_roots_raises_and_leaves_no_file_or_change(tmp_path, vault):
    scope = RecordingScope()

    with pytest.raises(ValueError, match="outside file roots"):
        DiskFiles(mounted_roots(vault), scope).create(tmp_path / "outside.tex", "x")

    assert not (tmp_path / "outside.tex").exists()
    assert scope.changes == []


def test_the_ceiling_is_the_mounted_root_or_the_own_folder_of_a_host_only_path(tmp_path, vault):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "main.tex").write_text("x")
    roots = mounted_roots(vault, host_access=True)

    assert roots.ceiling_of(vault / "deep" / "main.tex") == vault.resolve()
    assert roots.ceiling_of(elsewhere / "main.tex") == elsewhere
    assert roots.ceiling_of(elsewhere) == elsewhere


@pytest.mark.parametrize("name", ["", ".hidden", "..", "a/b", "a\\b", "a\x00b"])
def test_a_segment_is_one_visible_name_without_separators_or_control_characters(name):
    with pytest.raises(ValueError, match="Invalid name"):
        segment(name)


def test_a_segment_may_be_hidden_on_request():
    assert segment("main.tex") == "main.tex"
    assert segment(".gitignore", hidden=True) == ".gitignore"
    with pytest.raises(ValueError, match="Invalid name"):
        segment("..", hidden=True)
