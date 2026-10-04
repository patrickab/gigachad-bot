"""A workspace on disk (e.g. a checked-out repository) behaves like one in the database."""

from pathlib import Path
import subprocess

import pytest

from lib.architecture_workspace import ArchitectureError, ArchitectureNotFound, ArchitectureStore, DirectoryFiles
from lib.data_store import StorageConflictError


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _layout(root: Path) -> None:
    for relative in ("docs/model.c4", "docs/backend/backend.c4", "docs/frontend/web/frontend.c4"):
        _write(root, relative, f"// {relative}\n")
    _write(root, "docs/README.md", "not a source")
    _write(root, "docs/node_modules/pkg/lib.c4", "dependency")
    _write(root, "docs/backend/generated.c4", "build output")
    _write(root, "elsewhere.c4", "outside the location")
    (root / "docs" / "linked.c4").symlink_to(root / "elsewhere.c4")


def test_sources_in_a_repository_skip_what_git_ignores(tmp_path):
    _layout(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path, ".gitignore", "node_modules/\n")
    _write(tmp_path, "docs/backend/.gitignore", "generated.c4\n")
    # Tracked or not, only ignore rules decide; a tracked file removed from disk is gone.
    _write(tmp_path, "docs/removed.c4", "deleted after commit")
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/model.c4", "docs/removed.c4"], check=True)
    (tmp_path / "docs" / "removed.c4").unlink()

    snapshot = ArchitectureStore(DirectoryFiles(tmp_path), "docs").read()

    assert sorted(snapshot.sources) == ["backend/backend.c4", "frontend/web/frontend.c4", "model.c4"]
    assert snapshot.sources["backend/backend.c4"] == "// docs/backend/backend.c4\n"


def test_sources_outside_a_repository_are_every_c4_file(tmp_path):
    _layout(tmp_path)

    snapshot = ArchitectureStore(DirectoryFiles(tmp_path), "docs").read()

    assert sorted(snapshot.sources) == [
        "backend/backend.c4", "backend/generated.c4", "frontend/web/frontend.c4", "model.c4", "node_modules/pkg/lib.c4",
    ]


def test_a_stale_snapshot_cannot_overwrite_a_file_changed_on_disk(tmp_path):
    _write(tmp_path, "model.c4", "model {\n}\n")
    _write(tmp_path, "old.c4", "model {\n}\n")
    store = ArchitectureStore(DirectoryFiles(tmp_path))
    snapshot = store.read()
    _write(tmp_path, "model.c4", "model {\n  edited = system\n}\n")

    mine = {"model.c4": "model {\n  mine = system\n}\n", "api/api.c4": "model {\n}\n"}
    with pytest.raises(StorageConflictError):
        store.commit(snapshot, mine, snapshot.snapshots, deleted={"old.c4"})
    # Nothing was written or deleted, not even the file that was new.
    assert not (tmp_path / "api").exists()
    assert (tmp_path / "old.c4").exists()

    fresh = store.read()
    kept = {"model.c4": fresh.sources["model.c4"], "api/api.c4": "model {\n}\n"}
    store.commit(fresh, kept, {".likec4/index.likec4.snap": "{}\n"}, deleted={"old.c4"})
    assert sorted(store.read().sources) == ["api/api.c4", "model.c4"]
    assert store.read().snapshots == {".likec4/index.likec4.snap": "{}\n"}

    # A layout the C4 service no longer returns belongs to a view that is gone.
    store.commit(store.read(), store.read().sources, {})
    assert not (tmp_path / ".likec4/index.likec4.snap").exists()


def test_paths_cannot_leave_the_workspace(tmp_path):
    store = ArchitectureStore(DirectoryFiles(tmp_path / "repo"))
    for path in ("../outside.c4", "/etc/passwd.c4", "notes.md"):
        with pytest.raises(ArchitectureError):
            store.source_key(path)
    with pytest.raises(ArchitectureNotFound):
        store.read()
