from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException
import pytest

from backend.routes import deps, documents
from lib.asset_store import Asset, AssetStore
from lib.local_postgres_migration import inventory
from lib.storage_namespace import legacy_key_to_native
from scripts.purge_legacy_documents_artifacts import main as purge_legacy_artifacts


def make_asset(kind: str, logical_path: str, content: bytes) -> Asset:
    return Asset(
        id=uuid4(),
        kind=kind,
        logical_path=logical_path,
        mime="application/octet-stream",
        sha256="unused",
        size_bytes=len(content),
        version=1,
        content=content,
    )


def test_only_pdf_and_mineru_assets_can_be_mirrored(tmp_path: Path) -> None:
    store = AssetStore.__new__(AssetStore)
    root = tmp_path / "Documents"

    pdf = make_asset("pdf", "PDFs/paper.pdf", b"%PDF-1.7")
    assert store.mirror(pdf, root).read_bytes() == b"%PDF-1.7"
    with pytest.raises(ValueError, match="Only PDF and MinerU"):
        store.mirror(make_asset("drawing", "drawing/canvas.jpg", b"jpeg"), root)

    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    assert files == ["PDFs/paper.pdf"]


def test_only_pdfs_can_enter_the_persisted_library() -> None:
    recorded: list[tuple[str, str, bytes, str]] = []

    class Assets:
        def write(self, kind: str, logical_path: str, content: bytes, *, mime: str) -> Asset:
            recorded.append((kind, logical_path, content, mime))
            return make_asset(kind, logical_path, content)

    stored = documents._library_asset(Assets(), "paper.pdf", b"%PDF-1.7")

    assert (stored.kind, stored.logical_path) == ("pdf", "PDFs/paper.pdf")
    assert recorded == [("pdf", "PDFs/paper.pdf", b"%PDF-1.7", "application/pdf")]

    with pytest.raises(HTTPException, match="Only PDF documents"):
        documents._library_asset(Assets(), "notes.txt", b"database-only")


def test_postgres_vault_roots_use_the_database_repository(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Repository:
        def load(self) -> list[object]:
            return []

        def save(self, roots: list[dict[str, object]]) -> None:
            raise AssertionError(f"unexpected save: {roots}")

    def repository(pool: object, user_id: object, *, device_id: object) -> Repository:
        captured.update(pool=pool, user_id=user_id, device_id=device_id)
        return Repository()

    identity = SimpleNamespace(user_id="user", device_id="device")
    monkeypatch.setattr(deps, "get_postgres_pool", lambda: "pool")
    monkeypatch.setattr(deps, "PostgresVaultRootRepository", repository)

    vault = deps.get_file_vault(identity)

    assert vault.roots() == []
    assert captured == {"pool": "pool", "user_id": "user", "device_id": "device"}


def test_legacy_documents_keys_move_to_database_native_namespaces() -> None:
    assert legacy_key_to_native("chat_history/thread.json") == "chat/thread.json"
    assert legacy_key_to_native("chat_history/_uploads/chat-123/image.png") == "attachment/chat/chat-123/image.png"
    project_upload = "chat_history/project/_uploads/chat-123/image.png"
    assert legacy_key_to_native(project_upload) == "attachment/project/project/chat/chat-123/image.png"
    assert legacy_key_to_native("chat_history/memory/global-profile.md") == "memory/global-profile.md"
    assert legacy_key_to_native("chat_history/project/memory/rules.json") == "memory/project/project/rules.json"
    assert legacy_key_to_native("Prompts/custom.yaml") == "prompt/custom.yaml"
    assert legacy_key_to_native("Architecture_Graphs/.drafts/a.architecture.yaml") == "graph/draft/a.architecture.yaml"
    assert legacy_key_to_native("model-defaults.yaml") == "model/model-defaults.yaml"


def test_purge_removes_only_legacy_application_artifacts(tmp_path: Path) -> None:
    documents = tmp_path / "Documents"
    for name in ("chat_history", "Prompts", "Drawings", "PDFs", "Mineru"):
        (documents / name).mkdir(parents=True)
    (documents / "model-defaults.yaml").write_text("small_model: local")

    assert purge_legacy_artifacts(["--source", str(documents), "--confirm"]) == 0

    assert not (documents / "chat_history").exists()
    assert not (documents / "Prompts").exists()
    assert not (documents / "Drawings").exists()
    assert not (documents / "model-defaults.yaml").exists()
    assert (documents / "PDFs").is_dir()
    assert (documents / "Mineru").is_dir()


def test_legacy_import_inventory_targets_database_native_namespaces(tmp_path: Path) -> None:
    documents = tmp_path / "Documents"
    files = {
        "chat_history/chat-1.json": b"{}",
        "chat_history/_uploads/chat-1/image.png": b"image",
        "Drawings/canvas.jpg": b"drawing",
        "PDFs/paper.pdf": b"%PDF-1.7",
        "Mineru/paper.md": b"# paper",
    }
    for key, content in files.items():
        target = documents / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    artifacts, vault_config = inventory(documents)

    assert vault_config is None
    assert {(artifact.group, artifact.key) for artifact in artifacts} == {
        ("documents", "chat/chat-1.json"),
        ("upload", "attachment/chat/chat-1/image.png"),
        ("drawing", "drawing/canvas.jpg"),
        ("pdf", "PDFs/paper.pdf"),
        ("mineru_markdown", "Mineru/paper.md"),
    }
