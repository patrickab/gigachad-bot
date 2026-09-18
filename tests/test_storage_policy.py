from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException
import pytest

from backend.routes import deps, documents
from lib.asset_store import Asset, AssetStore


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
