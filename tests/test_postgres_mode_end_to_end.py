"""End-to-end gate for Postgres mode: identity, storage, revisions, and the change log.

Exercises the real ASGI app so route wiring, dependencies, and exception handlers are
covered together rather than per module.
"""

import os
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from lib.db_schema import upgrade

GRAPH = """\
version: 1
title: Checkout
nodes:
  - id: checkout-api
    title: Checkout API
    bullets:
      - Validates carts
    position: { x: 80, y: 160 }
edges: []
"""


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres end-to-end tests")
    upgrade(url)
    return url


@pytest.fixture
def client(database_url, monkeypatch):
    monkeypatch.setenv("GIGACHAD_DATABASE_URL", database_url)
    import config

    monkeypatch.setattr(config, "_postgres_pool", None)
    from backend.server import app

    with config.get_postgres_pool().connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
    # The lifespan is skipped on purpose: it starts the MinerU queue and the LISTEN broker,
    # neither of which this gate needs.
    yield TestClient(app)
    config.close_postgres_pool()


def headers(login: str, device_id: str | None = None) -> dict[str, str]:
    sent = {"Tailscale-User-Login": login}
    if device_id:
        sent["X-Device-Id"] = device_id
    return sent


def test_unidentified_request_is_rejected(client):
    assert client.get("/api/architecture-graphs").status_code == 401


def test_graph_lifecycle_is_revision_checked_and_user_scoped(client):
    alice = headers("alice@example.test", str(uuid4()))
    bob = headers("bob@example.test", str(uuid4()))

    created = client.post("/api/architecture-graphs", json={"name": "checkout.architecture.yaml", "content": GRAPH}, headers=alice)
    assert created.status_code == 200
    revision = created.json()["revision"]

    read = client.get("/api/architecture-graphs/checkout.architecture.yaml", headers=alice)
    assert read.status_code == 200
    assert read.headers["ETag"] == revision

    edited = GRAPH.replace("Checkout", "Checkout v2")
    blind = client.put("/api/architecture-graphs/checkout.architecture.yaml", json={"content": edited}, headers=alice)
    assert blind.status_code == 428

    saved = client.put(
        "/api/architecture-graphs/checkout.architecture.yaml",
        json={"content": edited},
        headers={**alice, "If-Match": revision},
    )
    assert saved.status_code == 200
    assert saved.json()["content"] == edited

    stale = client.put(
        "/api/architecture-graphs/checkout.architecture.yaml",
        json={"content": GRAPH},
        headers={**alice, "If-Match": revision},
    )
    assert stale.status_code == 412
    assert client.get("/api/architecture-graphs/checkout.architecture.yaml", headers=alice).json()["content"] == edited

    # Bob shares the key namespace but not the data.
    assert client.get("/api/architecture-graphs", headers=bob).json() == {"graphs": []}
    assert client.get("/api/architecture-graphs/checkout.architecture.yaml", headers=bob).status_code == 404


def test_attachment_bytes_round_trip_through_the_asset_route(client):
    device = str(uuid4())
    alice = headers("alice@example.test", device)
    bob = headers("bob@example.test", str(uuid4()))

    uploaded = client.post(
        "/api/files/upload",
        params={"chat_id": "chat-1"},
        files={"file": ("note.txt", b"attachment bytes", "text/plain")},
        headers=alice,
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["name"] == "note.txt"

    logical_path = "attachment/chat/chat-1/note.txt"
    served = client.get(f"/api/assets/{logical_path}", headers=alice)
    assert served.status_code == 200
    assert served.content == b"attachment bytes"
    assert client.get(f"/api/assets/{logical_path}", headers=bob).status_code == 404

    assert client.delete("/api/files/chat/chat-1/att/note.txt", headers=alice).status_code == 200
    assert client.get(f"/api/assets/{logical_path}", headers=alice).status_code == 404


def test_writes_append_a_replayable_change_log(client):
    device = str(uuid4())
    alice = headers("alice@example.test", device)
    client.post("/api/architecture-graphs", json={"name": "a.architecture.yaml", "content": GRAPH}, headers=alice)
    client.post("/api/architecture-graphs", json={"name": "b.architecture.yaml", "content": GRAPH}, headers=alice)

    import config

    with config.get_postgres_pool().connection() as connection:
        rows = connection.execute(
            "SELECT seq, resource_kind, resource_key, device_id FROM changes ORDER BY seq"
        ).fetchall()

    keys = [row[2] for row in rows]
    assert "graph/a.architecture.yaml" in keys
    assert "graph/b.architecture.yaml" in keys
    assert {row[1] for row in rows} == {"document"}
    # Echo suppression needs the writing device on every row.
    assert {str(row[3]) for row in rows} == {device}
    # Sequences must be strictly increasing so a client can resume with ?since=<seq>.
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)


def test_document_editors_read_the_stored_copy_not_the_filesystem(client, tmp_path, monkeypatch):
    """A canvas must read back through /api/fileviewer the way its editor reads it.

    The editors load with the file viewer and save with /api/documents/write, so the
    two paths have to agree. While the viewer read the filesystem it handed every
    editor the last on-disk write, and the autosave then persisted that stale copy
    over the current one.
    """
    import lib.document_library as lib_docs

    monkeypatch.setattr(lib_docs, "DOCUMENTS", tmp_path)

    alice = headers("alice@example.test", str(uuid4()))
    slug = f"canvas-{uuid4().hex[:8]}"
    assert client.post("/api/projects", json={"name": slug}, headers=alice).status_code == 200

    drawn = (
        '{"version":1,"viewport":{"scale":1,"centerX":0,"centerY":0},'
        '"frames":[],"strokes":[{"id":"s1"}],"texts":[],"attachments":[]}'
    )
    written = client.post(
        "/api/documents/write",
        json={"slug": slug, "name": "notes.canvas", "content": drawn},
        headers=alice,
    )
    assert written.status_code == 200
    key = written.json()["path"]

    # A stale copy on disk (what the migration left behind) must never be served:
    # reading it and autosaving it back is exactly how the newer strokes were lost.
    stale = tmp_path / key
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"version":1,"strokes":[]}')

    read = client.get(f"/api/fileviewer/text?path={key}", headers=alice)
    assert read.status_code == 200
    assert read.json()["content"] == drawn

    # The absolute spelling resolves to the same stored copy.
    absolute = str(stale)
    assert client.get(f"/api/fileviewer/text?path={absolute}", headers=alice).json()["content"] == drawn

    # A second edit reads back as the second edit, never as the first or the file.
    redrawn = drawn.replace('"s1"', '"s2"')
    client.post("/api/documents/write", json={"slug": slug, "name": "notes.canvas", "content": redrawn}, headers=alice)
    assert client.get(f"/api/fileviewer/text?path={absolute}", headers=alice).json()["content"] == redrawn

    # Another user's key namespace is separate, so the same path is not theirs.
    bob = headers("bob@example.test", str(uuid4()))
    assert client.get(f"/api/fileviewer/text?path={key}", headers=bob).status_code in {403, 404}


def test_embedded_canvas_images_come_from_storage(client, tmp_path, monkeypatch):
    """Canvas frames point at absolute paths; /raw must serve the stored bytes.

    Nothing is written to disk here, so a filesystem read could only fail or serve
    an unrelated leftover.
    """
    import lib.document_library as lib_docs

    monkeypatch.setattr(lib_docs, "DOCUMENTS", tmp_path)

    alice = headers("alice@example.test", str(uuid4()))
    slug = f"frames-{uuid4().hex[:8]}"
    assert client.post("/api/projects", json={"name": slug}, headers=alice).status_code == 200

    pasted = b"\x89PNG\r\n\x1a\nstored-frame-bytes"
    uploaded = client.post(
        "/api/documents/write-binary",
        params={"slug": slug},
        files={"file": ("pasted-1.png", pasted, "image/png")},
        headers=alice,
    )
    assert uploaded.status_code == 200
    key = uploaded.json()["path"]

    absolute = str(tmp_path / key)
    assert not (tmp_path / key).exists(), "the test must not depend on a disk copy"

    served = client.get(f"/api/fileviewer/raw?path={absolute}", headers=alice)
    assert served.status_code == 200
    assert served.content == pasted
    assert client.get(f"/api/fileviewer/raw?path={key}", headers=alice).content == pasted


def test_pdf_asset_attachment_serves_from_storage_not_only_disk(client, tmp_path, monkeypatch):
    """A canvas PDF attachment's path is the *relative* asset key ``document_meta``
    hands back (``PDFs/<name>``), not an absolute filesystem path. ``/raw`` must
    resolve it through the assets table: the old fallback resolved a relative
    path against the process's cwd, not the Nextcloud mirror root, and 404'd.
    """
    import lib.attachment_materialize as attachment_materialize
    import lib.document_library as lib_docs

    monkeypatch.setattr(lib_docs, "DOCUMENTS", tmp_path)
    monkeypatch.setattr(attachment_materialize, "DOCUMENTS", tmp_path)

    alice = headers("alice@example.test", str(uuid4()))
    slug = f"pdf-{uuid4().hex[:8]}"
    assert client.post("/api/projects", json={"name": slug}, headers=alice).status_code == 200

    pdf_bytes = b"%PDF-1.7 fake pdf body"
    uploaded = client.post(
        "/api/files/upload",
        params={"chat_id": "chat-1", "slug": slug},
        files={"file": ("exercise.pdf", pdf_bytes, "application/pdf")},
        headers=alice,
    )
    assert uploaded.status_code == 200

    registered = client.post(
        "/api/documents/register-upload",
        params={"slug": slug},
        json={"chat_id": "chat-1", "filename": "exercise.pdf"},
        headers=alice,
    )
    assert registered.status_code == 200
    key = registered.json()["path"]
    assert key == "PDFs/exercise.pdf"

    # The mirror write is a convenience copy, not the source of truth: remove it so
    # a pass can only mean the assets table served the bytes.
    (tmp_path / key).unlink()

    served = client.get(f"/api/fileviewer/raw?path={key}", headers=alice)
    assert served.status_code == 200
    assert served.content == pdf_bytes
