"""End-to-end gate for Postgres mode: identity, storage, revisions, and the change log.

Exercises the real ASGI app so route wiring, dependencies, and exception handlers are
covered together rather than per module.
"""

import os
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from lib.db_schema import upgrade


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
    assert client.get("/api/architecture/project").status_code == 401


@pytest.fixture
def c4_service(monkeypatch):
    import httpx

    from lib import c4_service as service

    url = os.environ.get("C4_SERVICE_URL", "http://127.0.0.1:8011")
    try:
        httpx.get(f"{url}/health", timeout=1).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("the C4 service (src/c4/server.ts) is required for architecture tests")
    monkeypatch.setattr(service, "C4_SERVICE_URL", url)


def test_architecture_edits_are_gated_by_the_model_and_user_scoped(client, c4_service):
    alice = headers("alice@example.test", str(uuid4()))
    slug = f"arch-{uuid4().hex[:8]}"
    client.post("/api/projects", json={"name": slug}, headers=alice)

    assert client.get(f"/api/architecture/{slug}", headers=alice).json()["model"]["tree"] == []
    package = {"ops": [{"op": "createPackage", "title": "Store"}]}
    created = client.post(f"/api/architecture/{slug}/ops", json=package, headers=alice)
    assert created.json()["created"] == ["store/store.c4"]
    assert f"graph/{slug}/store/store.c4" in client.get(f"/api/documents?slug={slug}", headers=alice).text

    ops = [
        {"op": "addElement", "parent": None, "kind": "system", "title": "Shop"},
        {"op": "addElement", "parent": None, "kind": "system", "title": "Bank"},
        {"op": "addRelation", "source": "shop", "target": "bank", "label": "pays"},
    ]
    edited = client.post(f"/api/architecture/{slug}/ops", json={"ops": ops, "file": "store/store.c4"}, headers=alice)
    assert edited.status_code == 200
    body = edited.json()
    assert body["created"][:2] == ["shop", "bank"]
    [relation] = {r for view in body["model"]["views"] for edge in view["edges"] for r in edge["relations"]}

    # Delete never cascades: the relation still names bank, so the batch is refused untouched.
    refused = client.post(
        f"/api/architecture/{slug}/ops",
        json={"ops": [{"op": "delete", "elements": ["bank"], "relations": []}], "file": "store/store.c4"},
        headers=alice,
    )
    assert refused.status_code == 400
    assert client.get(f"/api/architecture/{slug}", headers=alice).json()["model"] == body["model"]

    deleted = client.post(
        f"/api/architecture/{slug}/ops",
        json={"ops": [{"op": "delete", "elements": ["bank"], "relations": [relation]}], "file": "store/store.c4"},
        headers=alice,
    )
    assert deleted.status_code == 200
    # Drawn top-level elements sit beside the package's system.
    assert {e["id"] for e in deleted.json()["model"]["elements"]} == {"store", "shop"}

    source = client.get("/api/fileviewer/text", params={"path": f"graph/{slug}/store/store.c4"}, headers=alice).json()["content"]
    assert "Shop" in source and "bank" not in source

    bob = headers("bob@example.test", str(uuid4()))
    assert client.get(f"/api/architecture/{slug}", headers=bob).status_code == 404
    # Nor can bob start a workspace under a slug that is not one of his projects.
    assert client.post(f"/api/architecture/{slug}/ops", json=package, headers=bob).status_code == 404


def test_packages_modules_and_views_form_one_architecture(client, c4_service):
    alice = headers("alice@example.test", str(uuid4()))
    slug = f"arch-{uuid4().hex[:8]}"
    client.post("/api/projects", json={"name": slug}, headers=alice)
    base = f"/api/architecture/{slug}"

    def create(op: dict) -> dict:
        return client.post(f"{base}/ops", json={"ops": [op]}, headers=alice).json()

    # A package is a folder with one system and its view; the first also declares the kinds.
    assert "system" in create({"op": "createPackage", "title": "Backend"})["model"]["kinds"]
    assert create({"op": "createPackage", "title": "Frontend"})["created"] == ["frontend/frontend.c4"]
    # A module extends its package from a file beside it, and shows in the package's view.
    module = create({"op": "createModule", "package": "backend", "title": "API"})
    assert module["created"] == ["backend/api.c4"]
    assert {entry["path"]: entry["role"] for entry in module["model"]["tree"]} == {
        "backend/api.c4": "module",
        "backend/backend.c4": "package",
        "frontend/frontend.c4": "package",
    }

    # Drawn in frontend's view: the element joins frontend, the connection is stored once beside it.
    ops = [
        {"op": "addElement", "parent": None, "kind": "container", "title": "Web", "layout": {"view": "frontend", "x": 40, "y": 60}},
        {"op": "addRelation", "source": "frontend.web", "target": "backend.api", "label": "calls"},
    ]
    drawn = client.post(f"{base}/ops", json={"ops": ops, "file": "frontend/frontend.c4"}, headers=alice)
    assert drawn.status_code == 200
    views = {view["id"]: view for view in drawn.json()["model"]["views"]}
    assert views["frontend"]["manual"] and views["frontend"]["file"] == "frontend/frontend.c4"
    # backend's view shows the incoming connection, though no backend file mentions it.
    assert [edge["id"] for edge in views["backend"]["edges"]] == ["frontend->backend.api"]
    frontend = client.get("/api/fileviewer/text", params={"path": f"graph/{slug}/frontend/frontend.c4"}, headers=alice).json()
    assert "frontend.web -> backend.api 'calls'" in frontend["content"]

    # A saved view is its own file; adding a whole system writes element includes.
    assert create({"op": "createView", "title": "Checkout"})["created"] == ["views/checkout.c4"]
    include = [
        {"op": "includeInView", "view": "checkout", "elements": ["backend"], "descendants": True},
        {"op": "includeInView", "view": "checkout", "elements": ["frontend.web"]},
        {"op": "layout", "view": "checkout", "nodes": {"backend": {"x": 300, "y": 0}}, "edges": {}},
    ]
    checkout = client.post(f"{base}/ops", json={"ops": include}, headers=alice).json()
    [view] = [view for view in checkout["model"]["views"] if view["id"] == "checkout"]
    assert {"backend", "frontend.web"} <= {node["id"] for node in view["nodes"]}
    assert next(node for node in view["nodes"] if node["id"] == "backend")["x"] == 300

    # Sources are listed by their path in the workspace; saved layouts are not documents.
    listed = {d["name"] for d in client.get(f"/api/documents?slug={slug}", headers=alice).json()["documents"]}
    assert listed == {"backend/backend.c4", "backend/api.c4", "frontend/frontend.c4", "views/checkout.c4"}

    # A package goes with its modules, but not while frontend and the view still point at it.
    backend = {"path": ["backend/backend.c4", "backend/api.c4"]}
    refused = client.delete(f"{base}/source", params=backend, headers=alice)
    assert refused.status_code == 400
    assert refused.json()["detail"].startswith("Cannot delete backend/api.c4, backend/backend.c4: ")
    assert "still refers to 'backend" in refused.json()["detail"]
    assert client.delete(f"{base}/source", params={"path": "backend/api.c4"}, headers=alice).status_code == 400
    for path in ("views/checkout.c4", "frontend/frontend.c4"):
        assert client.delete(f"{base}/source", params={"path": path}, headers=alice).status_code == 200
    emptied = client.delete(f"{base}/source", params=backend, headers=alice)
    assert emptied.status_code == 200 and emptied.json()["model"]["tree"] == []

    assert client.post(f"{base}/source", json={"path": "../escape.c4"}, headers=alice).status_code == 400


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
    slug = f"log-{uuid4().hex[:8]}"
    client.post("/api/projects", json={"name": slug}, headers=alice)
    for name in ("a.md", "b.md"):
        client.post("/api/documents/write", json={"slug": slug, "name": name, "content": name}, headers=alice)

    import config

    with config.get_postgres_pool().connection() as connection:
        rows = connection.execute(
            "SELECT seq, resource_kind, resource_key, device_id FROM changes ORDER BY seq"
        ).fetchall()

    keys = [row[2] for row in rows]
    assert f"project/{slug}/document/a.md" in keys
    assert f"project/{slug}/document/b.md" in keys
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
