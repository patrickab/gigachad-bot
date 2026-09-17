import hashlib
import os
from uuid import uuid4

from psycopg_pool import ConnectionPool
import pytest

from lib.asset_store import AssetStore
from lib.data_store import StorageNotFoundError
from lib.db_schema import upgrade


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for asset store tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_database(postgres_pool):
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")


def make_store(pool, login="alice@example.test"):
    device_id = uuid4()
    with pool.connection() as connection, connection.transaction():
        user_id = connection.execute("INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (login,)).fetchone()[0]
        connection.execute("INSERT INTO devices (id, user_id) VALUES (%s, %s)", (device_id, user_id))
    return AssetStore(pool, user_id, device_id=device_id), device_id


def changes_for(pool, kind="asset"):
    with pool.connection() as connection:
        return connection.execute(
            "SELECT resource_key, version, operation, device_id FROM changes WHERE resource_kind = %s ORDER BY seq",
            (kind,),
        ).fetchall()


@pytest.fixture
def store(postgres_pool):
    store, _ = make_store(postgres_pool)
    return store


def test_write_then_read_round_trip(store):
    written = store.write("upload", "PDFs/paper.pdf", b"%PDF-1.7 body")

    read = store.read("PDFs/paper.pdf")

    assert read.id == written.id
    assert read.content == b"%PDF-1.7 body"
    assert read.mime == "application/pdf"
    assert read.size_bytes == len(b"%PDF-1.7 body")
    assert read.sha256 == written.sha256 == hashlib.sha256(b"%PDF-1.7 body").hexdigest()
    assert read.version == 1
    assert store.read_by_id(written.id).content == b"%PDF-1.7 body"


def test_rewrite_bumps_version_and_replaces_content(store):
    store.write("drawing", "drawings/sketch.json", b"first")
    second = store.write("drawing", "drawings/sketch.json", b"second-longer")

    assert second.version == 2
    current = store.read("drawings/sketch.json")
    assert current.version == 2
    assert current.content == b"second-longer"
    assert current.size_bytes == len(b"second-longer")


def test_read_missing_path_raises_not_found(store):
    with pytest.raises(StorageNotFoundError):
        store.read("PDFs/absent.pdf")

    with pytest.raises(StorageNotFoundError):
        store.read_by_id(uuid4())


def test_list_filters_by_prefix_and_kind_without_content(store):
    store.write("pdf", "PDFs/a.pdf", b"a")
    store.write("mineru_markdown", "PDFs/a.md", b"# a")
    store.write("drawing", "drawings/d.json", b"{}")

    under_pdfs = store.list("PDFs")

    assert [asset.logical_path for asset in under_pdfs] == ["PDFs/a.md", "PDFs/a.pdf"]
    assert all(asset.content is None for asset in under_pdfs)
    assert all(asset.size_bytes > 0 and asset.version == 1 for asset in under_pdfs)
    assert [asset.logical_path for asset in store.list("PDFs", kind="pdf")] == ["PDFs/a.pdf"]
    assert [asset.logical_path for asset in store.list(kind="drawing")] == ["drawings/d.json"]
    assert len(store.list()) == 3


def test_delete_removes_the_row(store):
    store.write("upload", "PDFs/a.pdf", b"a")

    store.delete("PDFs/a.pdf")

    assert store.list() == []
    with pytest.raises(StorageNotFoundError):
        store.read("PDFs/a.pdf")


def test_one_user_cannot_touch_another_users_identical_path(postgres_pool):
    alice, _ = make_store(postgres_pool, "alice@example.test")
    bob, _ = make_store(postgres_pool, "bob@example.test")
    alice.write("upload", "PDFs/shared.pdf", b"alice bytes")
    bob.write("upload", "PDFs/shared.pdf", b"bob bytes")

    alice_asset = alice.read("PDFs/shared.pdf")
    bob.delete("PDFs/shared.pdf")

    assert alice_asset.content == b"alice bytes"
    assert bob.list() == []
    assert alice.read("PDFs/shared.pdf").content == b"alice bytes"
    with pytest.raises(StorageNotFoundError):
        bob.read_by_id(alice_asset.id)


def test_each_mutation_appends_a_change_row(postgres_pool):
    store, device_id = make_store(postgres_pool)

    store.write("upload", "PDFs/a.pdf", b"a")
    store.write("upload", "PDFs/a.pdf", b"aa")
    store.delete("PDFs/a.pdf")
    store.delete("PDFs/absent.pdf")

    assert changes_for(postgres_pool) == [
        ("PDFs/a.pdf", 1, "write", device_id),
        ("PDFs/a.pdf", 2, "write", device_id),
        ("PDFs/a.pdf", None, "delete", device_id),
    ]


def test_mirror_writes_allowed_mineru_asset_bytes_under_root(store, tmp_path):
    asset = store.write("mineru_image", "Mineru/doc/images/0.png", b"\x89PNG\r\n\x1a\nbody")

    path = store.mirror(asset, tmp_path)

    assert path == tmp_path / "Mineru" / "doc" / "images" / "0.png"
    assert path.read_bytes() == b"\x89PNG\r\n\x1a\nbody"
    assert list(path.parent.iterdir()) == [path]


def test_mirror_reads_content_back_when_listing_omitted_it(store, tmp_path):
    store.write("pdf", "PDFs/paper.pdf", b"%PDF-1.7 body")
    listed = store.list("PDFs")[0]

    path = store.mirror(listed, tmp_path)

    assert path.read_bytes() == b"%PDF-1.7 body"


def test_mirror_rejects_non_pdf_and_non_mineru_assets(store, tmp_path):
    drawing = store.write("drawing", "drawing/sketch.json", b"{\"ok\":true}")

    with pytest.raises(ValueError, match="Only PDF and MinerU"):
        store.mirror(drawing, tmp_path)

    assert not tmp_path.exists()
