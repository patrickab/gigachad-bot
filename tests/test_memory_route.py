"""/api/memory: list, read and apply over the login's memory-store root."""

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from memory_store import store
import pytest

from backend.identity import RequestIdentity
from backend.routes import memory as memory_route
from tests.conftest import route_client


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store.create_memory_store("user", ["notes"], root=tmp_path)
    return tmp_path


@pytest.fixture
def client(root: Path) -> TestClient:
    return route_client(memory_route.router, overrides={memory_route.get_memory_root: root})


def test_list_without_a_memory_store_lists_the_memory_stores(client: TestClient) -> None:
    response = client.get("/api/memory/list")

    assert response.status_code == 200, response.text
    assert response.json()["memory_stores"] == ["user"]


def test_apply_then_read_a_memory_and_a_document(client: TestClient) -> None:
    ops = [
        {"op": "add", "path": "profile", "description": "Who the user is", "text": "Named Ada"},
        {"op": "write_doc", "path": "notes/plan.md", "text": "# Plan\n"},
        {"op": "retire", "path": "profile", "id": "zz"},
    ]

    applied = client.post("/api/memory/apply", json={"memory_store": "user", "ops": ops})
    memory = client.get("/api/memory/read", params={"memory_store": "user", "path": "profile"})
    document = client.get("/api/memory/read", params={"memory_store": "user", "path": "notes/plan.md"})
    listing = client.get("/api/memory/list", params={"memory_store": "user"})

    results = applied.json()["results"]
    assert [r["ok"] for r in results] == [True, True, False]
    assert results[2]["error"] == "not_found"
    body = memory.json()
    assert body["kind"] == "memory" and body["description"] == "Who the user is"
    assert [set(f) for f in body["facts"]] == [{"id", "text", "created", "updated"}]
    assert body["facts"][0]["text"] == "Named Ada"
    assert document.json() == {"kind": "document", "text": "# Plan\n"}
    assert [m["path"] for m in listing.json()["memories"]] == ["profile"]


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status", "code"),
    [
        ("get", "/api/memory/list", {"params": {"memory_store": "nope"}}, 404, "memory_store_missing"),
        ("get", "/api/memory/read", {"params": {"memory_store": "user", "path": "missing"}}, 404, "not_found"),
        ("get", "/api/memory/read", {"params": {"memory_store": "user", "path": "notes/missing.md"}}, 404, "not_found"),
        ("get", "/api/memory/list", {"params": {"memory_store": "user", "path": "../x"}}, 400, "invalid_path"),
        ("post", "/api/memory/apply", {"json": {"memory_store": "nope", "ops": []}}, 404, "memory_store_missing"),
    ],
)
def test_store_errors_map_to_status_and_code(client: TestClient, method: str, url: str, kwargs: dict, status: int, code: str) -> None:
    response = getattr(client, method)(url, **kwargs)

    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.json()["detail"]["message"]


def test_a_document_symlinked_out_of_the_root_is_refused(
    client: TestClient, root: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    secret = tmp_path_factory.mktemp("outside") / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    (root / "user" / "notes" / "link.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "user" / "notes" / "link.md").symlink_to(secret)

    response = client.get("/api/memory/read", params={"memory_store": "user", "path": "notes/link.md"})

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == "invalid_path"


def test_a_document_under_a_file_is_not_found(client: TestClient, root: Path) -> None:
    (root / "user" / "notes").mkdir(exist_ok=True)
    (root / "user" / "notes" / "a.md").write_text("a", encoding="utf-8")

    response = client.get("/api/memory/read", params={"memory_store": "user", "path": "notes/a.md/b.md"})

    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "not_found"


def test_the_root_follows_the_request_identity() -> None:
    host = RequestIdentity(uuid4(), "host@example.test", None, True)
    family = RequestIdentity(uuid4(), "kid@example.test", None, False)

    assert memory_route.get_memory_root(host) == store.default_root()
    assert memory_route.get_memory_root(family).name == str(family.user_id)
