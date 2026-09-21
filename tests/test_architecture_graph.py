import os
from uuid import uuid4

from fastapi import Response
from psycopg_pool import ConnectionPool
import pytest

from backend.routes import architecture_graphs
from lib.architecture_graph import ArchitectureGraphError, ArchitectureGraphStore, parse_graph
from lib.db_schema import upgrade
from lib.postgres_data_store import PostgresDataStore

CONTENT = """\
version: 1
title: Checkout
nodes:
  - id: checkout-api
    title: Checkout API
    bullets:
      - Validates carts
    position: { x: 80, y: 160 }
  - id: payments
    title: Payments
    bullets: []
    position: { x: 500, y: 160 }
edges:
  - id: create-payment
    source: checkout-api
    target: payments
    direction: one-way
    label: Create payment
    path: { bend: 30 }
"""


@pytest.fixture(scope="module")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL is required for Postgres-backed architecture graph tests")
    upgrade(url)
    pool = ConnectionPool(url, min_size=1, max_size=4)
    yield pool
    pool.close()


@pytest.fixture
def graph_store(postgres_pool) -> ArchitectureGraphStore:
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        user_id = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()[0]
    return ArchitectureGraphStore(data_store=PostgresDataStore(postgres_pool, user_id))


def test_store_writes_lists_and_promotes_a_valid_draft(graph_store: ArchitectureGraphStore):
    store = graph_store
    name = "checkout.architecture.yaml"

    store.write(name, CONTENT)
    assert store.list_paths() == [f"graph/{name}"]

    draft = CONTENT.replace("Checkout\n", "Reworked Checkout\n")
    store.write(name, draft, draft=True)
    assert store.has_draft(name)
    store.accept_draft(name)

    assert store.read(name) == draft
    assert not store.has_draft(name)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (CONTENT.replace("target: payments", "target: missing"), "unknown node"),
        (CONTENT.replace("direction: one-way", "direction: perhaps"), "one-way or bidirectional"),
        (CONTENT.replace("path: { bend: 30 }", "path: { bend: left }"), "path.bend must be a finite number"),
    ],
)
def test_parse_graph_rejects_invalid_relationships(content: str, message: str):
    with pytest.raises(ArchitectureGraphError, match=message):
        parse_graph(content)


def test_store_rejects_traversal_and_non_graph_names(graph_store: ArchitectureGraphStore):
    store = graph_store
    with pytest.raises(ArchitectureGraphError):
        store.write("../outside.architecture.yaml", CONTENT)
    with pytest.raises(ArchitectureGraphError):
        store.write("checkout.yaml", CONTENT)


def test_draft_requires_an_existing_canonical_graph(graph_store: ArchitectureGraphStore):
    store = graph_store
    with pytest.raises(FileNotFoundError):
        store.write("checkout.architecture.yaml", CONTENT, draft=True)


class FakeProjects:
    def __init__(self):
        self.files: list[str] = []

    def list_files(self, slug: str) -> list[str]:
        if slug != "project":
            raise FileNotFoundError(slug)
        return self.files

    def add_file(self, slug: str, path: str) -> list[str]:
        self.list_files(slug)
        if path not in self.files:
            self.files.append(path)
        return self.files

    def remove_file(self, slug: str, path: str) -> list[str]:
        self.list_files(slug)
        self.files = [candidate for candidate in self.files if candidate != path]
        return self.files


async def test_graph_route_writes_a_canonical_file_and_associates_project(graph_store: ArchitectureGraphStore):
    store = graph_store
    projects = FakeProjects()
    response = await architecture_graphs.create_graph(
        architecture_graphs.CreateGraphRequest(name="checkout.architecture.yaml", content=CONTENT, projectSlug="project"),
        Response(),
        store,
        projects,
    )

    assert response.hasDraft is False
    assert response.revision
    assert projects.files == ["graph/checkout.architecture.yaml"]
