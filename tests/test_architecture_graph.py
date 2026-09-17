from pathlib import Path

from fastapi import HTTPException, Response
import pytest

from backend.routes import architecture_graphs
from lib.architecture_graph import ArchitectureGraphError, ArchitectureGraphStore, parse_graph
from lib.data_store import StorageConflictError


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
"""


def test_store_writes_lists_and_promotes_a_valid_draft(tmp_path: Path):
    store = ArchitectureGraphStore(tmp_path / "Architecture_Graphs")
    name = "checkout.architecture.yaml"

    store.write(name, CONTENT)
    assert store.list_paths() == [str((tmp_path / "Architecture_Graphs" / name).resolve())]

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
    ],
)
def test_parse_graph_rejects_invalid_relationships(content: str, message: str):
    with pytest.raises(ArchitectureGraphError, match=message):
        parse_graph(content)


def test_store_rejects_traversal_and_non_graph_names(tmp_path: Path):
    store = ArchitectureGraphStore(tmp_path / "Architecture_Graphs")
    with pytest.raises(ArchitectureGraphError):
        store.write("../outside.architecture.yaml", CONTENT)
    with pytest.raises(ArchitectureGraphError):
        store.write("checkout.yaml", CONTENT)


def test_draft_requires_an_existing_canonical_graph(tmp_path: Path):
    store = ArchitectureGraphStore(tmp_path / "Architecture_Graphs")
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


async def test_graph_route_writes_a_canonical_file_and_associates_project(tmp_path: Path):
    store = ArchitectureGraphStore(tmp_path / "Architecture_Graphs")
    projects = FakeProjects()
    response = await architecture_graphs.create_graph(
        architecture_graphs.CreateGraphRequest(name="checkout.architecture.yaml", content=CONTENT, projectSlug="project"),
        Response(),
        store,
        projects,
    )

    assert response.hasDraft is False
    assert response.revision
    assert projects.files == [str((tmp_path / "Architecture_Graphs" / "checkout.architecture.yaml").resolve())]


async def test_graph_write_requires_a_matching_revision(tmp_path: Path):
    store = ArchitectureGraphStore(tmp_path / "Architecture_Graphs")
    name = "checkout.architecture.yaml"
    store.write(name, CONTENT)
    edited = CONTENT.replace("Checkout", "Checkout v2")
    http_response = Response()
    loaded = await architecture_graphs.read_graph(name, http_response, store)

    assert http_response.headers["ETag"] == loaded.revision

    # A blind overwrite of an existing graph is refused outright.
    with pytest.raises(HTTPException) as blind:
        await architecture_graphs.write_graph(
            name, architecture_graphs.GraphContentRequest(content=edited), Response(), store, None
        )
    assert blind.value.status_code == 428

    saved = await architecture_graphs.write_graph(
        name, architecture_graphs.GraphContentRequest(content=edited), Response(), store, loaded.revision
    )
    assert saved.content == edited
    assert saved.revision != loaded.revision

    # The first client's now-stale revision must not clobber the newer content.
    with pytest.raises(StorageConflictError):
        await architecture_graphs.write_graph(
            name, architecture_graphs.GraphContentRequest(content=CONTENT), Response(), store, loaded.revision
        )
    assert store.read(name) == edited
