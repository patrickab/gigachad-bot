"""Persistence and validation for canonical Architecture Graph YAML documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config import DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS
from lib.data_store import DataStore, LocalDataStore, StorageNotFoundError, read_text, write_text

GRAPH_SUFFIX = ".architecture.yaml"


class ArchitectureGraphError(ValueError):
    """Raised when graph input is malformed or names an unsafe graph file."""


class ArchitectureGraphNotFound(ArchitectureGraphError, FileNotFoundError):
    """Raised when a named graph or draft does not exist. Also a FileNotFoundError, so filesystem-contract callers keep working."""


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureGraphError(f"{field} must be a non-empty string")
    return value


def validate_graph(data: Any) -> dict[str, Any]:
    """Validate the intentionally small v1 graph schema and return *data*."""
    if not isinstance(data, dict):
        raise ArchitectureGraphError("Architecture Graph must be a YAML mapping")
    if data.get("version") != 1:
        raise ArchitectureGraphError("version must be 1")
    _require_string(data.get("title"), "title")
    nodes = data.get("nodes")
    edges = data.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ArchitectureGraphError("nodes and edges must be lists")

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ArchitectureGraphError(f"nodes[{index}] must be a mapping")
        node_id = _require_string(node.get("id"), f"nodes[{index}].id")
        if node_id in node_ids:
            raise ArchitectureGraphError(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        _require_string(node.get("title"), f"nodes[{index}].title")
        bullets = node.get("bullets", [])
        if not isinstance(bullets, list) or not all(isinstance(bullet, str) for bullet in bullets):
            raise ArchitectureGraphError(f"nodes[{index}].bullets must be a list of strings")
        position = node.get("position")
        if not isinstance(position, dict) or not all(isinstance(position.get(axis), (int, float)) for axis in ("x", "y")):
            raise ArchitectureGraphError(f"nodes[{index}].position must contain numeric x and y")
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ArchitectureGraphError(f"edges[{index}] must be a mapping")
        edge_id = _require_string(edge.get("id"), f"edges[{index}].id")
        if edge_id in edge_ids:
            raise ArchitectureGraphError(f"duplicate edge id: {edge_id}")
        edge_ids.add(edge_id)
        for field in ("source", "target"):
            endpoint = _require_string(edge.get(field), f"edges[{index}].{field}")
            if endpoint not in node_ids:
                raise ArchitectureGraphError(f"edges[{index}].{field} references unknown node: {endpoint}")
        if edge.get("direction") not in {"one-way", "bidirectional"}:
            raise ArchitectureGraphError(f"edges[{index}].direction must be one-way or bidirectional")
        if "label" in edge and not isinstance(edge["label"], str):
            raise ArchitectureGraphError(f"edges[{index}].label must be a string")
    return data


def parse_graph(content: str) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ArchitectureGraphError(f"Invalid YAML: {exc}") from exc
    return validate_graph(parsed)


class ArchitectureGraphStore:
    """Owns canonical graph files and their one-draft-per-graph lifecycle."""

    def __init__(self, directory: Path | None = None, *, data_store: DataStore | None = None) -> None:
        directory = (directory or DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS).expanduser().resolve()
        self._legacy_directory = directory if data_store is None else None
        self._store = data_store or LocalDataStore(directory.parent)
        self._prefix = directory.name

    def _key(self, name: str, *, draft: bool = False) -> str:
        if Path(name).name != name or not name.endswith(GRAPH_SUFFIX):
            raise ArchitectureGraphError(f"Graph name must end in {GRAPH_SUFFIX}")
        return f"{self._prefix}/.drafts/{name}" if draft else f"{self._prefix}/{name}"

    def list_paths(self) -> list[str]:
        keys = sorted(
            (entry.key for entry in self._store.list(self._prefix) if not entry.is_dir and entry.key.endswith(GRAPH_SUFFIX)), key=str.lower
        )
        return [self._display_key(key) for key in keys]

    def _display_key(self, key: str) -> str:
        if self._legacy_directory is None:
            return key
        return str((self._legacy_directory.parent / key).resolve())

    def path_for(self, name: str, *, draft: bool = False) -> str:
        """Return a validated canonical/draft path without reading its content."""
        return self._display_key(self._key(name, draft=draft))

    def read(self, name: str, *, draft: bool = False) -> str:
        try:
            content, _ = read_text(self._store, self._key(name, draft=draft))
            return content
        except StorageNotFoundError as exc:
            raise ArchitectureGraphNotFound(f"Architecture Graph not found: {name}") from exc

    def write(self, name: str, content: str, *, draft: bool = False) -> str:
        parse_graph(content)
        key = self._key(name, draft=draft)
        if draft and not self._store.exists(self._key(name)):
            raise ArchitectureGraphNotFound(f"Architecture Graph not found: {name}")
        expected = None
        try:
            _, expected = read_text(self._store, key)
        except StorageNotFoundError:
            pass
        write_text(self._store, key, content, expected=expected)
        return self._display_key(key)

    def has_draft(self, name: str) -> bool:
        return self._store.exists(self._key(name, draft=True))

    def accept_draft(self, name: str) -> str:
        draft = self._key(name, draft=True)
        if not self._store.exists(draft):
            raise ArchitectureGraphNotFound(f"Architecture Graph draft not found: {name}")
        # Revalidate immediately before publication; drafts are files a user may edit externally.
        content, revision = read_text(self._store, draft)
        parse_graph(content)
        destination = self._key(name)
        if self._store.exists(destination):
            _, current = self._store.read_bytes(destination)
            self._store.write_bytes(destination, content.encode("utf-8"), expected=current)
            self._store.delete(draft)
        else:
            self._store.move(draft, destination)
        return self._display_key(destination)

    def discard_draft(self, name: str) -> None:
        self._store.delete(self._key(name, draft=True))
