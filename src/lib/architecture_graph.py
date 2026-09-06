"""Persistence and validation for canonical Architecture Graph YAML documents."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from config import DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS
from lib.safe_path import safe_resolve

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

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = (directory or DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS).expanduser().resolve()
        self._drafts = self._directory / ".drafts"

    def _path(self, name: str, *, draft: bool = False) -> Path:
        if Path(name).name != name or not name.endswith(GRAPH_SUFFIX):
            raise ArchitectureGraphError(f"Graph name must end in {GRAPH_SUFFIX}")
        return safe_resolve(self._drafts if draft else self._directory, name)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(content)
            Path(tmp_name).replace(path)
        except OSError:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def list_paths(self) -> list[str]:
        if not self._directory.is_dir():
            return []
        paths = sorted(self._directory.glob(f"*{GRAPH_SUFFIX}"), key=lambda path: path.name.lower())
        return [str(path.resolve()) for path in paths if path.is_file()]

    def path_for(self, name: str, *, draft: bool = False) -> str:
        """Return a validated canonical/draft path without reading its content."""
        return str(self._path(name, draft=draft))

    def read(self, name: str, *, draft: bool = False) -> str:
        path = self._path(name, draft=draft)
        if not path.is_file():
            raise ArchitectureGraphNotFound(f"Architecture Graph not found: {name}")
        return path.read_text(encoding="utf-8")

    def write(self, name: str, content: str, *, draft: bool = False) -> str:
        parse_graph(content)
        path = self._path(name, draft=draft)
        if draft and not self._path(name).is_file():
            raise ArchitectureGraphNotFound(f"Architecture Graph not found: {name}")
        self._atomic_write(path, content)
        return str(path)

    def has_draft(self, name: str) -> bool:
        return self._path(name, draft=True).is_file()

    def accept_draft(self, name: str) -> str:
        draft = self._path(name, draft=True)
        if not draft.is_file():
            raise ArchitectureGraphNotFound(f"Architecture Graph draft not found: {name}")
        # Revalidate immediately before publication; drafts are files a user may edit externally.
        parse_graph(draft.read_text(encoding="utf-8"))
        destination = self._path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        draft.replace(destination)
        return str(destination)

    def discard_draft(self, name: str) -> None:
        self._path(name, draft=True).unlink(missing_ok=True)
