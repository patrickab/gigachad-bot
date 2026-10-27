"""Rewrite stored references from one project slug to another."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from lib.clock import utc_now_iso
from lib.json_io import dump_json
from lib.storage_namespace import PROJECT_JSON

SLUG_FIELDS = {"project_slug", "projectSlug", "slug"}
HISTORY_FIELDS = {"history_file", "historyFile", "parent_file", "filename"}
# App namespaces whose documents may link into a project; others (e.g. `model/`) are never interpreted.
CONTENT_NAMESPACES = {"project", "note", "memory", "graph", "sandbox", "prompt"}
JSON_SUFFIXES = (".json", ".canvas", ".snap")
TEXT_SUFFIXES = (".md", ".txt", ".c4")


class ProjectReferences:
    """Map keys, JSON fields and app-relative links of project `old` to project `new`.

    - Owned namespaces: `<chat_prefix>/`, `project/`, `graph/`, `memory/project/`, `attachment/project/`
    - Prose, absolute disk paths and similarly named projects stay unchanged
    """

    def __init__(self, old: str, new: str, chat_prefix: str) -> None:
        self.old, self.new, self.chat_prefix = old, new, chat_prefix
        self._namespaces = [f"{prefix}/" for prefix in (chat_prefix, "project", "graph", "memory/project", "attachment/project")]
        sources = "|".join(re.escape(namespace + old) for namespace in self._namespaces)
        self._link = re.compile(rf"(?<![\w/.-])({sources})(?=/|[?#\s)\]\"']|$)")

    def key(self, value: str) -> str:
        for namespace in self._namespaces:
            old = namespace + self.old
            if value == old or value.startswith(old + "/"):
                return namespace + self.new + value[len(old):]
        return value

    def text(self, value: str) -> str:
        return self._link.sub(lambda match: self.key(match.group(1)), value)

    def value(self, value: Any, field: str = "") -> Any:
        if isinstance(value, dict):
            return {key: self.value(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item, field) for item in value]
        if not isinstance(value, str):
            return value
        if field in SLUG_FIELDS:
            return self.new if value == self.old else value
        if field in HISTORY_FIELDS and value.startswith(self.old + "/"):
            return self.new + value[len(self.old):]
        return self.text(value)

    def content(self, key: str, content: bytes) -> bytes:
        """Rewrite JSON and text documents of app namespaces, leave anything else untouched."""
        if key.split("/", 1)[0] not in CONTENT_NAMESPACES | {self.chat_prefix} or not key.endswith(JSON_SUFFIXES + TEXT_SUFFIXES):
            return content
        try:
            text = content.decode()
            if key.endswith(TEXT_SUFFIXES):
                return self.text(text).encode()
            data = json.loads(text)
        except ValueError:
            return content
        updated = self.value(data)
        return content if updated == data else dump_json(updated).encode()


def renamed_documents(
    rows: Iterable[tuple[str, bytes, bool]], refs: ProjectReferences, name: str, linked_roots: Iterable[str],
) -> Iterator[tuple[str, str, bytes]]:
    """Yield `(source_key, target_key, content)` for each document row the rename changes.

    - `rows` are `(key, content, is_dir)` of every document of the user
    - Catalog entry and `project.json` take the new name
    - A disk folder linked by the old slug's folder name stays linked through `linked_root`
    """
    meta_key = f"{refs.chat_prefix}/projects-meta.json"
    project_key = f"{refs.chat_prefix}/{refs.old}/{PROJECT_JSON}"
    for key, content, is_dir in rows:
        target = refs.key(key)
        updated = content if is_dir else refs.content(key, content)
        if key == meta_key:
            meta = json.loads(updated)
            entry = next(project for project in meta["projects"] if project["slug"] == refs.new)
            entry.update(name=name, updatedAt=utc_now_iso())
            updated = dump_json(meta).encode()
        elif key == project_key:
            project = json.loads(updated)
            project["name"] = name
            linked_root = project.get("linked_root") or next((root for root in linked_roots if Path(root).name == refs.old), None)
            if linked_root:
                project["linked_root"] = linked_root
            updated = dump_json(project).encode()
        if target != key or updated != content:
            yield key, target, updated
