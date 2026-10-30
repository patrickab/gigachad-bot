"""Rewrite stored references in one transaction: rename a project, or relocate one stored document to a new key or disk path."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from lib.changes import UserScope
from lib.clock import utc_now_iso
from lib.data_store import StorageConflictError
from lib.json_io import dump_json
from lib.storage_namespace import ATTACHMENT, CHAT, GRAPH, MEMORY, META_JSON, NOTE, PROJECT, PROJECT_JSON, PROMPT, SANDBOX

if TYPE_CHECKING:
    from uuid import UUID

    from psycopg import Connection

SLUG_FIELDS = {"project_slug", "projectSlug", "slug"}
HISTORY_FIELDS = {"history_file", "historyFile", "parent_file", "filename"}
# App namespaces whose documents may link into a project; others (e.g. `model/`) are never interpreted.
CONTENT_NAMESPACES = {CHAT, PROJECT, NOTE, MEMORY, GRAPH, SANDBOX, PROMPT}
JSON_SUFFIXES = (".json", ".canvas", ".snap")
TEXT_SUFFIXES = (".md", ".txt", ".c4")


class ProjectReferences:
    """Map keys, JSON fields and app-relative links of project `old` to project `new`.

    - Owned namespaces: `chat/`, `project/`, `graph/`, `memory/project/`, `attachment/project/`
    - Prose, absolute disk paths and similarly named projects stay unchanged
    """

    def __init__(self, old: str, new: str) -> None:
        self.old, self.new = old, new
        self._namespaces = [f"{prefix}/" for prefix in (CHAT, PROJECT, GRAPH, f"{MEMORY}/project", f"{ATTACHMENT}/project")]
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
        return _rewrite_content(key, content, self.text, self.value)


class KeyReference:
    """Map the stored key `old` to `new`, another key or an absolute disk path, in JSON strings and app-relative links."""

    def __init__(self, old: str, new: str) -> None:
        self.old, self.new = old, new
        self._link = re.compile(rf"(?<![\w/.-]){re.escape(old)}(?=[?#\s)\]\"']|$)")

    def text(self, value: str) -> str:
        return self._link.sub(lambda _: self.new, value)

    def value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        return self.text(value) if isinstance(value, str) else value

    def content(self, key: str, content: bytes) -> bytes:
        return _rewrite_content(key, content, self.text, self.value)


def _rewrite_content(key: str, content: bytes, text: Callable[[str], str], value: Callable[[Any], Any]) -> bytes:
    """Rewrite JSON and text documents of app namespaces, leave anything else untouched."""
    if key.split("/", 1)[0] not in CONTENT_NAMESPACES or not key.endswith(JSON_SUFFIXES + TEXT_SUFFIXES):
        return content
    try:
        decoded = content.decode()
        if key.endswith(TEXT_SUFFIXES):
            return text(decoded).encode()
        data = json.loads(decoded)
    except ValueError:
        return content
    updated = value(data)
    return content if updated == data else dump_json(updated).encode()


def _rewrite_canvas_mutations(connection: Connection, user_id: UUID, value: Callable[[Any], Any]) -> None:
    """Apply *value* to every stored canvas mutation value, so replaying a canvas log yields rewritten references."""
    mutations = connection.execute(
        "SELECT mutation_id, value FROM canvas_mutations WHERE user_id = %s AND value IS NOT NULL FOR UPDATE", (user_id,),
    ).fetchall()
    for mutation_id, stored in mutations:
        updated = value(stored)
        if updated != stored:
            connection.execute("UPDATE canvas_mutations SET value = %s WHERE mutation_id = %s", (Jsonb(updated), mutation_id))


def renamed_documents(
    rows: Iterable[tuple[str, bytes, bool]], refs: ProjectReferences, name: str, linked_roots: Iterable[str],
) -> Iterator[tuple[str, str, bytes]]:
    """Yield `(source_key, target_key, content)` for each document row the rename changes.

    - `rows` are `(key, content, is_dir)` of every document of the user
    - Catalog entry and `project.json` take the new name
    - A disk folder linked by the old slug's folder name stays linked through `linked_root`
    """
    meta_key = f"{CHAT}/{META_JSON}"
    project_key = f"{CHAT}/{refs.old}/{PROJECT_JSON}"
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


class PostgresProjectRenamer:
    """Renames projects of one user in a single Postgres transaction over documents, assets and canvas mutations."""

    def __init__(self, scope: UserScope) -> None:
        self._scope = scope

    def rename(self, slug: str, new_slug: str, name: str) -> None:
        """Move a project's documents and assets to `new_slug` and rewrite references to them, in one transaction."""
        refs = ProjectReferences(slug, new_slug)
        user_id = self._scope.user_id
        try:
            with self._scope.connection() as connection, connection.transaction():
                rows = connection.execute(
                    "SELECT key, content, is_dir FROM documents WHERE user_id = %s ORDER BY key FOR UPDATE",
                    (user_id,),
                ).fetchall()
                roots = [path for (path,) in connection.execute(
                    "SELECT path FROM vault_roots WHERE user_id = %s ORDER BY created_at, path", (user_id,),
                ).fetchall()]
                documents = [(key, bytes(content), is_dir) for key, content, is_dir in rows]
                for source, target, content in renamed_documents(documents, refs, name, roots):
                    version = connection.execute(
                        "UPDATE documents SET key = %s, content = %s, version = version + 1, updated_at = now()"
                        " WHERE user_id = %s AND key = %s RETURNING version",
                        (target, content, user_id, source),
                    ).fetchone()[0]
                    # Canvas shares and log keys follow the document FK ON UPDATE CASCADE.
                    if source != target:
                        self._scope.record(connection, kind="document", key=source, operation="move")
                    self._scope.record(
                        connection, kind="document", key=target, operation="move" if source != target else "write", version=version,
                    )
                assets = connection.execute("SELECT logical_path FROM assets WHERE user_id = %s FOR UPDATE", (user_id,)).fetchall()
                for (source,) in assets:
                    target = refs.key(source)
                    if target == source:
                        continue
                    version = connection.execute(
                        "UPDATE assets SET logical_path = %s, version = version + 1, updated_at = now()"
                        " WHERE user_id = %s AND logical_path = %s RETURNING version",
                        (target, user_id, source),
                    ).fetchone()[0]
                    self._scope.record(connection, kind="asset", key=source, operation="move")
                    self._scope.record(connection, kind="asset", key=target, operation="move", version=version)
                _rewrite_canvas_mutations(connection, user_id, refs.value)
        except UniqueViolation as exc:
            raise StorageConflictError("Project rename destination already exists; reload before saving") from exc


class PostgresKeyRelocator:
    """Relocates stored documents of one user, rewriting every reference to them in a single Postgres transaction."""

    def __init__(self, scope: UserScope) -> None:
        self._scope = scope

    def relocate(self, old: str, new: str) -> None:
        """Point every reference to the stored document `old` at `new`, then rekey `old` to `new`, or delete it for a disk path `new`.

        - References: project file lists, chat attachments, canvases and their mutation log, app links in notes
        - A disk path `new` makes `old` a mounted file everywhere, so it is read from disk from then on
        """
        refs = KeyReference(old, new)
        user_id = self._scope.user_id
        try:
            with self._scope.connection() as connection, connection.transaction():
                rows = connection.execute(
                    "SELECT key, content FROM documents WHERE user_id = %s AND NOT is_dir AND key <> %s ORDER BY key FOR UPDATE",
                    (user_id, old),
                ).fetchall()
                for key, stored in rows:
                    content = bytes(stored)
                    updated = refs.content(key, content)
                    if updated == content:
                        continue
                    version = connection.execute(
                        "UPDATE documents SET content = %s, version = version + 1, updated_at = now()"
                        " WHERE user_id = %s AND key = %s RETURNING version",
                        (updated, user_id, key),
                    ).fetchone()[0]
                    self._scope.record(connection, kind="document", key=key, operation="write", version=version)
                _rewrite_canvas_mutations(connection, user_id, refs.value)
                if Path(new).is_absolute():
                    connection.execute("DELETE FROM documents WHERE user_id = %s AND key = %s", (user_id, old))
                    self._scope.record(connection, kind="document", key=old, operation="delete")
                else:
                    # Canvas shares and log keys follow the document FK ON UPDATE CASCADE.
                    version = connection.execute(
                        "UPDATE documents SET key = %s, version = version + 1, updated_at = now()"
                        " WHERE user_id = %s AND key = %s RETURNING version",
                        (new, user_id, old),
                    ).fetchone()[0]
                    self._scope.record(connection, kind="document", key=old, operation="move")
                    self._scope.record(connection, kind="document", key=new, operation="move", version=version)
        except UniqueViolation as exc:
            raise StorageConflictError(f"Destination already exists: {new}") from exc
