"""Single seam for all project catalog I/O, slug logic, kanban CRUD, and tab management."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Callable
import uuid

from lib.chat_store import ChatStore
from lib.clock import utc_now_iso
from lib.data_store import StorageConflictError
from lib.naming import slugify
from lib.storage_namespace import META_JSON, PROJECT_JSON, chat_root, project_dir, project_document_prefix

if TYPE_CHECKING:
    from lib.references import Released

# Moves a project to a new slug: `(slug, new_slug, name)`.
Rename = Callable[[str, str, str], None]

_META_KEY = f"{chat_root()}/{META_JSON}"
# The project.json field holding the ids of the project's learn sessions (LearnHarness chats).
LEARN_SESSIONS = "learn_sessions"
# Tries of a guarded read-modify-write of `project.json` before it gives up.
_GUARDED_ATTEMPTS = 3


class ProjectExists(ValueError):
    """A new project's directory is already taken."""


def _files_of(data: dict[str, Any]) -> dict[str, str | None]:
    """A project state's display names by referenced path, in reference order (None: shown under its file name).

    `files` entries are a plain path, or `{"path", "name"}` for a file with a display name.
    """
    files: dict[str, str | None] = {}
    for entry in data.get("files", []):
        if isinstance(entry, dict):
            if entry.get("path"):
                files[str(entry["path"])] = str(entry["name"]) if entry.get("name") else None
        else:
            files[str(entry)] = None
    return files


def _file_names(data: dict[str, Any]) -> dict[str, str]:
    """The display names a project state *data* sets, by path; files shown under their own name are left out."""
    return {path: name for path, name in _files_of(data).items() if name is not None}


def _sorted_by_name(files: dict[str, str | None]) -> list[tuple[str, str]]:
    """`(path, display name)` pairs, a missing name filled in by the file name, sorted case-insensitively by name."""
    named = [(path, name or PurePosixPath(path).name) for path, name in files.items()]
    return sorted(named, key=lambda item: item[1].lower())


def _project_key(slug: str) -> str:
    return f"{project_dir(slug)}/{PROJECT_JSON}"


class ProjectStore:
    """Owns all project-catalog reads/writes, slug generation, kanban CRUD, and tab management."""

    def __init__(self, chat_store: ChatStore, *, rename: Rename) -> None:
        self._chats = chat_store
        self._docs = chat_store.docs
        self._rename = rename

    def _read_json(self, key: str, default: dict[str, Any]) -> dict[str, Any]:
        """Return *default* when the key is missing or holds invalid or non-object JSON."""
        data = self._docs.read(key)
        return data if isinstance(data, dict) else default

    def _write_json(self, key: str, data: dict[str, Any]) -> None:
        """Write with the current revision so a concurrent change raises StorageConflictError."""
        self._docs.overwrite(key, data)

    def _read_meta(self) -> dict[str, Any]:
        return self._read_json(_META_KEY, {"projects": []})

    def _write_meta(self, meta: dict[str, Any]) -> None:
        self._write_json(_META_KEY, meta)

    @staticmethod
    def _find_entry(meta: dict[str, Any], slug: str) -> dict[str, Any] | None:
        return next((p for p in meta.get("projects", []) if p.get("slug") == slug), None)

    def _require(self, slug: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """The catalog and the entry of project *slug*. Raises FileNotFoundError when the project is unknown."""
        meta = self._read_meta()
        entry = self._find_entry(meta, slug)
        if not entry:
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir(slug)  # A malformed catalog slug raises ValueError before any write.
        return meta, entry

    def require(self, slug: str) -> None:
        """Raise FileNotFoundError when the catalog has no project *slug*."""
        self._require(slug)

    @staticmethod
    def _unique_slug(meta: dict[str, Any], base: str) -> str:
        existing = {p.get("slug") for p in meta.get("projects", [])}
        slug = base
        i = 2
        while slug in existing:
            slug = f"{base}-{i}"
            i += 1
        return slug

    @staticmethod
    def _default_project(slug: str) -> dict[str, Any]:
        return {"name": slug, "kanban": [], "tabs": [], "files": []}

    def _read_project(self, slug: str) -> dict[str, Any]:
        return self._read_json(_project_key(slug), self._default_project(slug))

    def _read_projects(self, slugs: list[str]) -> list[dict[str, Any]]:
        """The state of each project in *slugs*, read in one round trip, defaulted like `_read_project`."""
        stored = self._docs.read_many(_project_key(slug) for slug in slugs)
        return [data if isinstance(data := stored[_project_key(slug)], dict) else self._default_project(slug) for slug in slugs]

    def _write_project(self, slug: str, data: dict[str, Any]) -> None:
        self._write_json(_project_key(slug), data)

    def list_projects(self) -> list[dict[str, Any]]:
        entries = self._read_meta().get("projects", [])
        return [
            {
                "name": p["name"],
                "slug": p["slug"],
                "archived": p.get("archived", False),
                "tabs": data.get("tabs", []),
                "linked_root": data.get("linked_root"),
                "file_names": _file_names(data),
            }
            for p, data in zip(entries, self._read_projects([p["slug"] for p in entries]), strict=True)
        ]

    def get_project(self, slug: str) -> dict[str, Any]:
        _, entry = self._require(slug)
        data = self._read_project(slug)
        return {"name": entry["name"], "slug": entry["slug"], "kanban": data.get("kanban", []), "tabs": data.get("tabs", []),
                "linked_root": data.get("linked_root")}

    def create_project(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Project name required")
        meta = self._read_meta()
        slug = self._unique_slug(meta, slugify(name, fallback="project"))
        if self._docs.exists(project_dir(slug)):
            raise ProjectExists(f"Project already exists: {slug}")
        now = utc_now_iso()
        entry = {"name": name, "slug": slug, "createdAt": now, "updatedAt": now}
        meta.setdefault("projects", []).append(entry)
        self._write_meta(meta)
        self._write_project(slug, {"name": name, "kanban": [], "tabs": []})
        return {"name": name, "slug": slug, "kanban": [], "tabs": []}

    def update_project(self, slug: str, name: str | None = None, *, archived: bool | None = None) -> dict[str, Any]:
        """Update archival and name. A name with another slug moves the whole project to that slug."""
        meta, entry = self._require(slug)
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Project name required")
            new_slug = slugify(name, fallback="project")
            if new_slug == slug:
                self._write_project(slug, {**self._read_project(slug), "name": name})
            elif self._find_entry(meta, new_slug):
                raise StorageConflictError(f"Project already exists: {new_slug}")
            else:
                self._rename(slug, new_slug, name)
                self._chats.invalidate_index()
                slug, meta = new_slug, self._read_meta()
                entry = self._find_entry(meta, slug)
            entry["name"] = name
            entry["updatedAt"] = utc_now_iso()
        if archived is not None:
            entry["archived"] = archived
            entry["updatedAt"] = utc_now_iso()
        self._write_meta(meta)
        return {"name": entry["name"], "slug": slug, "archived": entry.get("archived", False)}

    def delete_project(self, slug: str, *, released: Released | None = None) -> dict[str, str]:
        """Delete the project's chats and catalog entry, recording its chats, files and stored documents in *released*."""
        meta, _ = self._require(slug)
        if released is not None:
            released.refs.update(self.list_files(slug))
            released.refs.update(e.key for e in self._docs.list(project_document_prefix(slug), recursive=True) if not e.is_dir)
        if self._docs.exists(project_dir(slug)):
            if released is not None:
                self._chats.release_chats_under(project_dir(slug), slug, released)
            self._docs.delete(project_dir(slug), recursive=True)
            self._chats.invalidate_index()
        meta["projects"] = [p for p in meta.get("projects", []) if p.get("slug") != slug]
        self._write_meta(meta)
        return {"status": "ok"}

    def learn_sessions(self, slug: str) -> list[str]:
        """Ids of the learn sessions grouped under project *slug*. Raises FileNotFoundError when the project is unknown."""
        self._require(slug)
        return list(self._read_project(slug).get(LEARN_SESSIONS, []))

    def learn_session_projects(self) -> dict[str, str]:
        """The project slug of every grouped learn session, by session id."""
        slugs = [p["slug"] for p in self._read_meta().get("projects", [])]
        return {
            session_id: slug
            for slug, data in zip(slugs, self._read_projects(slugs), strict=True)
            for session_id in data.get(LEARN_SESSIONS, [])
        }

    def assign_learn_session(self, session_id: str, slug: str | None) -> None:
        """Group learn session *session_id* under project *slug*, moving it out of any other project; None ungroups it.

        Raises FileNotFoundError when project *slug* is unknown.
        """
        if slug is not None:
            self._require(slug)
        slugs = [p["slug"] for p in self._read_meta().get("projects", [])]
        for owner, data in zip(slugs, self._read_projects(slugs), strict=True):
            sessions = data.get(LEARN_SESSIONS, [])
            if owner == slug and session_id not in sessions:
                self._update_learn_sessions(owner, lambda ids: ids if session_id in ids else [*ids, session_id])
            elif owner != slug and session_id in sessions:
                self._update_learn_sessions(owner, lambda ids: [s for s in ids if s != session_id])

    def forget_learn_sessions(self, session_ids: set[str]) -> None:
        """Drop *session_ids* from every project's learn sessions: sessions deleted in LearnHarness."""
        slugs = [p["slug"] for p in self._read_meta().get("projects", [])]
        for owner, data in zip(slugs, self._read_projects(slugs), strict=True):
            if not session_ids.isdisjoint(data.get(LEARN_SESSIONS, [])):
                self._update_learn_sessions(owner, lambda ids: [s for s in ids if s not in session_ids])

    def _update_learn_sessions(self, slug: str, change: Callable[[list[str]], list[str]]) -> None:
        """Apply *change* to project *slug*'s learn sessions, guarded by the revision of the read it changes.

        Another writer of `project.json` in between (tab autosave, cards, files) retries on its result, so neither
        write is lost. Raises StorageConflictError when the file keeps changing.
        """
        key = _project_key(slug)
        for _ in range(_GUARDED_ATTEMPTS):
            try:
                data, revision = self._docs.read_with_revision(key)
            except FileNotFoundError:
                data, revision = None, None
            if not isinstance(data, dict):
                data = self._default_project(slug)
            sessions = list(data.get(LEARN_SESSIONS, []))
            changed = change(sessions)
            if changed == sessions:
                return
            try:
                self._docs.write(key, {**data, LEARN_SESSIONS: changed}, expected=revision)
            except StorageConflictError:
                continue
            return
        raise StorageConflictError(f"{key} kept changing while its learn sessions were updated")

    def add_card(self, slug: str, title: str, description: str = "", state: str = "backlog") -> dict[str, Any]:
        self._require(slug)
        data = self._read_project(slug)
        card = {"id": str(uuid.uuid4()), "title": title, "description": description, "state": state}
        data.setdefault("kanban", []).append(card)
        self._write_project(slug, data)
        return card

    def update_card(self, slug: str, card_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        self._require(slug)
        data = self._read_project(slug)
        for card in data.get("kanban", []):
            if card["id"] == card_id:
                card.update(updates)
                self._write_project(slug, data)
                return card
        raise FileNotFoundError(f"Card not found: {card_id}")

    def delete_card(self, slug: str, card_id: str) -> dict[str, str]:
        self._require(slug)
        data = self._read_project(slug)
        data["kanban"] = [c for c in data.get("kanban", []) if c["id"] != card_id]
        self._write_project(slug, data)
        return {"status": "ok"}

    def save_tab(self, slug: str, filename: str, data: dict[str, Any]) -> dict[str, str]:
        self._require(slug)
        self._chats.save(f"{slug}/{filename}", data)
        project_data = self._read_project(slug)
        tabs = project_data.get("tabs", [])
        existing = next((t for t in tabs if t["filename"] == filename), None)
        tab_name = data.get("tab_name")
        title = data.get("title")
        if existing:
            if tab_name is not None:
                existing["name"] = tab_name
            if title is not None:
                existing["title"] = title
        else:
            tabs.append({"filename": filename, "name": tab_name, "title": title})
            project_data["tabs"] = tabs
        self._write_project(slug, project_data)
        return {"status": "ok"}

    def list_files(self, slug: str) -> list[str]:
        """List document paths from project.json (arbitrary absolute paths), sorted by display name."""
        return [path for path, _ in self.list_named_files(slug)]

    def list_named_files(self, slug: str) -> list[tuple[str, str]]:
        """The project's `(path, display name)` pairs, sorted by display name."""
        return _sorted_by_name(self._mutate_files(slug, lambda files: files))

    def add_file(self, slug: str, path: str) -> None:
        self._mutate_files(slug, lambda files: files if path in files else {**files, path: None})

    def remove_file(self, slug: str, path: str) -> None:
        self._mutate_files(slug, lambda files: {f: name for f, name in files.items() if f != path})

    def replace_file(self, slug: str, old: str, new: str) -> None:
        """Point the reference *old* at *new*, in place and keeping its display name, merging when *new* is already referenced."""
        self._mutate_files(slug, lambda files: {
            new if f == old else f: name for f, name in files.items() if f != old or old == new or new not in files
        })

    def set_file_name(self, slug: str, path: str, name: str) -> None:
        """Show the referenced *path* as *name*; its own file name clears the display name.

        Raises FileNotFoundError when the project does not reference *path*.
        """
        def rename(files: dict[str, str | None]) -> dict[str, str | None]:
            if path not in files:
                raise FileNotFoundError(f"Project file not found: {path}")
            return {**files, path: None if name == PurePosixPath(path).name else name}

        self._mutate_files(slug, rename)

    def _mutate_files(self, slug: str, transform: Callable[[dict[str, str | None]], dict[str, str | None]]) -> dict[str, str | None]:
        """Read, transform, and (when changed) persist a project's display names by path, in reference order."""
        self._require(slug)
        data = self._read_project(slug)
        current = _files_of(data)
        updated = transform(current)
        if list(updated.items()) != list(current.items()):
            data["files"] = [path if name is None else {"path": path, "name": name} for path, name in updated.items()]
            self._write_project(slug, data)
        return updated

    def list_all_files(self) -> list[str]:
        """Union of every project's documents (the global document pool)."""
        return [path for path, _ in self.list_all_named_files()]

    def list_all_named_files(self) -> list[tuple[str, str]]:
        """The global document pool as `(path, display name)`, named by the first project that names it."""
        files: dict[str, str | None] = {}
        for data in self._read_projects([p["slug"] for p in self._read_meta().get("projects", [])]):
            for path, name in _files_of(data).items():
                if files.get(path) is None:
                    files[path] = name
        return _sorted_by_name(files)
