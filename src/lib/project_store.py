"""Single seam for all project catalog I/O, slug logic, kanban CRUD, and tab management."""

from pathlib import PurePosixPath
from typing import Any, Callable
import uuid

from lib.chat_store import ChatStore, CleanupUploads
from lib.clock import utc_now_iso
from lib.data_store import StorageConflictError
from lib.naming import slugify
from lib.storage_namespace import META_JSON, PROJECT_JSON, chat_root, project_dir

# Moves a project to a new slug: `(slug, new_slug, name)`.
Rename = Callable[[str, str, str], None]

_META_KEY = f"{chat_root()}/{META_JSON}"


class ProjectExists(ValueError):
    """A new project's directory is already taken."""


def _by_file_name(path: str) -> str:
    return PurePosixPath(path).name.lower()


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

    def delete_project(self, slug: str, *, cleanup_uploads_fn: CleanupUploads | None = None) -> dict[str, str]:
        meta, _ = self._require(slug)
        if self._docs.exists(project_dir(slug)):
            if cleanup_uploads_fn:
                for chat_id in self._chats.chat_ids_in(slug):
                    cleanup_uploads_fn(chat_id, slug)
            self._docs.delete(project_dir(slug), recursive=True)
            self._chats.invalidate_index()
        meta["projects"] = [p for p in meta.get("projects", []) if p.get("slug") != slug]
        self._write_meta(meta)
        return {"status": "ok"}

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
        """List document paths from project.json (arbitrary absolute paths), sorted by file name."""
        return sorted(self._mutate_files(slug, lambda files: files), key=_by_file_name)

    def add_file(self, slug: str, path: str) -> list[str]:
        return self._mutate_files(slug, lambda files: files if path in files else [*files, path])

    def remove_file(self, slug: str, path: str) -> list[str]:
        return self._mutate_files(slug, lambda files: [f for f in files if f != path])

    def replace_file(self, slug: str, old: str, new: str) -> list[str]:
        """Point the reference *old* at *new*, in place, merging when *new* is already referenced."""
        return self._mutate_files(
            slug, lambda files: [f for f in files if f != old] if new in files else [new if f == old else f for f in files],
        )

    def replace_file_everywhere(self, old: str, new: str) -> None:
        """In every project referencing *old*, drop it and reference *new* last unless it already is."""
        slugs = [entry["slug"] for entry in self._read_meta().get("projects", [])]
        for slug, data in zip(slugs, self._read_projects(slugs), strict=True):
            files = [str(p) for p in data.get("files", [])]
            if old in files:
                kept = [f for f in files if f != old]
                data["files"] = kept if new in kept else [*kept, new]
                self._write_project(slug, data)

    def _mutate_files(self, slug: str, transform: Callable[[list[str]], list[str]]) -> list[str]:
        """Read, transform, and (when changed) persist a project's document list."""
        self._require(slug)
        data = self._read_project(slug)
        current = [str(p) for p in data.get("files", [])]
        updated = transform(current)
        if updated != current:
            data["files"] = updated
            self._write_project(slug, data)
        return updated

    def list_all_files(self) -> list[str]:
        """Union of every project's documents (the global document pool)."""
        files: dict[str, None] = {}
        for data in self._read_projects([p["slug"] for p in self._read_meta().get("projects", [])]):
            files.update(dict.fromkeys(str(f) for f in data.get("files", [])))
        return sorted(files, key=_by_file_name)
