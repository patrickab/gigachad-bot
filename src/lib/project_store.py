"""Single seam for all project catalog I/O, slug logic, kanban CRUD, and tab management."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Callable

from config import DIRECTORY_CHAT_HISTORIES
from lib.chat_store import META_JSON, PROJECT_JSON, ChatStore
from lib.json_io import safe_read_json, safe_write_json
from lib.naming import slugify
from lib.safe_path import safe_resolve


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    """Owns all project-catalog reads/writes, slug generation, kanban CRUD, and tab management."""

    def __init__(self, base_dir: Path | None = None, chat_store: ChatStore | None = None) -> None:
        self._base = (base_dir or DIRECTORY_CHAT_HISTORIES).resolve()
        self._store = chat_store or ChatStore(base_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _meta_path(self) -> Path:
        return self._base / META_JSON

    def _resolve_project_dir(self, slug: str) -> Path:
        return safe_resolve(self._base, slug)

    # ------------------------------------------------------------------
    # Catalog (projects-meta.json)
    # ------------------------------------------------------------------

    def _read_meta(self) -> dict[str, Any]:
        return safe_read_json(self._meta_path(), {"projects": []})

    def _write_meta(self, meta: dict[str, Any]) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        safe_write_json(self._meta_path(), meta)

    def _find_entry(self, meta: dict[str, Any], slug: str) -> dict[str, Any] | None:
        for p in meta.get("projects", []):
            if p.get("slug") == slug:
                return p
        return None

    def _unique_slug(self, meta: dict[str, Any], base: str) -> str:
        existing = {p.get("slug") for p in meta.get("projects", [])}
        slug = base
        i = 2
        while slug in existing:
            slug = f"{base}-{i}"
            i += 1
        return slug

    # ------------------------------------------------------------------
    # Project file (project.json)
    # ------------------------------------------------------------------

    def _read_project(self, project_dir: Path) -> dict[str, Any]:
        return safe_read_json(
            project_dir / PROJECT_JSON,
            {"name": project_dir.name, "kanban": [], "tabs": [], "files": []},
        )

    def _write_project(self, project_dir: Path, data: dict[str, Any]) -> None:
        project_dir.mkdir(parents=True, exist_ok=True)
        safe_write_json(project_dir / PROJECT_JSON, data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        meta = self._read_meta()
        projects: list[dict[str, Any]] = []
        for p in meta.get("projects", []):
            project_dir = self._resolve_project_dir(p["slug"])
            data = self._read_project(project_dir)
            projects.append(
                {
                    "name": p["name"],
                    "slug": p["slug"],
                    "tabs": data.get("tabs", []),
                }
            )
        return projects

    def get_project(self, slug: str) -> dict[str, Any]:
        meta = self._read_meta()
        entry = self._find_entry(meta, slug)
        if not entry:
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        data = self._read_project(project_dir)
        return {"name": entry["name"], "slug": entry["slug"], "kanban": data.get("kanban", []), "tabs": data.get("tabs", [])}

    def create_project(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Project name required")
        meta = self._read_meta()
        slug = self._unique_slug(meta, slugify(name, fallback="project"))
        project_dir = self._resolve_project_dir(slug)
        if project_dir.exists():
            raise ValueError(f"Project already exists: {slug}")
        now = _now_iso()
        entry = {"name": name, "slug": slug, "createdAt": now, "updatedAt": now}
        meta.setdefault("projects", []).append(entry)
        self._write_meta(meta)
        self._write_project(project_dir, {"name": name, "kanban": [], "tabs": []})
        return {"name": name, "slug": slug, "kanban": [], "tabs": []}

    def update_project(self, slug: str, name: str | None = None) -> dict[str, str]:
        meta = self._read_meta()
        entry = self._find_entry(meta, slug)
        if not entry:
            raise FileNotFoundError(f"Project not found: {slug}")
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Project name required")
            entry["name"] = name
            entry["updatedAt"] = _now_iso()
            project_dir = self._resolve_project_dir(slug)
            data = self._read_project(project_dir)
            data["name"] = name
            self._write_project(project_dir, data)
        self._write_meta(meta)
        return {"name": entry["name"], "slug": entry["slug"]}

    def delete_project(self, slug: str) -> dict[str, str]:
        meta = self._read_meta()
        entry = self._find_entry(meta, slug)
        if not entry:
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        if project_dir.exists():
            shutil.rmtree(project_dir)
        meta["projects"] = [p for p in meta.get("projects", []) if p.get("slug") != slug]
        self._write_meta(meta)
        return {"status": "ok"}

    def update_project_state(self, slug: str, kanban: list[dict[str, Any]], tabs: list[dict[str, Any]]) -> dict[str, Any]:
        meta = self._read_meta()
        entry = self._find_entry(meta, slug)
        if not entry:
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        data = self._read_project(project_dir)
        data["kanban"] = kanban
        seen: set[str] = set()
        deduped = []
        for t in tabs:
            d = dict(t)
            if d.get("filename") not in seen:
                seen.add(d.get("filename"))
                deduped.append(d)
        data["tabs"] = deduped
        self._write_project(project_dir, data)
        return {"name": entry["name"], "slug": entry["slug"], "kanban": data["kanban"], "tabs": data["tabs"]}

    # ------------------------------------------------------------------
    # Kanban cards
    # ------------------------------------------------------------------

    def add_card(self, slug: str, title: str, description: str = "", state: str = "backlog") -> dict[str, Any]:
        meta = self._read_meta()
        if not self._find_entry(meta, slug):
            raise FileNotFoundError(f"Project not found: {slug}")
        import uuid

        project_dir = self._resolve_project_dir(slug)
        data = self._read_project(project_dir)
        card = {"id": str(uuid.uuid4()), "title": title, "description": description, "state": state}
        data.setdefault("kanban", []).append(card)
        self._write_project(project_dir, data)
        return card

    def update_card(self, slug: str, card_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        meta = self._read_meta()
        if not self._find_entry(meta, slug):
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        data = self._read_project(project_dir)
        for card in data.get("kanban", []):
            if card["id"] == card_id:
                card.update(updates)
                self._write_project(project_dir, data)
                return card
        raise FileNotFoundError(f"Card not found: {card_id}")

    def delete_card(self, slug: str, card_id: str) -> dict[str, str]:
        meta = self._read_meta()
        if not self._find_entry(meta, slug):
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        data = self._read_project(project_dir)
        data["kanban"] = [c for c in data.get("kanban", []) if c["id"] != card_id]
        self._write_project(project_dir, data)
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Tab files
    # ------------------------------------------------------------------

    def save_tab(self, slug: str, filename: str, data: dict[str, Any]) -> dict[str, str]:
        meta = self._read_meta()
        if not self._find_entry(meta, slug):
            raise FileNotFoundError(f"Project not found: {slug}")
        full_filename = f"{slug}/{filename}"
        try:
            self._store.save(full_filename, data)
        except ValueError as e:
            raise ValueError(str(e))
        project_dir = self._resolve_project_dir(slug)
        project_data = self._read_project(project_dir)
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
        self._write_project(project_dir, project_data)
        return {"status": "ok"}

    def delete_tab(
        self, slug: str, filename: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None
    ) -> dict[str, str]:
        meta = self._read_meta()
        if not self._find_entry(meta, slug):
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        tab_path = project_dir / filename
        chat_id_to_clean: str | None = None
        if tab_path.exists():
            raw = safe_read_json(tab_path, {})
            cid = raw.get("chat_id")
            if isinstance(cid, str):
                chat_id_to_clean = cid
            tab_path.unlink()
        self._store.invalidate_index()
        if chat_id_to_clean and cleanup_uploads_fn:
            cleanup_uploads_fn(chat_id_to_clean, slug)
        project_data = self._read_project(project_dir)
        project_data["tabs"] = [t for t in project_data.get("tabs", []) if t.get("filename") != filename]
        self._write_project(project_dir, project_data)
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Documents (project.json "files" — arbitrary absolute paths)
    # ------------------------------------------------------------------

    def list_files(self, slug: str) -> list[str]:
        return self._mutate_files(slug, lambda files: files)

    def add_file(self, slug: str, path: str) -> list[str]:
        return self._mutate_files(slug, lambda files: files if path in files else [*files, path])

    def remove_file(self, slug: str, path: str) -> list[str]:
        return self._mutate_files(slug, lambda files: [f for f in files if f != path])

    def _mutate_files(self, slug: str, transform: Callable[[list[str]], list[str]]) -> list[str]:
        """Read, transform, and (when changed) persist a project's document list."""
        meta = self._read_meta()
        if not self._find_entry(meta, slug):
            raise FileNotFoundError(f"Project not found: {slug}")
        project_dir = self._resolve_project_dir(slug)
        data = self._read_project(project_dir)
        current = [str(p) for p in data.get("files", [])]
        updated = transform(current)
        if updated != current:
            data["files"] = updated
            self._write_project(project_dir, data)
        return updated

    def list_all_files(self) -> list[str]:
        """Union of every project's documents — the global document pool."""
        meta = self._read_meta()
        seen: set[str] = set()
        out: list[str] = []
        for p in meta.get("projects", []):
            data = self._read_project(self._resolve_project_dir(p["slug"]))
            for f in data.get("files", []):
                f = str(f)
                if f not in seen:
                    seen.add(f)
                    out.append(f)
        out.sort(key=lambda f: Path(f).name.lower())
        return out
