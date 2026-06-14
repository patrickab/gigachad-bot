"""Single seam for all chat-file I/O, validation, indexing, and cleanup."""

import json
from pathlib import Path
import re
from typing import Any, Callable
import uuid

from config import DIRECTORY_CHAT_HISTORIES
from lib.json_io import safe_read_json, safe_write_json

PROJECT_JSON = "project.json"
META_JSON = "projects-meta.json"
MEMORY_DIR = "memory"


class ChatStore:
    """Owns all chat-file reads, writes, deletes, branching, merging, and indexing."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = (base_dir or DIRECTORY_CHAT_HISTORIES).resolve()
        self._chat_id_index: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolve_path(self, filename: str) -> Path:
        """Resolve *filename* under the base directory, rejecting traversal."""
        resolved = (self._base / filename).resolve()
        if not str(resolved).startswith(str(self._base)):
            raise ValueError(f"Invalid path: {filename}")
        return resolved

    def project_dir_for(self, path: Path) -> Path | None:
        """Return the project directory containing *path*, or None if at root."""
        try:
            rel = path.relative_to(self._base)
        except ValueError:
            return None
        if len(rel.parts) < 2:
            return None
        candidate = self._base / rel.parts[0]
        if candidate.is_dir() and (candidate / PROJECT_JSON).exists():
            return candidate
        return None

    def slug_for(self, path: Path) -> str | None:
        pd = self.project_dir_for(path)
        return pd.name if pd else None

    # ------------------------------------------------------------------
    # Chat ID index
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        if not self._base.exists():
            self._chat_id_index = index
            return index
        for f in self._base.rglob("*.json"):
            if f.name in (PROJECT_JSON, META_JSON):
                continue
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("chat_id"):
                    rel = str(f.relative_to(self._base))
                    index[raw["chat_id"]] = rel
            except Exception:
                continue
        self._chat_id_index = index
        return index

    def invalidate_index(self) -> None:
        self._chat_id_index = None

    def find_by_chat_id(self, chat_id: str) -> str | None:
        if self._chat_id_index is None:
            self._rebuild_index()
        return self._chat_id_index.get(chat_id)

    # ------------------------------------------------------------------
    # Core read / write
    # ------------------------------------------------------------------

    def load(self, filename: str) -> dict[str, Any] | None:
        """Load a chat file by relative filename. Returns None on failure."""
        path = self.resolve_path(filename)
        return self._load_path(path)

    def _load_path(self, path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(raw, list):
            return {
                "messages": raw,
                "chat_id": None,
                "title": None,
                "usage": None,
                "parent_id": None,
                "branch_message_idx": None,
                "children": [],
            }
        return {
            "messages": raw.get("messages", []),
            "chat_id": raw.get("chat_id"),
            "title": raw.get("title"),
            "usage": raw.get("usage"),
            "parent_id": raw.get("parent_id"),
            "branch_message_idx": raw.get("branch_message_idx"),
            "children": raw.get("children", []),
        }

    def save(self, filename: str, data: dict[str, Any] | None = None, *, title: str | None = None) -> dict[str, str]:
        """Write a chat file. Validates chat_id mismatch. Builds payload from *data* and existing."""
        path = self.resolve_path(filename)
        existing = self._load_path(path) if path.exists() else None
        if existing and existing.get("chat_id") and data and data.get("chat_id") and existing["chat_id"] != data["chat_id"]:
            raise ValueError("Chat ID mismatch")
        payload = _build_payload(data, existing, title=title)
        safe_write_json(path, payload)
        self.invalidate_index()
        return {"status": "ok", "filename": str(filename)}

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, filename: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, str]:
        """Delete a single chat file. Updates parent children array if applicable."""
        path = self.resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(filename)
        data = self._load_path(path)
        project_dir = self.project_dir_for(path)
        slug = project_dir.name if project_dir else None
        if data and data.get("chat_id"):
            if cleanup_uploads_fn:
                cleanup_uploads_fn(data["chat_id"], slug)
            if data.get("parent_id"):
                parent_file = self.find_by_chat_id(data["parent_id"])
                if parent_file:
                    self._remove_child_from_parent(parent_file, data["chat_id"])
        if path.is_dir():
            import shutil
            shutil.rmtree(str(path))
        else:
            path.unlink()
        self.invalidate_index()
        self._remove_from_project(project_dir, path.name)
        return {"status": "ok"}

    def _remove_child_from_parent(self, parent_file: str, child_chat_id: str) -> None:
        parent_path = self.resolve_path(parent_file)
        parent_data = self._load_path(parent_path)
        if parent_data:
            parent_data["children"] = [
                c for c in parent_data.get("children", []) if c.get("chat_id") != child_chat_id
            ]
            safe_write_json(parent_path, parent_data)

    def _remove_from_project(self, project_dir: Path | None, filename: str) -> None:
        if project_dir is None:
            return
        proj_data_path = project_dir / PROJECT_JSON
        proj_data = safe_read_json(proj_data_path, {"name": project_dir.name, "kanban": [], "tabs": []})
        proj_data["tabs"] = [t for t in proj_data.get("tabs", []) if t.get("filename") != filename]
        safe_write_json(proj_data_path, proj_data)

    # ------------------------------------------------------------------
    # Cascade delete
    # ------------------------------------------------------------------

    def cascade_delete(self, filename: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, Any]:
        """Delete a root conversation and all its descendants."""
        root_path = self.resolve_path(filename)
        if not root_path.exists():
            raise FileNotFoundError(filename)
        to_delete: list[Path] = [root_path]
        visited: set[str] = {str(root_path)}
        queue = [root_path]
        while queue:
            current = queue.pop(0)
            current_data = self._load_path(current)
            if current_data:
                for child_info in current_data.get("children", []):
                    child_file = self.find_by_chat_id(child_info.get("chat_id", ""))
                    if child_file:
                        child_path = self.resolve_path(child_file)
                        if child_path.exists() and str(child_path) not in visited:
                            queue.append(child_path)
                            to_delete.append(child_path)
                            visited.add(str(child_path))
        deleted: list[str] = []
        for path in to_delete:
            data = self._load_path(path)
            if data and data.get("chat_id"):
                slug = self.slug_for(path)
                if cleanup_uploads_fn:
                    cleanup_uploads_fn(data["chat_id"], slug)
            rel = str(path.resolve().relative_to(self._base))
            deleted.append(rel)
            if path.exists():
                path.unlink()
        self.invalidate_index()
        return {"status": "ok", "deleted": deleted}

    # ------------------------------------------------------------------
    # Orphan
    # ------------------------------------------------------------------

    def orphan(self, filename: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, Any]:
        """Promote children of a deleted root to independent roots, then delete the root."""
        root_path = self.resolve_path(filename)
        if not root_path.exists():
            raise FileNotFoundError(filename)
        root_data = self._load_path(root_path)
        if root_data is None:
            raise ValueError("Failed to read chat file")
        orphaned: list[str] = []
        for child_info in root_data.get("children", []):
            child_file = self.find_by_chat_id(child_info.get("chat_id", ""))
            if child_file:
                child_path = self.resolve_path(child_file)
                child_data = self._load_path(child_path)
                if child_data:
                    child_data["parent_id"] = None
                    child_data["branch_message_idx"] = None
                    safe_write_json(child_path, child_data)
                    orphaned.append(child_file)
        root_slug = self.slug_for(root_path)
        if root_data.get("chat_id") and cleanup_uploads_fn:
            cleanup_uploads_fn(root_data["chat_id"], root_slug)
        root_path.unlink()
        self.invalidate_index()
        return {"status": "ok", "orphaned": orphaned}

    # ------------------------------------------------------------------
    # Rename
    # ------------------------------------------------------------------

    def rename(self, old_path: str, new_title: str) -> dict[str, str]:
        old = self.resolve_path(old_path)
        if not old.exists():
            raise FileNotFoundError(old_path)
        project_dir = self.project_dir_for(old)
        sanitized = _sanitize_title(new_title)
        new_filename = f"{sanitized}.json"
        new_dir = project_dir if project_dir is not None else self._base
        unique_name = _unique_filename(new_dir, new_filename)
        new_path = new_dir / unique_name
        data = self._load_path(old) or {}
        payload = _build_payload(data, title=sanitized)
        safe_write_json(new_path, payload)
        old.unlink()
        if project_dir is not None:
            proj_data_path = project_dir / PROJECT_JSON
            proj_data = safe_read_json(proj_data_path, {"name": project_dir.name, "kanban": [], "tabs": []})
            for t in proj_data.get("tabs", []):
                if t.get("filename") == old.name:
                    t["filename"] = unique_name
                    t["name"] = sanitized
                    t["title"] = sanitized
            safe_write_json(proj_data_path, proj_data)
        rel = new_path.resolve().relative_to(self._base)
        self.invalidate_index()
        return {"status": "ok", "new_path": str(rel), "filename": unique_name}

    # ------------------------------------------------------------------
    # Create directory
    # ------------------------------------------------------------------

    def create_directory(self, parent_path: str, name: str) -> dict[str, str]:
        parent = self._base / parent_path if parent_path else self._base
        dir_path = (parent / name).resolve()
        if not str(dir_path).startswith(str(self._base)):
            raise ValueError("Invalid path")
        dir_path.mkdir(parents=True, exist_ok=True)
        rel = dir_path.relative_to(self._base)
        return {"status": "ok", "path": str(rel)}

    # ------------------------------------------------------------------
    # Move
    # ------------------------------------------------------------------

    def move(self, filename: str, target_dir: str) -> dict[str, str]:
        src = self.resolve_path(filename)
        if not src.exists():
            raise FileNotFoundError(filename)
        src_data = self._load_path(src) or {}
        dst_dir = (self._base / target_dir).resolve() if target_dir else self._base
        if not str(dst_dir).startswith(str(self._base)):
            raise ValueError("Invalid target directory")
        dst_dir.mkdir(parents=True, exist_ok=True)

        is_root = src_data.get("parent_id") is None
        has_children = bool(src_data.get("children"))

        if is_root and has_children:
            children_files = self._collect_descendants(src_data)
            for child_file in children_files:
                child_src = self.resolve_path(child_file)
                if not child_src.exists():
                    continue
                child_data = self._load_path(child_src) or {}
                child_payload = _build_payload(child_data)
                dst_dir.mkdir(parents=True, exist_ok=True)
                child_target = dst_dir / child_src.name
                n = 2
                while child_target.exists() and child_target != child_src:
                    child_target = dst_dir / f"{child_src.stem}-{n}{child_src.suffix}"
                    n += 1
                safe_write_json(child_target, child_payload)
                if child_target != child_src:
                    child_src.unlink()

        dst = dst_dir / src.name
        if dst.exists():
            stem, suffix = src.stem, src.suffix
            n = 2
            while dst.exists():
                dst = dst_dir / f"{stem}-{n}{suffix}"
                n += 1
        new_payload = _build_payload(src_data)
        safe_write_json(dst, new_payload)
        src.unlink()
        self.invalidate_index()
        rel = dst.relative_to(self._base)
        return {"status": "ok", "new_path": str(rel)}

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------

    def branch(self, parent_file: str, branch_message_idx: int) -> dict[str, Any]:
        parent_path = self.resolve_path(parent_file)
        if not parent_path.exists():
            raise FileNotFoundError(parent_file)
        parent_data = self._load_path(parent_path)
        if parent_data is None:
            raise ValueError("Failed to read parent chat history")
        parent_chat_id = parent_data.get("chat_id")
        message_pairs = _extract_qa_pairs(parent_data.get("messages", []))
        if branch_message_idx < 0 or branch_message_idx > len(message_pairs):
            raise ValueError(f"branch_message_idx {branch_message_idx} out of range")
        cutoff = min(branch_message_idx * 2 + 2, len(parent_data.get("messages", [])))
        child_messages = parent_data.get("messages", [])[:cutoff]
        child_chat_id = uuid.uuid4().hex[:12]
        child_filename = _next_branch_filename(parent_path, parent_file, self._base)
        child_path = self.resolve_path(child_filename)
        children = parent_data.get("children", [])
        children.append({"chat_id": child_chat_id, "branch_message_idx": branch_message_idx})
        parent_data["children"] = children
        safe_write_json(parent_path, parent_data)
        child_payload: dict[str, Any] = {
            "messages": child_messages,
            "parent_id": parent_chat_id,
            "branch_message_idx": branch_message_idx,
            "children": [],
        }
        if parent_chat_id:
            child_payload["chat_id"] = child_chat_id
        if parent_data.get("title"):
            child_payload["title"] = parent_data["title"]
        child_path.parent.mkdir(parents=True, exist_ok=True)
        safe_write_json(child_path, child_payload)
        self.invalidate_index()
        return {"status": "ok", "child_file": child_filename, "chat_id": child_chat_id}

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, child_file: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, Any]:
        child_path = self.resolve_path(child_file)
        if not child_path.exists():
            raise FileNotFoundError(child_file)
        child_data = self._load_path(child_path)
        if child_data is None:
            raise ValueError("Failed to read child chat history")
        parent_id = child_data.get("parent_id")
        if not parent_id:
            raise ValueError("Cannot merge a root conversation")
        parent_file = self.find_by_chat_id(parent_id)
        if parent_file is None:
            raise FileNotFoundError("Parent chat history not found")
        parent_path = self.resolve_path(parent_file)
        parent_data = self._load_path(parent_path)
        if parent_data is None:
            raise ValueError("Failed to read parent chat history")
        branch_idx = child_data.get("branch_message_idx")
        if branch_idx is None:
            raise ValueError("Child has no branch_message_idx")
        parent_message_pairs = _extract_qa_pairs(parent_data.get("messages", []))
        if len(parent_message_pairs) > branch_idx + 1:
            raise ValueError("Parent has diverged past the branch point")
        child_messages = child_data.get("messages", [])
        new_messages = child_messages[branch_idx * 2:] if branch_idx * 2 < len(child_messages) else []
        parent_data["messages"] = parent_data.get("messages", []) + new_messages
        parent_children = parent_data.get("children", [])
        child_chat_id = child_data.get("chat_id")
        for sub_child in child_data.get("children", []):
            sub_file = self.find_by_chat_id(sub_child.get("chat_id", ""))
            if sub_file:
                sub_path = self.resolve_path(sub_file)
                sub_data = self._load_path(sub_path)
                if sub_data:
                    sub_data["parent_id"] = parent_data.get("chat_id")
                    sub_data["branch_message_idx"] = branch_idx + len(parent_message_pairs)
                    safe_write_json(sub_path, sub_data)
                    parent_children.append(
                        {
                            "chat_id": sub_child.get("chat_id", ""),
                            "branch_message_idx": branch_idx + len(parent_message_pairs),
                        }
                    )
        parent_children = [c for c in parent_children if c.get("chat_id") != child_chat_id]
        parent_data["children"] = parent_children
        safe_write_json(parent_path, parent_data)
        child_slug = self.slug_for(child_path)
        if child_data.get("chat_id") and cleanup_uploads_fn:
            cleanup_uploads_fn(child_data["chat_id"], child_slug)
        child_path.unlink()
        self.invalidate_index()
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------

    def list_histories(self) -> dict[str, Any]:
        if not self._base.exists():
            return {"files": [], "histories": {}}
        root_files: list[str] = []
        histories: dict[str, list[str]] = {}
        if self._base.is_dir():
            for f in sorted(self._base.iterdir()):
                if f.name in (PROJECT_JSON, META_JSON):
                    continue
                if f.is_dir() and f.name.startswith("_"):
                    continue
                if f.name == MEMORY_DIR:
                    continue
                if f.is_file() and f.suffix == ".json":
                    root_files.append(f.name)
                elif f.is_dir() and not (f / PROJECT_JSON).exists():
                    histories.setdefault(f.name, [])
                    for sf in sorted(f.iterdir()):
                        if sf.is_file() and sf.suffix == ".json" and sf.name != PROJECT_JSON:
                            histories[f.name].append(sf.name)
        return {"files": root_files, "histories": histories}

    def get_branch_meta(self, dirs: str | None = None) -> dict[str, dict[str, Any]]:
        if not self._base.exists():
            return {}
        if dirs is None:
            return self._scan_dir_for_meta(self._base)
        result: dict[str, dict[str, Any]] = {}
        for d in dirs.split(","):
            d = d.strip()
            target = (self._base / d).resolve() if d else self._base
            if not str(target).startswith(str(self._base)):
                continue
            if not target.exists():
                continue
            prefix = f"{d}/" if d else ""
            result.update(self._scan_dir_for_meta(target, prefix))
        return result

    def _scan_dir_for_meta(self, base: Path, rel_prefix: str = "") -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for f in sorted(base.iterdir()):
            if f.name in (PROJECT_JSON, META_JSON):
                continue
            if f.name.startswith("_"):
                continue
            if f.is_file() and f.suffix == ".json":
                key = f"{rel_prefix}{f.name}" if rel_prefix else f.name
                result[key] = self._extract_meta(f)
            elif f.is_dir() and f.name not in ("_uploads", MEMORY_DIR):
                sub_prefix = f"{f.name}/" if not rel_prefix else f"{rel_prefix}{f.name}/"
                result.update(self._scan_dir_for_meta(f, sub_prefix))
        return result

    def _extract_meta(self, path: Path) -> dict[str, Any]:
        data = self._load_path(path)
        if data is None:
            return {}
        messages = data.get("messages", [])
        qa_count = sum(1 for m in messages if m.get("role") == "user")
        return {
            "chat_id": data.get("chat_id"),
            "parent_id": data.get("parent_id"),
            "branch_message_idx": data.get("branch_message_idx"),
            "children": data.get("children", []),
            "qa_count": qa_count,
        }

    # ------------------------------------------------------------------
    # Helpers (project-level, not chat-file specific)
    # ------------------------------------------------------------------

    def _collect_descendants(self, data: dict[str, Any]) -> list[str]:
        files: list[str] = []
        queue = list(data.get("children", []))
        while queue:
            child_info = queue.pop(0)
            child_file = self.find_by_chat_id(child_info.get("chat_id", ""))
            if child_file:
                files.append(child_file)
                child_path = self.resolve_path(child_file)
                if child_path.exists():
                    child_data = self._load_path(child_path)
                    if child_data:
                        queue.extend(child_data.get("children", []))
        return files


# ------------------------------------------------------------------
# Pure helpers (module-level, no state)
# ------------------------------------------------------------------

def _sanitize_title(title: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._\- ]+", "_", title).strip()
    return sanitized or "untitled"


def _unique_filename(directory: Path, base: str) -> str:
    target = directory / base
    if not target.exists():
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    n = 2
    while True:
        candidate = f"{stem}-{n}{suffix}"
        if not (directory / candidate).exists():
            return candidate
        n += 1


def _extract_qa_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "user":
            pair: dict[str, Any] = {"user": messages[i]}
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                pair["assistant"] = messages[i + 1]
                i += 2
            else:
                i += 1
            pairs.append(pair)
        else:
            i += 1
    return pairs


def _next_branch_filename(parent_path: Path, parent_file: str, base_dir: Path) -> str:
    parent_dir = parent_path.parent
    parent_stem = parent_path.stem

    if "/" in parent_file:
        dir_prefix = parent_file.rsplit("/", 1)[0]
    elif parent_path.resolve().parent != base_dir.resolve():
        dir_prefix = parent_path.parent.name
    else:
        dir_prefix = None

    base_stem = re.sub(r"(-\d\d)+$", "", parent_stem)

    if parent_stem == base_stem:
        sibling_pattern = re.compile(r"^" + re.escape(base_stem) + r"(-\d\d)$")
        existing_nums: set[int] = set()
        for f in parent_dir.glob("*.json"):
            m = sibling_pattern.match(f.stem)
            if m:
                existing_nums.add(int(m.group(1)[1:]))
        next_num = 0
        while next_num in existing_nums:
            next_num += 1
        name = f"{base_stem}-{next_num:02d}.json"
    else:
        child_pattern = re.compile(r"^" + re.escape(parent_stem) + r"(-\d\d)$")
        existing_nums: set[int] = set()
        for f in parent_dir.glob("*.json"):
            m = child_pattern.match(f.stem)
            if m:
                existing_nums.add(int(m.group(1)[1:]))
        next_num = 0
        while next_num in existing_nums:
            next_num += 1
        name = f"{parent_stem}-{next_num:02d}.json"

    if dir_prefix:
        return f"{dir_prefix}/{name}"
    return name


def _build_payload(
    data: dict[str, Any] | None = None,
    existing: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Merge chat history fields from *data* and *existing* into a write payload."""
    payload: dict[str, Any] = {"messages": data.get("messages", []) if data else []}

    if data and data.get("chat_id"):
        payload["chat_id"] = data["chat_id"]
    elif existing and existing.get("chat_id"):
        payload["chat_id"] = existing["chat_id"]

    if title:
        payload["title"] = title
    elif data and data.get("title"):
        payload["title"] = data["title"]
    elif existing and existing.get("title"):
        payload["title"] = existing["title"]

    if data and data.get("usage"):
        payload["usage"] = data["usage"]
    elif existing and existing.get("usage"):
        payload["usage"] = existing["usage"]

    if data and data.get("parent_id") is not None:
        payload["parent_id"] = data["parent_id"]
    elif existing and "parent_id" in existing and (data is None or data.get("parent_id") is None):
        payload["parent_id"] = existing["parent_id"]

    if data and data.get("branch_message_idx") is not None:
        payload["branch_message_idx"] = data["branch_message_idx"]
    elif existing and "branch_message_idx" in existing and (data is None or data.get("branch_message_idx") is None):
        payload["branch_message_idx"] = existing["branch_message_idx"]

    if data and data.get("children") is not None:
        payload["children"] = data["children"]
    elif existing and "children" in existing and (data is None or data.get("children") is None):
        payload["children"] = existing["children"]

    return payload