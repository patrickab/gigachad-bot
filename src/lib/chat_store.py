"""Single seam for all chat-file I/O, validation, indexing, and cleanup."""

import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable
import uuid

from lib.data_store import DataStore, DataStorePath, Revision
from lib.json_io import dump_json, load_json, safe_read_json, safe_write_json
from lib.storage_namespace import CHAT, PROJECT_JSON

META_JSON = "projects-meta.json"
MEMORY_DIR = "memory"


class ChatStore:
    """Owns all chat-file reads, writes, deletes, branching, merging, and indexing."""

    def __init__(self, prefix: str = CHAT, *, data_store: DataStore) -> None:
        self._base = DataStorePath(data_store, prefix)
        self._chat_id_index: dict[str, str] | None = None

    def resolve_path(self, filename: str) -> DataStorePath:
        """Resolve *filename* under the base directory, rejecting traversal."""
        return self._base / filename

    def project_dir_for(self, path: DataStorePath) -> DataStorePath | None:
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

    def slug_for(self, path: DataStorePath) -> str | None:
        pd = self.project_dir_for(path)
        return pd.name if pd else None

    def _rebuild_index(self) -> dict[str, str]:
        """Scan every chat file once to map chat_id to its relative path."""
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
        """Return the relative path of the chat with this ID, building the index on first use."""
        if self._chat_id_index is None:
            self._rebuild_index()
        return self._chat_id_index.get(chat_id)

    def load(self, filename: str) -> dict[str, Any] | None:
        """Load a chat file and its revision in one read. Returns None on failure."""
        key = self.resolve_path(filename).key
        try:
            content, revision = self._base.store.read_bytes(key)
            raw = json.loads(content)
        except Exception:
            return None
        return {**_normalize_chat(raw), "revision": revision.token}

    def _load_path(self, path: DataStorePath) -> dict[str, Any] | None:
        raw = load_json(path)
        return None if raw is None else _normalize_chat(raw)

    def revision(self, filename: str) -> str | None:
        """Return the opaque revision clients must send with an editable save."""
        try:
            _, revision = self._base.store.read_bytes(self.resolve_path(filename).key)
            return revision.token
        except FileNotFoundError:
            return None

    def save(
        self, filename: str, data: dict[str, Any] | None = None, *, title: str | None = None, expected_revision: str | None = None
    ) -> dict[str, str]:
        """Write a chat file, merging *data* over the stored chat. A given *expected_revision* makes the write conflict-checked."""
        path = self.resolve_path(filename)
        existing = self._load_path(path) if path.exists() else None
        if existing and existing.get("chat_id") and data and data.get("chat_id") and existing["chat_id"] != data["chat_id"]:
            raise ValueError("Chat ID mismatch")
        payload = _build_payload(data, existing, title=title)
        if expected_revision is None:
            safe_write_json(path, payload)
        else:
            self._base.store.write_bytes(path.key, dump_json(payload).encode(), expected=Revision(expected_revision))
        self.invalidate_index()
        return {"status": "ok", "filename": str(filename), "revision": self.revision(filename) or ""}

    def delete(self, filename: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, str]:
        """Delete a single chat file. Updates parent children array and project tab registry if applicable."""
        path = self.resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(filename)
        project_dir = self._release_chat(path, cleanup_uploads_fn)
        if path.is_dir():
            # Delete through the store. A filesystem call such as shutil.rmtree would
            # resolve the relative storage key against the process working directory
            # instead of the storage root.
            self._base.store.delete(path.key, recursive=True)
        else:
            path.unlink()
        self.invalidate_index()
        self._remove_from_project(project_dir, path.name)
        return {"status": "ok"}

    def _release_chat(self, path: DataStorePath, cleanup_uploads_fn: Callable[[str, str | None], None] | None) -> DataStorePath | None:
        """Clean up the uploads of the chat in *path* and detach it from its parent chat. Returns its project directory."""
        data = self._load_path(path)
        project_dir = self.project_dir_for(path)
        if data and data.get("chat_id"):
            if cleanup_uploads_fn:
                cleanup_uploads_fn(data["chat_id"], project_dir.name if project_dir else None)
            if data.get("parent_id"):
                parent_file = self.find_by_chat_id(data["parent_id"])
                if parent_file:
                    self._remove_child_from_parent(parent_file, data["chat_id"])
        return project_dir

    def _prune_children_list(
        self, children: list[dict[str, Any]], known_chat_ids: frozenset[str] | set[str] = frozenset()
    ) -> list[dict[str, Any]]:
        """Drop children whose files are gone. *known_chat_ids* were just read, so they skip the lookup."""
        pruned: list[dict[str, Any]] = []
        for child in children:
            chat_id = child.get("chat_id", "")
            if not chat_id:
                continue
            if chat_id in known_chat_ids:
                pruned.append(child)
                continue
            child_file = self.find_by_chat_id(chat_id)
            if child_file and self.resolve_path(child_file).exists():
                pruned.append(child)
        return pruned

    def _remove_child_from_parent(self, parent_file: str, child_chat_id: str) -> None:
        parent_path = self.resolve_path(parent_file)
        parent_data = self._load_path(parent_path)
        if parent_data:
            parent_data["children"] = [c for c in parent_data.get("children", []) if c.get("chat_id") != child_chat_id]
            safe_write_json(parent_path, parent_data)

    def _remove_from_project(self, project_dir: DataStorePath | None, filename: str) -> None:
        if project_dir is None:
            return
        proj_data_path = project_dir / PROJECT_JSON
        proj_data = safe_read_json(proj_data_path, {"name": project_dir.name, "kanban": [], "tabs": []})
        proj_data["tabs"] = [t for t in proj_data.get("tabs", []) if t.get("filename") != filename]
        safe_write_json(proj_data_path, proj_data)

    def cascade_delete(self, filename: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, Any]:
        """Delete a root conversation and all its descendants."""
        root_path = self.resolve_path(filename)
        if not root_path.exists():
            raise FileNotFoundError(filename)
        to_delete: list[DataStorePath] = [root_path]
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
            project_dir = self._release_chat(path, cleanup_uploads_fn)
            deleted.append(str(path.relative_to(self._base)))
            if path.exists():
                path.unlink()
            self._remove_from_project(project_dir, path.name)
        self.invalidate_index()
        return {"status": "ok", "deleted": deleted}

    def move(self, filename: str, target_dir: str) -> dict[str, str]:
        """Move a chat into project *target_dir* (empty for root), carrying a root chat's descendants. Name clashes get a numeric suffix."""
        src = self.resolve_path(filename)
        if not src.exists():
            raise FileNotFoundError(filename)
        src_data = self._load_path(src) or {}
        if target_dir and not (self._base / target_dir / PROJECT_JSON).exists():
            raise ValueError(f"'{target_dir}' is not a project")
        dst_dir = self._base / target_dir if target_dir else self._base

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
                child_target = dst_dir / child_src.name
                if child_target != child_src:
                    child_target = dst_dir / _unique_filename(dst_dir, child_src.name)
                safe_write_json(child_target, child_payload)
                if child_target != child_src:
                    child_src.unlink()

        dst = dst_dir / _unique_filename(dst_dir, src.name)
        new_payload = _build_payload(src_data)
        safe_write_json(dst, new_payload)
        src.unlink()
        self.invalidate_index()
        rel = dst.relative_to(self._base)
        return {"status": "ok", "new_path": str(rel)}

    def branch(self, parent_file: str, branch_message_idx: int) -> dict[str, Any]:
        """Copy history through pair *branch_message_idx* into a new child chat and link it from the parent."""
        parent_path = self.resolve_path(parent_file)
        if not parent_path.exists():
            raise FileNotFoundError(parent_file)
        parent_data = self._load_path(parent_path)
        if parent_data is None:
            raise ValueError("Failed to read parent chat history")
        parent_chat_id = parent_data.get("chat_id")
        message_count = _qa_count(parent_data.get("messages", []))
        if branch_message_idx < 0 or branch_message_idx > message_count:
            raise ValueError(f"branch_message_idx {branch_message_idx} out of range")
        # Keep pairs 0..idx inclusive, two messages each.
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

    def merge(self, child_file: str, *, cleanup_uploads_fn: Callable[[str, str | None], None] | None = None) -> dict[str, Any]:
        """Fold a child branch back into its parent, re-parent its own children, and delete it."""
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
        parent_message_count = _qa_count(parent_data.get("messages", []))
        # The parent may hold at most the branch-point pair, otherwise it advanced since branching.
        if parent_message_count > branch_idx + 1:
            raise ValueError("Parent has diverged past the branch point")
        child_messages = child_data.get("messages", [])
        # Child messages from the branch point on are new to the parent. Each pair is two messages.
        new_messages = child_messages[branch_idx * 2 :] if branch_idx * 2 < len(child_messages) else []
        parent_data["messages"] = parent_data.get("messages", []) + new_messages
        parent_children = parent_data.get("children", [])
        child_chat_id = child_data.get("chat_id")
        # Children of the merged child move up to the parent, after the pairs the parent already has.
        for sub_child in child_data.get("children", []):
            sub_file = self.find_by_chat_id(sub_child.get("chat_id", ""))
            if sub_file:
                sub_path = self.resolve_path(sub_file)
                sub_data = self._load_path(sub_path)
                if sub_data:
                    sub_data["parent_id"] = parent_data.get("chat_id")
                    sub_data["branch_message_idx"] = branch_idx + parent_message_count
                    safe_write_json(sub_path, sub_data)
                    parent_children.append(
                        {
                            "chat_id": sub_child.get("chat_id", ""),
                            "branch_message_idx": branch_idx + parent_message_count,
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

    def list_histories(self) -> dict[str, list[str]]:
        """List every chat outside a project: a history is simply a chat with no project."""
        # One listing query instead of an is_file/is_dir round trip per entry.
        prefix = f"{self._base.key}/"
        entries = {e.key.removeprefix(prefix): e.is_dir for e in self._base.store.list(self._base.key, recursive=True)}
        project_dirs = {rel.removesuffix(f"/{PROJECT_JSON}") for rel in entries if rel.endswith(f"/{PROJECT_JSON}") and rel.count("/") == 1}
        files: list[str] = []
        for rel in sorted(entries):
            *dirs, name = rel.split("/")
            if entries[rel] or PurePosixPath(name).suffix != ".json" or name in (PROJECT_JSON, META_JSON):
                continue
            if any(d.startswith("_") or d == MEMORY_DIR for d in dirs) or (dirs and dirs[0] in project_dirs):
                continue
            files.append(rel)
        return {"files": files}

    def get_branch_meta(self, dirs: str | None = None) -> dict[str, dict[str, Any]]:
        """Tree metadata for each chat file. *dirs* is comma-separated project dirs (empty entry for root), None scans all."""
        if not self._base.exists():
            return {}
        if dirs is None:
            return self._scan_dir_for_meta(self._base)
        result: dict[str, dict[str, Any]] = {}
        for d in dirs.split(","):
            d = d.strip()
            target = self._base / d if d else self._base
            if not target.exists():
                continue
            prefix = f"{d}/" if d else ""
            result.update(self._scan_dir_for_meta(target, prefix))
        return result

    def _scan_dir_for_meta(self, base: DataStorePath, rel_prefix: str = "") -> dict[str, dict[str, Any]]:
        """Per-chat tree metadata keyed by path. Prunes dead children from chat files as a side effect."""
        # One recursive listing, one read per chat. Chat IDs read here vouch for their own
        # existence, so pruning only falls back to the full index for genuinely missing children.
        base_prefix = f"{base.key}/"
        loaded: dict[str, tuple[DataStorePath, dict[str, Any] | None]] = {}
        for entry in sorted(base.store.list(base.key, recursive=True), key=lambda e: e.key):
            if entry.is_dir:
                continue
            rel = PurePosixPath(entry.key.removeprefix(base_prefix))
            *dirs, name = rel.parts
            if name in (PROJECT_JSON, META_JSON) or name.startswith("_") or rel.suffix != ".json":
                continue
            if any(d.startswith("_") or d == MEMORY_DIR for d in dirs):
                continue
            path = DataStorePath(base.store, entry.key)
            loaded[f"{rel_prefix}{rel}"] = (path, self._load_path(path))
        known = {data["chat_id"] for _, data in loaded.values() if data and data.get("chat_id")}
        return {key: self._extract_meta(path, data, known) for key, (path, data) in loaded.items()}

    def _extract_meta(self, path: DataStorePath, data: dict[str, Any] | None, known_chat_ids: set[str]) -> dict[str, Any]:
        if data is None:
            return {}
        children = self._prune_children_list(data.get("children", []), known_chat_ids)
        if children != data.get("children", []):
            data["children"] = children
            safe_write_json(path, data)
        qa_count = _qa_count(data.get("messages", []))
        return {
            "chat_id": data.get("chat_id"),
            "parent_id": data.get("parent_id"),
            "branch_message_idx": data.get("branch_message_idx"),
            "children": children,
            "qa_count": qa_count,
        }

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


def _normalize_chat(raw: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Shape a stored chat (legacy bare message list or object) into the canonical fields."""
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


def _unique_filename(directory: DataStorePath, base: str) -> str:
    """Return *base*, or the first free `stem-N.suffix` (N from 2) if it already exists."""
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


def _qa_count(messages: list[dict[str, Any]]) -> int:
    """Number of user/assistant pairs: one per user message, answered or not."""
    return sum(m.get("role") == "user" for m in messages)


def _next_branch_filename(parent_path: DataStorePath, parent_file: str, base_dir: DataStorePath) -> str:
    """Pick the next free `<parent stem>-NN.json` name beside the parent, keeping its directory prefix."""
    if "/" in parent_file:
        dir_prefix = parent_file.rsplit("/", 1)[0]
    elif parent_path.parent != base_dir:
        dir_prefix = parent_path.parent.name
    else:
        dir_prefix = None

    # Root `foo.json` -> `foo-00.json`, child `foo-00.json` -> `foo-00-00.json`. Lowest free number wins.
    parent_stem = parent_path.stem
    sibling_pattern = re.compile(r"^" + re.escape(parent_stem) + r"-(\d\d)$")
    existing_nums = {int(m.group(1)) for f in parent_path.parent.glob("*.json") if (m := sibling_pattern.match(f.stem))}
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
    """Merge chat fields, preferring *data*, then *existing*. *title* overrides both."""
    data = data or {}
    existing = existing or {}
    if data.get("messages"):
        messages = data["messages"]
    elif existing.get("messages") is not None:
        messages = existing["messages"]
    else:
        messages = data.get("messages", [])
    payload: dict[str, Any] = {"messages": messages}

    # Falsy values never win: the first truthy of override, data, existing is kept.
    for key, override in (("chat_id", None), ("title", title), ("usage", None)):
        value = override or data.get(key) or existing.get(key)
        if value:
            payload[key] = value

    # Only None is "unset" here, so a falsy value in data still replaces the stored one.
    for key in ("parent_id", "branch_message_idx", "children"):
        if data.get(key) is not None:
            payload[key] = data[key]
        elif key in existing:
            payload[key] = existing[key]

    return payload
