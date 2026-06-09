import json
from pathlib import Path
import re
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.routes.files import delete_chat_upload_dir
from config import DIRECTORY_CHAT_HISTORIES
from lib.chat_index import find_file_by_chat_id, invalidate_chat_id_index
from lib.json_io import safe_read_json
from lib.payload import _build_payload

router = APIRouter(prefix="/api/chat-histories", tags=["histories"])


class SaveRequest(BaseModel):
    messages: list[dict[str, Any]] = []
    chat_id: str | None = None
    title: str | None = None
    usage: dict[str, int] | None = None
    parent_id: str | None = None
    branch_message_idx: int | None = None
    children: list[dict[str, Any]] | None = None


class RenameRequest(BaseModel):
    old_path: str
    new_title: str


class MkdirRequest(BaseModel):
    parent_path: str
    name: str


class MoveRequest(BaseModel):
    filename: str
    target_dir: str


class BranchRequest(BaseModel):
    parent_file: str
    branch_message_idx: int


class MergeRequest(BaseModel):
    child_file: str


PROJECT_JSON = "project.json"
META_JSON = "projects-meta.json"


def _sanitize_title(title: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._\- ]+", "_", title).strip()
    return sanitized or "untitled"


def _unique_filename(directory: Path, base: str) -> str:
    """Return a unique filename in `directory` for a given base like 'foo.json'.
    Disambiguates with '-2', '-3', ... suffixes.
    """
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


def _project_dir_for_path(path: Path) -> Path | None:
    """Return the project directory containing `path`, or None if at root."""
    try:
        rel = path.relative_to(DIRECTORY_CHAT_HISTORIES.resolve())
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    candidate = DIRECTORY_CHAT_HISTORIES / rel.parts[0]
    if candidate.is_dir() and (candidate / PROJECT_JSON).exists():
        return candidate
    return None


def _load_chat_file(path: Path) -> dict[str, Any] | None:
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


@router.get("")
async def list_chat_histories() -> dict[str, Any]:
    if not DIRECTORY_CHAT_HISTORIES.exists():
        return {"files": [], "histories": {}}

    root_files: list[str] = []
    histories: dict[str, list[str]] = {}

    if DIRECTORY_CHAT_HISTORIES.is_dir():
        for f in sorted(DIRECTORY_CHAT_HISTORIES.iterdir()):
            if f.name in (PROJECT_JSON, META_JSON):
                continue
            if f.is_dir() and f.name in ("_uploads",):
                continue
            if f.is_file() and f.suffix == ".json":
                root_files.append(f.name)
            elif f.is_dir() and not (f / PROJECT_JSON).exists():
                histories.setdefault(f.name, [])
                for sf in sorted(f.iterdir()):
                    if sf.is_file() and sf.suffix == ".json" and sf.name != PROJECT_JSON:
                        histories[f.name].append(sf.name)

    return {"files": root_files, "histories": histories}


def _extract_meta(path: Path) -> dict[str, Any]:
    data = _load_chat_file(path)
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


def _scan_dir_for_meta(base: Path, rel_prefix: str = "") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for f in sorted(base.iterdir()):
        if f.name in (PROJECT_JSON, META_JSON):
            continue
        if f.name.startswith("_"):
            continue
        if f.is_file() and f.suffix == ".json":
            key = f"{rel_prefix}{f.name}" if rel_prefix else f.name
            result[key] = _extract_meta(f)
        elif f.is_dir() and f.name != "_uploads":
            sub_prefix = f"{f.name}/" if not rel_prefix else f"{rel_prefix}{f.name}/"
            result.update(_scan_dir_for_meta(f, sub_prefix))
    return result


@router.get("/branch-meta")
async def get_branch_meta(dirs: str | None = None) -> dict[str, dict[str, Any]]:
    if not DIRECTORY_CHAT_HISTORIES.exists():
        return {}
    if dirs is None:
        return _scan_dir_for_meta(DIRECTORY_CHAT_HISTORIES)
    result: dict[str, dict[str, Any]] = {}
    for d in dirs.split(","):
        d = d.strip()
        target = (DIRECTORY_CHAT_HISTORIES / d).resolve() if d else DIRECTORY_CHAT_HISTORIES.resolve()
        if not str(target).startswith(str(DIRECTORY_CHAT_HISTORIES.resolve())):
            continue
        if not target.exists():
            continue
        prefix = f"{d}/" if d else ""
        result.update(_scan_dir_for_meta(target, prefix))
    return result


@router.get("/{filename:path}")
async def load_chat_history(filename: str) -> dict[str, Any]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    data = _load_chat_file(filepath)
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to read chat history")
    return {**data, "filename": filename}


@router.put("/{filename:path}")
async def save_chat_history(filename: str, data: SaveRequest | None = None) -> dict[str, str]:
    DIRECTORY_CHAT_HISTORIES.mkdir(parents=True, exist_ok=True)
    filepath = _resolve_history_path(filename)

    # Prevent cross-chat overwrites: if the file exists and has a chat_id,
    # the incoming chat_id must match.
    existing = _load_chat_file(filepath) if filepath.exists() else None
    if existing and existing.get("chat_id") and data and data.chat_id and existing["chat_id"] != data.chat_id:
        raise HTTPException(
            status_code=409,
            detail="Chat ID mismatch: cannot overwrite an existing chat history with a different chat_id",
        )

    payload = _build_payload(data.model_dump() if data else None, existing)
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    invalidate_chat_id_index()
    return {"status": "ok", "filename": str(filename)}


@router.delete("/cascade/{filename:path}")
async def cascade_delete(filename: str) -> dict[str, Any]:
    root_path = _resolve_history_path(filename)
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")

    deleted: list[str] = []

    to_delete = [root_path]
    visited: set[str] = set()

    queue = [root_path]
    while queue:
        current = queue.pop(0)
        current_str = str(current)
        if current_str in visited:
            continue
        visited.add(current_str)

        current_data = _load_chat_file(current)
        if current_data:
            for child_info in current_data.get("children", []):
                child_file = find_file_by_chat_id(child_info.get("chat_id", ""))
                if child_file:
                    child_path = _resolve_history_path(child_file)
                    if child_path.exists() and str(child_path) not in visited:
                        queue.append(child_path)
                        to_delete.append(child_path)

    for path in to_delete:
        data = _load_chat_file(path)
        if data and data.get("chat_id"):
            slug = _slug_for_path(path)
            delete_chat_upload_dir(data["chat_id"], slug)

        rel = str(path.resolve().relative_to(DIRECTORY_CHAT_HISTORIES.resolve()))
        deleted.append(rel)
        if path.exists():
            path.unlink()

    invalidate_chat_id_index()
    return {"status": "ok", "deleted": deleted}


@router.delete("/orphan/{filename:path}")
async def orphan_children(filename: str) -> dict[str, Any]:
    root_path = _resolve_history_path(filename)
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")

    root_data = _load_chat_file(root_path)
    if root_data is None:
        raise HTTPException(status_code=500, detail="Failed to read chat history")

    orphaned: list[str] = []

    for child_info in root_data.get("children", []):
        child_file = find_file_by_chat_id(child_info.get("chat_id", ""))
        if child_file:
            child_path = _resolve_history_path(child_file)
            child_data = _load_chat_file(child_path)
            if child_data:
                child_data["parent_id"] = None
                child_data["branch_message_idx"] = None
                child_path.write_text(json.dumps(child_data, indent=2, ensure_ascii=False), encoding="utf-8")
                orphaned.append(child_file)

    root_slug = _slug_for_path(root_path)
    if root_data.get("chat_id"):
        delete_chat_upload_dir(root_data["chat_id"], root_slug)
    root_path.unlink()

    invalidate_chat_id_index()
    return {"status": "ok", "orphaned": orphaned}


@router.delete("/{filename:path}")
async def delete_chat_history(filename: str) -> dict[str, str]:
    filepath = _resolve_history_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    if filepath.is_dir():
        shutil.rmtree(str(filepath))
        return {"status": "ok"}
    data = _load_chat_file(filepath)
    project_dir = _project_dir_for_path(filepath)
    slug = project_dir.name if project_dir else None
    if data and data.get("chat_id"):
        delete_chat_upload_dir(data["chat_id"], slug)

        if data.get("parent_id"):
            parent_file = find_file_by_chat_id(data["parent_id"])
            if parent_file:
                parent_path = _resolve_history_path(parent_file)
                parent_data = _load_chat_file(parent_path)
                if parent_data:
                    parent_children = [c for c in parent_data.get("children", []) if c.get("chat_id") != data["chat_id"]]
                    parent_data["children"] = parent_children
                    parent_path.write_text(json.dumps(parent_data, indent=2, ensure_ascii=False), encoding="utf-8")

    filepath.unlink()
    invalidate_chat_id_index()
    if project_dir is not None:
        proj_data_path = project_dir / PROJECT_JSON
        proj_data = safe_read_json(proj_data_path, {"name": slug, "kanban": [], "tabs": []})
        proj_data["tabs"] = [t for t in proj_data.get("tabs", []) if t.get("filename") != filepath.name]
        proj_data_path.write_text(json.dumps(proj_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok"}


@router.post("/rename")
async def rename_chat_history(req: RenameRequest) -> dict[str, str]:
    old_path = _resolve_history_path(req.old_path)
    if not old_path.exists():
        raise HTTPException(status_code=404, detail="Chat history not found")
    project_dir = _project_dir_for_path(old_path)
    new_title = _sanitize_title(req.new_title)
    new_filename = f"{new_title}.json"
    new_dir = project_dir if project_dir is not None else DIRECTORY_CHAT_HISTORIES
    unique_name = _unique_filename(new_dir, new_filename)
    new_path = new_dir / unique_name
    data = _load_chat_file(old_path) or {}
    new_payload = _build_payload(data, title=new_title)
    new_path.write_text(json.dumps(new_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    old_path.unlink()
    if project_dir is not None:
        proj_data_path = project_dir / PROJECT_JSON
        proj_data = safe_read_json(proj_data_path, {"name": project_dir.name, "kanban": [], "tabs": []})
        for t in proj_data.get("tabs", []):
            if t.get("filename") == old_path.name:
                t["filename"] = unique_name
                t["name"] = new_title
                t["title"] = new_title
        proj_data_path.write_text(json.dumps(proj_data, indent=2, ensure_ascii=False), encoding="utf-8")
    rel = new_path.resolve().relative_to(DIRECTORY_CHAT_HISTORIES.resolve())
    invalidate_chat_id_index()
    return {"status": "ok", "new_path": str(rel), "filename": unique_name}


@router.post("/mkdir")
async def create_directory(req: MkdirRequest) -> dict[str, str]:
    parent = DIRECTORY_CHAT_HISTORIES / req.parent_path if req.parent_path else DIRECTORY_CHAT_HISTORIES
    dir_path = (parent / req.name).resolve()
    if not str(dir_path).startswith(str(DIRECTORY_CHAT_HISTORIES.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    dir_path.mkdir(parents=True, exist_ok=True)
    rel = dir_path.relative_to(DIRECTORY_CHAT_HISTORIES.resolve())
    return {"status": "ok", "path": str(rel)}


@router.post("/move")
async def move_history(req: MoveRequest) -> dict[str, str]:
    src = _resolve_history_path(req.filename)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    src_data = _load_chat_file(src) or {}
    dst_dir = (DIRECTORY_CHAT_HISTORIES / req.target_dir).resolve() if req.target_dir else DIRECTORY_CHAT_HISTORIES.resolve()
    if not str(dst_dir).startswith(str(DIRECTORY_CHAT_HISTORIES.resolve())):
        raise HTTPException(status_code=400, detail="Invalid target directory")
    dst_dir.mkdir(parents=True, exist_ok=True)

    is_root = src_data.get("parent_id") is None
    has_children = bool(src_data.get("children"))

    if is_root and has_children:
        children_files = _collect_descendants(src_data)
        for child_file in children_files:
            child_src = _resolve_history_path(child_file)
            if not child_src.exists():
                continue
            child_dst = dst_dir / child_src.name
            if child_src != child_dst:
                child_data = _load_chat_file(child_src) or {}
                child_payload = _build_payload(child_data)
                dst_dir.mkdir(parents=True, exist_ok=True)
                child_target = dst_dir / child_src.name
                n = 2
                while child_target.exists() and child_target != child_src:
                    child_target = dst_dir / f"{child_src.stem}-{n}{child_src.suffix}"
                    n += 1
                child_target.write_text(json.dumps(child_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                if child_target != child_src:
                    child_src.unlink()

    dst = dst_dir / src.name
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        n = 2
        while dst.exists():
            dst = dst_dir / f"{stem}-{n}{suffix}"
            n += 1
    new_payload = _build_payload(src_data)
    dst.write_text(json.dumps(new_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    src.unlink()
    invalidate_chat_id_index()
    rel = dst.relative_to(DIRECTORY_CHAT_HISTORIES.resolve())
    return {"status": "ok", "new_path": str(rel)}


@router.post("/branch")
async def create_branch(req: BranchRequest) -> dict[str, Any]:
    parent_path = _resolve_history_path(req.parent_file)
    if not parent_path.exists():
        raise HTTPException(status_code=404, detail="Parent chat history not found")

    parent_data = _load_chat_file(parent_path)
    if parent_data is None:
        raise HTTPException(status_code=500, detail="Failed to read parent chat history")

    parent_chat_id = parent_data.get("chat_id")

    message_pairs = _extract_qa_pairs(parent_data.get("messages", []))
    if req.branch_message_idx < 0 or req.branch_message_idx > len(message_pairs):
        raise HTTPException(
            status_code=400,
            detail=f"branch_message_idx {req.branch_message_idx} out of range (0-{len(message_pairs)})",
        )

    cutoff = min(req.branch_message_idx * 2 + 2, len(parent_data.get("messages", [])))
    child_messages = parent_data.get("messages", [])[:cutoff]

    child_chat_id = __import__("uuid").uuid4().hex[:12]
    child_filename = _next_branch_filename(parent_path, req.parent_file)
    child_path = _resolve_history_path(child_filename)

    children = parent_data.get("children", [])
    children.append({"chat_id": child_chat_id, "branch_message_idx": req.branch_message_idx})
    parent_data["children"] = children
    parent_path.write_text(json.dumps(parent_data, indent=2, ensure_ascii=False), encoding="utf-8")

    child_payload: dict[str, Any] = {
        "messages": child_messages,
        "parent_id": parent_chat_id,
        "branch_message_idx": req.branch_message_idx,
        "children": [],
    }
    if parent_chat_id:
        child_payload["chat_id"] = child_chat_id
    if parent_data.get("title"):
        child_payload["title"] = parent_data["title"]

    child_path.parent.mkdir(parents=True, exist_ok=True)
    child_path.write_text(json.dumps(child_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    invalidate_chat_id_index()
    return {"status": "ok", "child_file": child_filename, "chat_id": child_chat_id}


@router.post("/merge")
async def merge_branch(req: MergeRequest) -> dict[str, Any]:
    child_path = _resolve_history_path(req.child_file)
    if not child_path.exists():
        raise HTTPException(status_code=404, detail="Child chat history not found")

    child_data = _load_chat_file(child_path)
    if child_data is None:
        raise HTTPException(status_code=500, detail="Failed to read child chat history")

    parent_id = child_data.get("parent_id")
    if not parent_id:
        raise HTTPException(status_code=400, detail="Cannot merge a root conversation")

    parent_file = find_file_by_chat_id(parent_id)
    if parent_file is None:
        raise HTTPException(status_code=404, detail="Parent chat history not found")

    parent_path = _resolve_history_path(parent_file)
    parent_data = _load_chat_file(parent_path)
    if parent_data is None:
        raise HTTPException(status_code=500, detail="Failed to read parent chat history")

    branch_idx = child_data.get("branch_message_idx")
    if branch_idx is None:
        raise HTTPException(status_code=400, detail="Child has no branch_message_idx")

    parent_message_pairs = _extract_qa_pairs(parent_data.get("messages", []))
    if len(parent_message_pairs) > branch_idx + 1:
        raise HTTPException(status_code=409, detail="Parent has diverged past the branch point")

    child_messages = child_data.get("messages", [])
    new_messages = child_messages[branch_idx * 2 :] if branch_idx * 2 < len(child_messages) else []

    parent_data["messages"] = parent_data.get("messages", []) + new_messages

    parent_children = parent_data.get("children", [])
    child_chat_id = child_data.get("chat_id")

    for sub_child in child_data.get("children", []):
        sub_file = find_file_by_chat_id(sub_child.get("chat_id", ""))
        if sub_file:
            sub_path = _resolve_history_path(sub_file)
            sub_data = _load_chat_file(sub_path)
            if sub_data:
                sub_data["parent_id"] = parent_data.get("chat_id")
                sub_data["branch_message_idx"] = branch_idx + len(parent_message_pairs)
                sub_path.write_text(json.dumps(sub_data, indent=2, ensure_ascii=False), encoding="utf-8")
                parent_children.append(
                    {
                        "chat_id": sub_child.get("chat_id", ""),
                        "branch_message_idx": branch_idx + len(parent_message_pairs),
                    }
                )

    parent_children = [c for c in parent_children if c.get("chat_id") != child_chat_id]
    parent_data["children"] = parent_children
    parent_path.write_text(json.dumps(parent_data, indent=2, ensure_ascii=False), encoding="utf-8")

    child_slug = _slug_for_path(child_path)
    if child_data.get("chat_id"):
        delete_chat_upload_dir(child_data["chat_id"], child_slug)
    child_path.unlink()

    invalidate_chat_id_index()
    return {"status": "ok"}


def _slug_for_path(path: Path) -> str | None:
    pd = _project_dir_for_path(path)
    return pd.name if pd else None


def _extract_qa_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "user":
            pair = {"user": messages[i]}
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                pair["assistant"] = messages[i + 1]
                i += 2
            else:
                i += 1
            pairs.append(pair)
        else:
            i += 1
    return pairs


def _next_branch_filename(parent_path: Path, parent_file: str) -> str:
    parent_dir = parent_path.parent
    parent_stem = parent_path.stem

    if "/" in parent_file:
        dir_prefix = parent_file.rsplit("/", 1)[0]
    elif parent_path.resolve().parent != DIRECTORY_CHAT_HISTORIES.resolve():
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


def _collect_descendants(data: dict[str, Any]) -> list[str]:
    files: list[str] = []
    queue = list(data.get("children", []))
    while queue:
        child_info = queue.pop(0)
        child_file = find_file_by_chat_id(child_info.get("chat_id", ""))
        if child_file:
            files.append(child_file)
            child_path = _resolve_history_path(child_file)
            if child_path.exists():
                child_data = _load_chat_file(child_path)
                if child_data:
                    queue.extend(child_data.get("children", []))
    return files


def _resolve_history_path(filename: str) -> Path:
    resolved = (DIRECTORY_CHAT_HISTORIES / filename).resolve()
    if not str(resolved).startswith(str(DIRECTORY_CHAT_HISTORIES.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved
