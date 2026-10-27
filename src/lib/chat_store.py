"""Single seam for all chat-file I/O, validation, indexing, and cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from typing import TYPE_CHECKING, Any
import uuid

from lib.data_store import DataStore, Meta, Revision, validate_key
from lib.json_docs import JsonDocs
from lib.storage_namespace import CHAT, MEMORY, META_JSON, PROJECT_JSON, library_pdf, project_json

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lib.references import Released


class ParentDiverged(ValueError):
    """The parent chat gained pairs after the branch point, so a merge would interleave histories."""


@dataclass
class Chat:
    """A stored chat in its canonical fields. Legacy files hold a bare message list."""

    messages: list[dict[str, Any]]
    chat_id: str | None = None
    title: str | None = None
    usage: dict[str, Any] | None = None
    parent_id: str | None = None
    branch_message_idx: int | None = None
    children: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | list[Any]) -> "Chat":
        if isinstance(raw, list):
            return cls(messages=raw)
        return cls(
            messages=raw.get("messages", []),
            chat_id=raw.get("chat_id"),
            title=raw.get("title"),
            usage=raw.get("usage"),
            parent_id=raw.get("parent_id"),
            branch_message_idx=raw.get("branch_message_idx"),
            children=raw.get("children", []),
        )

    def as_dict(self) -> dict[str, Any]:
        """Every canonical field, in the stored key order."""
        return dict(vars(self))

    @property
    def qa_count(self) -> int:
        """Number of user/assistant pairs: one per user message, answered or not."""
        return sum(m.get("role") == "user" for m in self.messages)


@dataclass
class ChatTxn:
    """The writes and deletes of one tree operation, committed together."""

    writes: dict[str, dict[str, Any]] = field(default_factory=dict)
    deletes: list[str] = field(default_factory=list)


class ChatStore:
    """Owns all chat-file reads, writes, deletes, branching, merging, and indexing."""

    def __init__(self, *, data_store: DataStore) -> None:
        self._docs = JsonDocs(data_store)
        self._chat_id_index: dict[str, str] | None = None

    @property
    def docs(self) -> JsonDocs:
        """The documents this store reads and writes, shared with the project catalog."""
        return self._docs

    @staticmethod
    def _key(filename: str) -> str:
        """Storage key of *filename* under the chat root, rejecting traversal."""
        return validate_key(f"{CHAT}/{filename}")

    @staticmethod
    def _rel(key: str) -> str:
        return str(PurePosixPath(key).relative_to(CHAT))

    def _project_slug(self, key: str) -> str | None:
        """Slug of the project whose directory holds *key*, None for chats outside a project."""
        parts = PurePosixPath(key).relative_to(CHAT).parts
        if len(parts) < 2 or not self._docs.exists(project_json(parts[0])):
            return None
        return parts[0]

    def _read(self, key: str) -> Chat | None:
        """The chat at *key*, None when it is missing or unreadable."""
        raw = self._docs.read(key)
        return None if raw is None else Chat.from_raw(raw)

    def _find(self, chat_id: str) -> str | None:
        """Key of the chat with this ID, building the index on first use."""
        if self._chat_id_index is None:
            self._chat_id_index = {}
            # Listing order, so a duplicated ID resolves to the last chat listed.
            for key, meta in self._metas(CHAT, _is_chat_key).items():
                if meta and meta["chat_id"]:
                    self._chat_id_index[meta["chat_id"]] = key
        return self._chat_id_index.get(chat_id)

    def invalidate_index(self) -> None:
        self._chat_id_index = None

    def _reindex(self, writes: dict[str, dict[str, Any]], deletes: Sequence[str]) -> None:
        """Keep a built index in step with committed writes and deletes instead of rebuilding it."""
        if self._chat_id_index is None:
            return
        touched = set(deletes) | set(writes)
        self._chat_id_index = {chat_id: key for chat_id, key in self._chat_id_index.items() if key not in touched}
        for key, data in writes.items():
            if key not in deletes and data.get("chat_id") and _is_chat_key(key):
                self._chat_id_index[data["chat_id"]] = key

    def _commit(self, txn: ChatTxn) -> None:
        """Apply a tree operation in one transaction. Unguarded, so the last writer still wins."""
        meta = {key: m for key, data in txn.writes.items() if _is_chat_key(key) and (m := _tree_meta(data)) is not None}
        self._docs.apply(txn.writes.items(), txn.deletes, meta=meta)
        self._reindex(txn.writes, txn.deletes)

    def load(self, filename: str) -> dict[str, Any] | None:
        """Load a chat file and its revision in one read. Returns None on failure."""
        key = self._key(filename)
        try:
            raw, revision = self._docs.read_with_revision(key)
        except Exception:
            return None
        return {**Chat.from_raw(raw).as_dict(), "revision": revision.token}

    def save(
        self, filename: str, data: dict[str, Any] | None = None, *, title: str | None = None, expected_revision: str | None = None
    ) -> dict[str, str]:
        """Write a chat file, merging *data* over the stored chat. A given *expected_revision* makes the write conflict-checked."""
        key = self._key(filename)
        existing = self._read(key)
        if existing and existing.chat_id and data and data.get("chat_id") and existing.chat_id != data["chat_id"]:
            raise ValueError("Chat ID mismatch")
        payload = _build_payload(data, existing.as_dict() if existing else None, title=title)
        expected = None if expected_revision is None else Revision(expected_revision)
        meta = _tree_meta(payload) if _is_chat_key(key) else None
        revision = self._docs.write(key, payload, expected=expected, meta=meta)
        self._reindex({key: payload}, [])
        return {"status": "ok", "filename": str(filename), "revision": revision.token}

    def delete(self, filename: str, *, released: Released | None = None) -> dict[str, str]:
        """Delete a single chat file. Updates parent children array and project tab registry if applicable."""
        key = self._key(filename)
        if not self._docs.exists(key):
            raise FileNotFoundError(filename)
        nodes = {key: self._read(key)}
        if self._docs.is_dir(key):
            # A folder holds no chat of its own: it goes with everything under it.
            if released is not None:
                self.release_chats_under(key, self._project_slug(f"{key}/_"), released)
            self._docs.delete(key, recursive=True)
            self.invalidate_index()
        self._delete_nodes(nodes, released)
        return {"status": "ok"}

    def cascade_delete(self, filename: str, *, released: Released | None = None) -> dict[str, Any]:
        """Delete a root conversation and all its descendants."""
        root = self._key(filename)
        if not self._docs.exists(root):
            raise FileNotFoundError(filename)
        nodes: dict[str, Chat | None] = {}
        queue = [root]
        while queue:
            current = queue.pop(0)
            chat = nodes[current] = self._read(current)
            for child in chat.children if chat else []:
                child_key = self._find(child.get("chat_id", ""))
                if child_key and child_key not in nodes and child_key not in queue and self._docs.exists(child_key):
                    queue.append(child_key)
        self._delete_nodes(nodes, released)
        return {"status": "ok", "deleted": [self._rel(key) for key in nodes]}

    def release_chats_under(self, directory: str, slug: str | None, released: Released) -> None:
        """Record every chat stored under the storage key *directory* in *released*, read from meta, before it is deleted."""
        for meta in self._metas(directory, _is_chat_key).values():
            if meta is None:
                continue
            if isinstance(chat_id := meta["chat_id"], str) and chat_id:
                released.chat(chat_id, slug, meta["refs"])
            else:
                released.refs.update(meta["refs"])

    def _delete_nodes(self, nodes: dict[str, Chat | None], released: Released | None) -> None:
        """Delete *nodes*, detaching them from parents and project tabs in the same commit, and record them in *released*.

        Their uploads and derived data are cleaned up by the caller after this commit, never before.
        """
        txn = ChatTxn(deletes=list(nodes))
        for key, chat in nodes.items():
            slug = self._project_slug(key)
            if chat and released is not None:
                if chat.chat_id:
                    released.chat(chat.chat_id, slug, _attachment_refs(chat.messages))
                else:
                    released.refs.update(_attachment_refs(chat.messages))
            if chat and chat.chat_id:
                parent_key = self._find(chat.parent_id) if chat.parent_id else None
                # A parent that is deleted too needs no update.
                if parent_key and parent_key not in nodes:
                    self._stage_detach(txn, parent_key, chat.chat_id)
            if slug:
                self._stage_tab_removal(txn, slug, PurePosixPath(key).name)
        self._commit(txn)

    def _stage_detach(self, txn: ChatTxn, parent_key: str, child_chat_id: str) -> None:
        parent = txn.writes.get(parent_key)
        if parent is None:
            chat = self._read(parent_key)
            if chat is None:
                return
            parent = chat.as_dict()
        parent["children"] = [c for c in parent["children"] if c.get("chat_id") != child_chat_id]
        txn.writes[parent_key] = parent

    def _stage_tab_removal(self, txn: ChatTxn, slug: str, filename: str) -> None:
        key = project_json(slug)
        project = txn.writes.get(key)
        if project is None:
            stored = self._docs.read(key)
            project = stored if isinstance(stored, dict) else {"name": slug, "kanban": [], "tabs": []}
        project["tabs"] = [t for t in project.get("tabs", []) if t.get("filename") != filename]
        txn.writes[key] = project

    def _prune_children_list(self, children: list[dict[str, Any]], known_chat_ids: set[str]) -> list[dict[str, Any]]:
        """Drop children whose files are gone. *known_chat_ids* were just read, so they skip the lookup."""
        pruned: list[dict[str, Any]] = []
        for child in children:
            chat_id = child.get("chat_id", "")
            if not chat_id:
                continue
            if chat_id in known_chat_ids:
                pruned.append(child)
                continue
            child_key = self._find(chat_id)
            if child_key and self._docs.exists(child_key):
                pruned.append(child)
        return pruned

    def move(self, filename: str, target_dir: str) -> dict[str, str]:
        """Move a chat into project *target_dir* (empty for root) with a root chat's descendants. Name clashes get a numeric suffix."""
        src = self._key(filename)
        if not self._docs.exists(src):
            raise FileNotFoundError(filename)
        chat = self._read(src)
        if target_dir and not self._docs.exists(f"{self._key(target_dir)}/{PROJECT_JSON}"):
            raise ValueError(f"'{target_dir}' is not a project")
        dst_dir = self._key(target_dir) if target_dir else CHAT

        txn = ChatTxn()
        if chat and chat.parent_id is None and chat.children:
            for child_key, child in self._descendants(chat).items():
                target = f"{dst_dir}/{PurePosixPath(child_key).name}"
                # A descendant already in the target directory is rewritten in place.
                if target != child_key:
                    target = self._free_key(txn, dst_dir, PurePosixPath(child_key).name)
                    txn.deletes.append(child_key)
                txn.writes[target] = _build_payload(child.as_dict() if child else None)

        dst = self._free_key(txn, dst_dir, PurePosixPath(src).name)
        txn.writes[dst] = _build_payload(chat.as_dict() if chat else None)
        txn.deletes.append(src)
        self._commit(txn)
        return {"status": "ok", "new_path": self._rel(dst)}

    def _free_key(self, txn: ChatTxn, directory: str, name: str) -> str:
        """`directory/name`, or the first free `stem-N.suffix` (N from 2) when it is stored or already planned."""
        stem, suffix = PurePosixPath(name).stem, PurePosixPath(name).suffix
        candidate, n = f"{directory}/{name}", 2
        while candidate in txn.writes or self._docs.exists(candidate):
            candidate = f"{directory}/{stem}-{n}{suffix}"
            n += 1
        return candidate

    def branch(self, parent_file: str, branch_message_idx: int) -> dict[str, Any]:
        """Copy history through pair *branch_message_idx* into a new child chat and link it from the parent."""
        parent_key = self._key(parent_file)
        if not self._docs.exists(parent_key):
            raise FileNotFoundError(parent_file)
        parent = self._read(parent_key)
        if parent is None:
            raise ValueError("Failed to read parent chat history")
        if branch_message_idx < 0 or branch_message_idx > parent.qa_count:
            raise ValueError(f"branch_message_idx {branch_message_idx} out of range")
        # Keep pairs 0..idx inclusive, two messages each.
        cutoff = min(branch_message_idx * 2 + 2, len(parent.messages))
        child_chat_id = uuid.uuid4().hex[:12]
        child_filename = self._next_branch_filename(parent_key, parent_file)
        child_key = self._key(child_filename)
        parent.children.append({"chat_id": child_chat_id, "branch_message_idx": branch_message_idx})
        child: dict[str, Any] = {
            "messages": parent.messages[:cutoff],
            "parent_id": parent.chat_id,
            "branch_message_idx": branch_message_idx,
            "children": [],
        }
        if parent.chat_id:
            child["chat_id"] = child_chat_id
        if parent.title:
            child["title"] = parent.title
        # Branching has always stored an explicit row for the child's folder.
        self._docs.mkdir(str(PurePosixPath(child_key).parent))
        self._commit(ChatTxn(writes={parent_key: parent.as_dict(), child_key: child}))
        return {"status": "ok", "child_file": child_filename, "chat_id": child_chat_id}

    def merge(self, child_file: str, *, released: Released | None = None) -> dict[str, Any]:
        """Fold a child branch back into its parent, re-parent its own children, and delete it."""
        child_key = self._key(child_file)
        if not self._docs.exists(child_key):
            raise FileNotFoundError(child_file)
        child = self._read(child_key)
        if child is None:
            raise ValueError("Failed to read child chat history")
        if not child.parent_id:
            raise ValueError("Cannot merge a root conversation")
        parent_key = self._find(child.parent_id)
        if parent_key is None:
            raise FileNotFoundError("Parent chat history not found")
        parent = self._read(parent_key)
        if parent is None:
            raise ValueError("Failed to read parent chat history")
        branch_idx = child.branch_message_idx
        if branch_idx is None:
            raise ValueError("Child has no branch_message_idx")
        parent_message_count = parent.qa_count
        # The parent may hold at most the branch-point pair, otherwise it advanced since branching.
        if parent_message_count > branch_idx + 1:
            raise ParentDiverged("Parent has diverged past the branch point")
        # Child messages from the branch point on are new to the parent. Each pair is two messages.
        parent.messages = parent.messages + child.messages[branch_idx * 2 :]
        txn = ChatTxn(deletes=[child_key])
        # Children of the merged child move up to the parent, after the pairs the parent already has.
        sub_idx = branch_idx + parent_message_count
        for sub_ref in child.children:
            sub_key = self._find(sub_ref.get("chat_id", ""))
            sub = self._read(sub_key) if sub_key else None
            if sub:
                sub.parent_id = parent.chat_id
                sub.branch_message_idx = sub_idx
                txn.writes[sub_key] = sub.as_dict()
                parent.children.append({"chat_id": sub_ref.get("chat_id", ""), "branch_message_idx": sub_idx})
        parent.children = [c for c in parent.children if c.get("chat_id") != child.chat_id]
        txn.writes[parent_key] = parent.as_dict()
        if child.chat_id and released is not None:
            released.chat(child.chat_id, self._project_slug(child_key), _attachment_refs(child.messages))
        self._commit(txn)
        return {"status": "ok"}

    def list_histories(self) -> dict[str, list[str]]:
        """List every chat outside a project: a history is simply a chat with no project."""
        # One listing query instead of an is_file/is_dir round trip per entry.
        prefix = f"{CHAT}/"
        entries = {e.key.removeprefix(prefix): e.is_dir for e in self._docs.list(CHAT, recursive=True)}
        marker = f"/{PROJECT_JSON}"
        project_dirs = {rel.removesuffix(marker) for rel in entries if rel.count("/") == 1 and rel.endswith(marker)}
        files: list[str] = []
        for rel in sorted(entries):
            *dirs, name = rel.split("/")
            if entries[rel] or not _is_chat_file(name) or _in_hidden_dir(dirs) or (dirs and dirs[0] in project_dirs):
                continue
            files.append(rel)
        return {"files": files}

    def tree_metas(self) -> dict[str, Meta | None]:
        """Stored tree meta (chat ID and attachment references included) of every chat, without reading chat bodies."""
        return self._metas(CHAT, _is_chat_key)

    def _metas(self, prefix: str, select: Callable[[str], bool]) -> dict[str, Meta | None]:
        """Tree meta of the chats under *prefix* that *select* accepts. Meta stored before `refs` existed is derived again."""
        return self._docs.list_meta(prefix, select, _tree_meta, require="refs")

    def get_branch_meta(self, dirs: str | None = None) -> dict[str, dict[str, Any]]:
        """Tree metadata for each chat file. *dirs* is comma-separated project dirs (empty entry for root), None scans all."""
        if dirs is None:
            return self._scan_dir_for_meta(CHAT)
        result: dict[str, dict[str, Any]] = {}
        for d in dirs.split(","):
            d = d.strip()
            result.update(self._scan_dir_for_meta(self._key(d) if d else CHAT, f"{d}/" if d else ""))
        return result

    def _scan_dir_for_meta(self, root: str, rel_prefix: str = "") -> dict[str, dict[str, Any]]:
        """Per-chat tree metadata keyed by path. Prunes dead children from chat files as a side effect."""
        # One query for the metadata stored beside each chat, no chat bodies. Chat IDs read here vouch
        # for their own existence, so pruning only falls back to the full index for genuinely missing children.
        root_prefix = f"{root}/"

        def listed(key: str) -> bool:
            rel = PurePosixPath(key.removeprefix(root_prefix))
            return _is_chat_file(rel.name) and not rel.name.startswith("_") and not _in_hidden_dir(rel.parts[:-1])

        metas = self._metas(root, listed)
        known = {meta["chat_id"] for meta in metas.values() if meta and meta["chat_id"]}
        return {
            f"{rel_prefix}{PurePosixPath(key.removeprefix(root_prefix))}": self._extract_meta(key, metas[key], known)
            for key in sorted(metas)
        }

    def _extract_meta(self, key: str, meta: Meta | None, known_chat_ids: set[str]) -> dict[str, Any]:
        if meta is None:
            return {}
        children = self._prune_children_list(meta["children"], known_chat_ids)
        if children != meta["children"]:
            chat = self._read(key)
            if chat is not None:
                chat.children = self._prune_children_list(chat.children, known_chat_ids)
                payload = chat.as_dict()
                self._docs.write(key, payload, meta=_tree_meta(payload))
        return {
            "chat_id": meta["chat_id"],
            "parent_id": meta["parent_id"],
            "branch_message_idx": meta["branch_message_idx"],
            "children": children,
            "qa_count": meta["qa_count"],
        }

    def _descendants(self, chat: Chat) -> dict[str, Chat | None]:
        """Every stored descendant of *chat*, breadth first, with its contents (None when unreadable)."""
        found: dict[str, Chat | None] = {}
        queue = list(chat.children)
        while queue:
            child_key = self._find(queue.pop(0).get("chat_id", ""))
            if not child_key or child_key in found or not self._docs.exists(child_key):
                continue
            child = found[child_key] = self._read(child_key)
            if child:
                queue.extend(child.children)
        return found

    def _next_branch_filename(self, parent_key: str, parent_file: str) -> str:
        """Pick the next free `<parent stem>-NN.json` name beside the parent, keeping its directory prefix."""
        parent = PurePosixPath(parent_key)
        if "/" in parent_file:
            dir_prefix = parent_file.rsplit("/", 1)[0]
        elif parent.parent != PurePosixPath(CHAT):
            dir_prefix = parent.parent.name
        else:
            dir_prefix = None

        # Root `foo.json` -> `foo-00.json`, child `foo-00.json` -> `foo-00-00.json`. Lowest free number wins.
        sibling_pattern = re.compile(r"^" + re.escape(parent.stem) + r"-(\d\d)$")
        names = (PurePosixPath(entry.key).name for entry in self._docs.list(str(parent.parent)))
        stems = (PurePosixPath(name).stem for name in names if name.endswith(".json"))
        existing_nums = {int(m.group(1)) for stem in stems if (m := sibling_pattern.match(stem))}
        next_num = 0
        while next_num in existing_nums:
            next_num += 1
        name = f"{parent.stem}-{next_num:02d}.json"
        return f"{dir_prefix}/{name}" if dir_prefix else name


def _is_chat_file(name: str) -> bool:
    """Whether a file named *name* in the chat namespace is a chat (not the catalog or a project's state)."""
    return PurePosixPath(name).suffix == ".json" and name not in (PROJECT_JSON, META_JSON)


def _is_chat_key(key: str) -> bool:
    """Whether the file at storage *key* is a chat."""
    return _is_chat_file(PurePosixPath(key).name)


def _in_hidden_dir(dirs: Sequence[str]) -> bool:
    """Whether a folder on the path is private (`_` prefix) or the memory store's, both left out of chat listings."""
    return any(d.startswith("_") or d == MEMORY for d in dirs)


def _attachment_refs(messages: object) -> list[str]:
    """What the attachments of *messages* reference: each `vaultPath`, and the library copy `PDFs/<name>` of a PDF."""
    refs: set[str] = set()
    for message in messages if isinstance(messages, list) else []:
        attachments = message.get("attachments") if isinstance(message, dict) else None
        for attachment in attachments if isinstance(attachments, list) else []:
            if not isinstance(attachment, dict):
                continue
            if isinstance(path := attachment.get("vaultPath"), str) and path:
                refs.add(path)
            name = attachment.get("name")
            pdf = attachment.get("mime") == "application/pdf" or (isinstance(name, str) and name.lower().endswith(".pdf"))
            # A PDF upload is copied into the library, and a missing reference falls back to that copy.
            if pdf and isinstance(name, str) and name and "/" not in name:
                refs.add(library_pdf(name))
    return sorted(refs)


def _tree_meta(raw: object) -> Meta | None:
    """Tree fields and attachment references of a parsed chat, stored beside it. None when *raw* is not a chat."""
    if not isinstance(raw, (dict, list)):
        return None
    chat = Chat.from_raw(raw)
    try:
        qa_count = chat.qa_count
    except (AttributeError, TypeError):
        # Malformed messages: the chat still counts for the ID index, it just has no pair count.
        qa_count = None
    return {
        "chat_id": chat.chat_id,
        "parent_id": chat.parent_id,
        "branch_message_idx": chat.branch_message_idx,
        "children": chat.children,
        "qa_count": qa_count,
        "refs": _attachment_refs(chat.messages),
    }


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
