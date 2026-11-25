"""One-time move of every login's old memories (the Postgres `memory/*` documents) into its memory-store root.

    .venv/bin/python scripts/migrate_memory.py --host-login <login> [--dry-run] [--delete-old] [--database-url URL]

Run it from the checkout whose roots it should fill: the family roots follow that checkout (`memory_context.memory_root`),
so for production run it from the production checkout with the production environment.

- Global topics become memories of the memory_store `user`, a project's topics memories of the project's memory_store
  (`memory_context.project_memory_store`). One memory per topic (a name the store cannot take is slugged), one fact per
  old memory, the old relevance order kept as recency (the most relevant fact is the newest).
- Never loses a fact: an existing memory only gets the facts whose text it lacks, so a second run writes nothing.
- `--delete-old` deletes an old fact document (with its topics document) only when every one of its facts is confirmed
  in the new root, and never a document it could not parse or does not know; it reports what it kept.
- `--dry-run` prints the plan and changes nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import sys
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_store import store  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

from lib.changes import UserScope  # noqa: E402
from lib.chat_store import ChatStore  # noqa: E402
from lib.db_schema import database_url  # noqa: E402
from lib.json_docs import JsonDocs  # noqa: E402
from lib.memory_context import USER_MEMORY_STORE, memory_root, project_memory_store  # noqa: E402
from lib.postgres_data_store import PostgresDataStore  # noqa: E402
from lib.project_store import ProjectStore  # noqa: E402
from lib.storage_namespace import MEMORY  # noqa: E402

if TYPE_CHECKING:
    from uuid import UUID

# The old store's layout (`lib/memory_store.py` before memory-store replaced it).
GLOBAL_TOPICS_KEY = f"{MEMORY}/global-topics.json"
GLOBAL_FACTS_KEY = f"{MEMORY}/global-profile.json"
PROJECTS_PREFIX = f"{MEMORY}/project"
# Bookkeeping of the old extraction, no facts: deleted with the fact documents.
WATERMARKS_KEY = f"{MEMORY}/watermarks.json"
DEFAULT_GLOBAL_TOPICS = [
    {"name": "profile", "summary": "Who the user is: background, education, work, and setup."},
    {"name": "preferences", "summary": "How the user wants the assistant to respond and work."},
]
FALLBACK_TOPIC = "note"
SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
MISSING_CODES = ("not_found", "memory_store_missing")
_MISSING = object()


class Unparsable(Exception):
    """An old document that is not the shape the old store wrote."""


@dataclass(frozen=True)
class PlannedMemory:
    memory_store: str
    path: str
    description: str
    facts: list[str]  # most relevant first


@dataclass(frozen=True)
class OldDocument:
    """One old fact document, its topics document and the memories its facts go to."""

    facts_key: str
    topics_key: str
    memories: list[PlannedMemory]


@dataclass
class LoginPlan:
    login: str
    root: Path
    documents: list[OldDocument] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)  # old documents that stay, with the reason
    has_old_documents: bool = False


def memory_path(topic: str) -> str:
    """A topic name as one memory path segment: kept when the store takes it as a memory, else slugged."""
    if SEGMENT.fullmatch(topic) and not topic.endswith(".md"):
        return topic
    return re.sub(r"[^A-Za-z0-9_-]+", "-", topic.strip().lower()).strip("-") or FALLBACK_TOPIC


def planned_memories(memory_store: str, topics: object, facts: object) -> list[PlannedMemory]:
    """One memory per topic that has facts, its facts in their stored (relevance) order; raises Unparsable."""
    if not isinstance(topics, list) or not isinstance(facts, list):
        raise Unparsable("not a list")
    summaries = {str(t["name"]): str(t.get("summary") or "") for t in topics if isinstance(t, dict) and "name" in t}
    grouped: dict[str, tuple[str, list[str]]] = {}
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("text"), str):
            raise Unparsable(f"a fact without text: {fact!r}")
        text = fact["text"].strip()
        if not text:
            continue
        topic = str(fact.get("topic") or FALLBACK_TOPIC)
        description = summaries.get(topic) or topic
        grouped.setdefault(memory_path(topic), (description, []))[1].append(text)
    return [PlannedMemory(memory_store, path, description, texts) for path, (description, texts) in grouped.items()]


def usable_memory_store(name: str | None) -> str | None:
    """*name* when chats can address it as a project's memory_store (`memory_context.memory_prompt`), else None."""
    return name if name and SEGMENT.fullmatch(name) else None


def _read(docs: PostgresDataStore, key: str, default: object) -> object:
    """The parsed document at *key*, *default* when there is none; raises Unparsable for malformed JSON."""
    value = JsonDocs(docs).read(key, _MISSING)
    if value is _MISSING:
        if docs.exists(key):
            raise Unparsable("malformed JSON")
        return default
    return value


def login_plan(scope: UserScope, login: str, root: Path) -> LoginPlan:
    docs = PostgresDataStore(scope)
    plan = LoginPlan(login, root, has_old_documents=docs.exists(MEMORY))
    if not plan.has_old_documents:
        return plan
    projects = {p["slug"]: p for p in ProjectStore(ChatStore(data_store=docs), rename=_no_rename).list_projects()}
    sources = [(USER_MEMORY_STORE, GLOBAL_FACTS_KEY, GLOBAL_TOPICS_KEY, DEFAULT_GLOBAL_TOPICS)]
    for entry in docs.list(PROJECTS_PREFIX) if docs.is_dir(PROJECTS_PREFIX) else []:
        if entry.is_dir:
            slug = PurePosixPath(entry.key).name
            wanted = project_memory_store(projects.get(slug, {}), slug)
            sources.append((wanted or "", f"{entry.key}/memory/memory.json", f"{entry.key}/memory/topics.json", []))
    known = {WATERMARKS_KEY}
    for memory_store, facts_key, topics_key, default_topics in sources:
        known |= {facts_key, topics_key}
        try:
            memories = planned_memories(memory_store, _read(docs, topics_key, default_topics), _read(docs, facts_key, []))
        except Unparsable as exc:
            plan.kept += [f"{key}: could not parse ({exc})" for key in (facts_key, topics_key) if docs.exists(key)]
            continue
        if memories and usable_memory_store(memory_store) is None:
            reason = f"{memory_store!r} is not a usable memory_store name, {len(memories)} memories left"
            plan.kept += [f"{key}: {reason}" for key in (facts_key, topics_key) if docs.exists(key)]
            continue
        plan.documents.append(OldDocument(facts_key, topics_key, memories))
    plan.kept += [f"{e.key}: not a memory document" for e in docs.list(MEMORY, recursive=True) if not e.is_dir and e.key not in known]
    return plan


def _no_rename(*_: str) -> None:
    raise RuntimeError("the memory migration never renames projects")


def missing_facts(root: Path, memory: PlannedMemory) -> tuple[list[str], bool]:
    """The planned facts whose text the target memory lacks, and whether the memory exists; raises MemoryStoreError
    when the target cannot be read (corrupt, not a memory)."""
    try:
        present = {fact["text"] for fact in store.read_memory(memory.memory_store, memory.path, root=root)["facts"]}
    except store.MemoryStoreError as exc:
        if exc.code not in MISSING_CODES:
            raise
        return list(memory.facts), False
    return [text for text in memory.facts if text not in present], True


def write(root: Path, memory: PlannedMemory, texts: list[str], *, exists: bool) -> None:
    """Add *texts*, least relevant first, so the most relevant ends up the newest; a new memory gets the description."""
    store.create_memory_store(memory.memory_store, root=root)
    described = {} if exists else {"description": memory.description}
    ops = [{"op": "add", "path": memory.path, "text": text, **described} for text in reversed(texts)]
    failed = [r for r in store.apply_memory_ops(memory.memory_store, ops, root=root) if not r["ok"]]
    if failed:
        raise store.MemoryStoreError(failed[0]["error"], failed[0]["message"])


def migrate_memory(root: Path, memory: PlannedMemory, *, dry_run: bool, out: Any) -> bool:  # noqa: ANN401 - a text stream
    """Add the facts *memory* lacks in *root*; True when all its facts are (or, with *dry_run*, would be) present."""
    target = f"{memory.memory_store}/{memory.path}"
    try:
        texts, exists = missing_facts(root, memory)
        if not texts:
            print(f"  = {target} complete", file=out)
            return True
        print(f"  + {target} — {memory.description} ({len(texts)} of {len(memory.facts)} facts)", file=out)
        if dry_run:
            return True
        write(root, memory, texts, exists=exists)
        return not missing_facts(root, memory)[0]
    except store.MemoryStoreError as exc:
        print(f"  ! {target}: {exc.code}: {exc.message}", file=out)
        return False


def migrate(pool: ConnectionPool[Any], host_login: str, *, dry_run: bool, delete_old: bool, out: Any = sys.stdout) -> list[LoginPlan]:  # noqa: ANN401 - a text stream
    """Migrate (or with *dry_run* only print) every login's old memories; *delete_old* then drops the documents fully migrated."""
    with pool.connection() as connection:
        users: list[tuple[UUID, str]] = connection.execute("SELECT id, tailscale_login FROM users ORDER BY tailscale_login").fetchall()
    if host_login not in {login for _, login in users}:
        raise SystemExit(f"no login {host_login!r} in the database")
    plans = []
    for user_id, login in users:
        scope = UserScope(pool, user_id)
        plan = login_plan(scope, login, memory_root(user_id, login == host_login))
        plans.append(plan)
        if not plan.has_old_documents:
            continue
        print(f"{login} -> {plan.root}", file=out)
        migrated = []
        for document in plan.documents:
            # Every memory is migrated, also after one failed; the document goes only when all succeeded.
            results = [migrate_memory(plan.root, memory, dry_run=dry_run, out=out) for memory in document.memories]
            if all(results):
                migrated.append(document)
            else:
                plan.kept.append(f"{document.facts_key}: some facts are not in the new root")
        if delete_old:
            delete_migrated(PostgresDataStore(scope), migrated, dry_run=dry_run, out=out)
        for reason in plan.kept:
            print(f"  keep {reason}", file=out)
    return plans


def delete_migrated(docs: PostgresDataStore, migrated: list[OldDocument], *, dry_run: bool, out: Any) -> None:  # noqa: ANN401 - a text stream
    """Delete the migrated fact and topics documents and the watermarks; `memory/` itself once nothing is left in it."""
    keys = [key for document in migrated for key in (document.facts_key, document.topics_key)] + [WATERMARKS_KEY]
    for key in keys:
        if docs.exists(key):
            print(f"  - delete {key}", file=out)
            if not dry_run:
                docs.delete(key)
    if not dry_run and not any(not entry.is_dir for entry in docs.list(MEMORY, recursive=True)):
        docs.delete(MEMORY, recursive=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host-login", required=True, help="the login whose memories go to the shared host root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-old", action="store_true")
    parser.add_argument("--database-url")
    args = parser.parse_args()
    with ConnectionPool(database_url(args.database_url), min_size=1, max_size=2) as pool:
        migrate(pool, args.host_login, dry_run=args.dry_run, delete_old=args.delete_old)


if __name__ == "__main__":
    main()
