from uuid import uuid4

from backend.routes import deps
from lib.changes import UserScope
from lib.prompt_store import seed_prompts


def test_prompt_seed_preserves_edits_and_deletions(data_store):
    seed_prompts(data_store)
    prompt_key = next(
        entry.key for entry in data_store.list("prompt", recursive=True) if entry.key.endswith(".md")
    )

    _, revision = data_store.read_bytes(prompt_key)
    data_store.write_bytes(prompt_key, b"User edit", expected=revision)
    seed_prompts(data_store)
    assert data_store.read_bytes(prompt_key)[0] == b"User edit"

    data_store.delete(prompt_key)
    seed_prompts(data_store)
    assert not data_store.exists(prompt_key)


def test_prompt_store_dependency_seeds_once_per_user(data_store, monkeypatch):
    seeded = []
    monkeypatch.setattr(deps, "seed_prompts", lambda store: seeded.append(store))
    monkeypatch.setattr(deps, "_seeded_prompt_users", set())
    user, other = UserScope(None, uuid4()), UserScope(None, uuid4())  # type: ignore[arg-type]

    for scope in (user, user, other, user):
        deps.get_prompt_store(scope, data_store)

    assert seeded == [data_store, data_store]
