
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
