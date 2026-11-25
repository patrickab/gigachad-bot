"""`lib.memory_context`: the root each login uses and the memory block the chat's system prompt gets."""

import os
from pathlib import Path
from uuid import uuid4

from memory_store import store

from lib.memory_context import memory_prompt, memory_root, project_memory_store


def test_the_host_login_uses_the_memory_store_default_root() -> None:
    assert memory_root(uuid4(), True) == Path(os.environ["MEMORY_STORE_ROOT"])


def test_other_logins_get_their_own_folder_under_the_family_dir() -> None:
    user_id = uuid4()

    assert memory_root(user_id, False) == Path(os.environ["GIGACHAD_MEMORY_DIR"]) / str(user_id)


def test_tests_never_reach_the_real_roots() -> None:
    real = (Path.home() / ".pi" / "memory_store", Path(__file__).resolve().parents[1] / ".memory_store")

    assert not any(memory_root(uuid4(), host).is_relative_to(path) for host in (True, False) for path in real)


def test_two_logins_do_not_see_each_others_memories() -> None:
    first, second = memory_root(uuid4(), False), memory_root(uuid4(), False)
    memory_prompt(first, None)
    store.apply_memory_ops("user", [{"op": "add", "path": "profile", "description": "Who", "text": "Likes tea"}], root=first)

    assert "Likes tea" in memory_prompt(first, None)
    assert "Likes tea" not in memory_prompt(second, None)
    assert store.list_memories("user", root=second)["memories"] == []


def test_project_memory_store_is_the_linked_folder_name_else_the_slug() -> None:
    assert project_memory_store({"linked_root": "/home/me/git/shop"}, "shop-2") == "shop"
    assert project_memory_store({"linked_root": None}, "shop-2") == "shop-2"
    assert project_memory_store(None, None) is None


def test_a_project_named_user_gets_project_user_not_the_logins_user_store() -> None:
    assert project_memory_store({"linked_root": None}, "user") == "project-user"
    assert project_memory_store({"linked_root": "/home/me/user"}, "notes") == "project-user"


def test_memory_prompt_creates_the_stores_and_renders_pinned_memories_and_indexes(tmp_path: Path) -> None:
    first = memory_prompt(tmp_path, "shop")
    store.apply_memory_ops(
        "user",
        [
            {"op": "add", "path": "profile", "description": "Who the user is", "text": "Named Ada"},
            {"op": "add", "path": "hobbies", "description": "Free time", "text": "Climbs"},
        ],
        root=tmp_path,
    )
    decision = {"op": "add", "path": "decisions", "description": "Shop calls", "text": "Ships EU"}
    store.apply_memory_ops("shop", [decision], root=tmp_path)

    prompt = memory_prompt(tmp_path, "shop")

    assert store.list_memories(root=tmp_path)["memory_stores"] == ["shop", "user"]
    assert "`user`" in first and "`shop`" in first
    assert "- Named Ada" in prompt and "[" not in prompt.split("## Index")[0].split("# Memory", 1)[1]
    assert "hobbies — Free time" in prompt and "Climbs" not in prompt
    assert "decisions — Shop calls" in prompt and "Ships EU" not in prompt
    assert "Read a memory before you answer from it." in prompt


def test_memory_prompt_without_a_project_names_only_user(tmp_path: Path) -> None:
    prompt = memory_prompt(tmp_path, None)

    assert "Index of `user`" in prompt
    assert "this project" not in prompt
    assert store.list_memories(root=tmp_path)["memory_stores"] == ["user"]


def test_memory_prompt_cuts_nothing(tmp_path: Path) -> None:
    memory_prompt(tmp_path, None)
    long_fact = {"op": "add", "path": "profile", "description": "Who", "text": "x" * 20_000}
    many = [{"op": "add", "path": f"topic-{n:03d}", "description": f"Topic {n}", "text": "t"} for n in range(100)]
    store.apply_memory_ops("user", [long_fact, *many], root=tmp_path)

    prompt = memory_prompt(tmp_path, None)

    assert "x" * 20_000 in prompt
    assert prompt.count(" · 1 facts") == 101
