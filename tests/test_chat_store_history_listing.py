from lib.chat_store import ChatStore
from lib.data_store import LocalDataStore
from lib.json_io import safe_write_json


def test_history_listing_sorts_storage_backed_paths_by_name(tmp_path):
    store = ChatStore(tmp_path / "chat_history", data_store=LocalDataStore(tmp_path))
    for filename in ("z.json", "a.json", "folder/b.json"):
        safe_write_json(store.resolve_path(filename), {"chat_id": filename, "messages": []})

    assert store.list_histories() == {
        "files": ["a.json", "z.json"],
        "histories": {"folder": ["b.json"]},
    }
    assert list(store.get_branch_meta()) == ["a.json", "folder/b.json", "z.json"]
