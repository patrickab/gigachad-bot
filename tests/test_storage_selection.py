import config

from lib.data_store import LocalDataStore


def _reset_store(monkeypatch) -> None:
    monkeypatch.setattr(config, "_data_store", None)


def test_storage_uses_local_mirror_without_complete_webdav_environment(monkeypatch):
    for name in ("GIGACHAD_WEBDAV_URL", "GIGACHAD_WEBDAV_USER", "GIGACHAD_WEBDAV_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIGACHAD_STORAGE", "webdav")
    _reset_store(monkeypatch)

    store = config.get_data_store()

    assert isinstance(store, LocalDataStore)
    assert store.root == config.DOCUMENTS.resolve()


def test_storage_uses_webdav_when_all_credentials_exist(monkeypatch):
    for name in ("GIGACHAD_WEBDAV_URL", "GIGACHAD_WEBDAV_USER", "GIGACHAD_WEBDAV_PASSWORD"):
        monkeypatch.setenv(name, "configured")
    _reset_store(monkeypatch)
    expected = object()
    fallbacks = []

    def from_environment(cls, *, fallback):
        del cls
        fallbacks.append(fallback)
        return expected

    monkeypatch.setattr(config.WebDavDataStore, "from_environment", classmethod(from_environment))

    assert config.get_data_store() is expected
    assert len(fallbacks) == 1
    assert isinstance(fallbacks[0], LocalDataStore)
    assert fallbacks[0].root == config.DOCUMENTS.resolve()


def test_ensure_directories_uses_the_selected_store(monkeypatch):
    calls = []

    class RecordingStore:
        def mkdir(self, key):
            calls.append(key)

    store = RecordingStore()
    monkeypatch.setattr(config, "get_data_store", lambda: store)
    monkeypatch.setattr(config, "seed_prompts", lambda supplied: calls.append(("seed", supplied)))

    config.ensure_directories()

    assert "chat_history" in calls
    assert ("seed", store) in calls
