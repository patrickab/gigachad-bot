from lib.data_store import LocalDataStore

import config


def _point_config_at(tmp_path, monkeypatch) -> None:
    """Re-root every configured directory under a temporary Documents tree."""
    documents = tmp_path / "Documents"
    for name, value in list(vars(config).items()):
        if name.startswith("DIRECTORY_"):
            monkeypatch.setattr(config, name, documents / value.relative_to(config.DOCUMENTS))
    monkeypatch.setattr(config, "DOCUMENTS", documents)
    monkeypatch.setattr(config, "_data_store", None)


def test_runtime_paths_are_outside_checkout():
    assert config.DIRECTORY_CHAT_HISTORIES == config.REMOTE_ROOT / "Documents/chat_history"
    for name, value in vars(config).items():
        if name.startswith("DIRECTORY_"):
            assert value.is_absolute()
            assert value.is_relative_to(config.REMOTE_ROOT)


def test_data_store_is_a_cached_local_store_rooted_at_documents(monkeypatch):
    monkeypatch.setattr(config, "_data_store", None)

    store = config.get_data_store()

    assert isinstance(store, LocalDataStore)
    assert store.root == config.DOCUMENTS.resolve()
    assert config.get_data_store() is store


def test_ensure_directories_creates_the_configured_tree(tmp_path, monkeypatch):
    _point_config_at(tmp_path, monkeypatch)

    config.ensure_directories()

    expected = [
        config.DIRECTORY_CHAT_HISTORIES,
        config.DIRECTORY_OUTPUT_MINERU,
        config.DIRECTORY_OUTPUT_MINERU / "images",
        config.DIRECTORY_OUTPUT_PDF,
        config.DIRECTORY_OUTPUT_MARKDOWN,
        config.DIRECTORY_OUTPUT_LATEX,
        config.DIRECTORY_OUTPUT_DRAWINGS,
        config.DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS,
        config.DIRECTORY_OUTPUT_ARCHITECTURE_GRAPHS / ".drafts",
        config.DIRECTORY_CHAT_UPLOADS,
        config.DIRECTORY_NOTES,
        config.DIRECTORY_CHAT_HISTORIES / "memory",
        config.DIRECTORY_CHAT_HISTORIES / "memory" / "pending",
    ]
    missing = [str(directory) for directory in expected if not directory.is_dir()]
    assert not missing


def test_prompt_seed_preserves_edits_and_deletions(tmp_path, monkeypatch):
    target = tmp_path / "Prompts"
    monkeypatch.setattr(config, "DIRECTORY_PROMPTS", target)
    config.seed_prompts()
    prompt = next(target.glob("*.md"))
    prompt.write_text("User edit")
    config.seed_prompts()
    assert prompt.read_text() == "User edit"
    prompt.unlink()
    config.seed_prompts()
    assert not prompt.exists()
