import config


def test_runtime_paths_are_outside_checkout():
    assert config.DIRECTORY_CHAT_HISTORIES == config.REMOTE_ROOT / "Documents/chat_history"
    for name, value in vars(config).items():
        if name.startswith("DIRECTORY_"):
            assert value.is_absolute()
            assert value.is_relative_to(config.REMOTE_ROOT)


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
