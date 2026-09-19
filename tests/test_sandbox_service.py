from __future__ import annotations

import os
from uuid import uuid4

import pytest

from config import TEST_MODEL
from lib.agent_sandbox_adapter import FakeSandboxRunner
from lib.asset_store import AssetStore, StorageNotFoundError
from lib.postgres_data_store import PostgresDataStore
from lib.sandbox_service import PROMPT_MAX_LEN, SandboxService
from lib.sandbox_store import SandboxStore
from lib.storage_namespace import sandbox_asset

# The model id is caller-supplied configuration; these tests only check it is forwarded.
SANDBOX_MODEL = TEST_MODEL


@pytest.fixture(scope="session")
def postgres_pool():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_DATABASE_URL not set")
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(url, min_size=1, max_size=4, open=False)
    pool.open()
    try:
        yield pool
    finally:
        pool.close()


@pytest.fixture
def user_id(postgres_pool) -> str:
    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute("TRUNCATE changes, assets, vault_roots, devices, documents, users CASCADE")
        row = connection.execute(
            "INSERT INTO users (tailscale_login) VALUES (%s) RETURNING id", (f"test-{uuid4()}@example.test",)
        ).fetchone()
    return str(row[0])


@pytest.fixture
def data_store(postgres_pool, user_id) -> PostgresDataStore:
    return PostgresDataStore(postgres_pool, user_id)


@pytest.fixture
def asset_store(postgres_pool, user_id) -> AssetStore:
    return AssetStore(postgres_pool, user_id)


@pytest.fixture
def runner() -> FakeSandboxRunner:
    return FakeSandboxRunner()


@pytest.fixture
def sandbox_service(data_store, asset_store, runner) -> SandboxService:
    return SandboxService(
        data_store=data_store,
        asset_store=asset_store,
        runner=runner,
        model=SANDBOX_MODEL,
        profile="gigachad",
        slot_prefix="test-prefix",
    )


async def test_workspace_is_staged_until_checkpoint_while_outputs_are_durable(sandbox_service, asset_store, data_store, runner):
    chat_id = "chat-first-invoke"
    result = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="tool-call-1", prompt="Create a workspace")

    from lib.sandbox_service import _STAGED

    staged = _STAGED["test-prefix:workspace:chat-first-invoke"]
    assert staged.manifest is not None
    assert staged.manifest.workspace is not None
    assert result.status == "completed"
    assert result.manifest_id is not None
    output = result.outputs[0]
    assert asset_store.read(output.mime_bundle["text/markdown"]["asset_path"]).content == output.text.encode("utf-8")
    assert SandboxStore(data_store).read_active(chat_id) is None
    with pytest.raises(StorageNotFoundError):
        asset_store.read(sandbox_asset(staged.manifest.workspace.sha256))

    await sandbox_service.checkpoint(chat_id=chat_id)

    active = SandboxStore(data_store).read_active(chat_id)
    assert active is not None
    assert active.manifest_id == result.manifest_id
    manifest = SandboxStore(data_store).read_manifest(chat_id, active.manifest_id)
    assert manifest.workspace is not None
    assert asset_store.read(sandbox_asset(manifest.workspace.sha256)).content


async def test_tool_scopes_keep_staged_and_persisted_workspaces_independent(sandbox_service, data_store, runner):
    chat_id = "chat-scopes"
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="shared-call", prompt="workspace", scope="workspace_agent")
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="shared-call", prompt="plot", scope="sandbox_plot")
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="workspace-next", prompt="again", scope="workspace_agent")
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="plot-next", prompt="again", scope="sandbox_plot")

    workspace_prior = runner.calls[-2].prior_manifest
    plot_prior = runner.calls[-1].prior_manifest
    assert workspace_prior is not None and workspace_prior.workspace is not None
    assert plot_prior is not None and plot_prior.workspace is not None
    assert runner.calls[-2].prior_assets[workspace_prior.workspace.sha256].decode("utf-8") == "> workspace\n"
    assert runner.calls[-1].prior_assets[plot_prior.workspace.sha256].decode("utf-8") == "> plot\n"

    await sandbox_service.checkpoint(chat_id=chat_id)
    assert SandboxStore(data_store, scope="workspace_agent").read_active(chat_id) is not None
    assert SandboxStore(data_store, scope="sandbox_plot").read_active(chat_id) is not None

    await sandbox_service.delete_chat_state(chat_id=chat_id)
    assert SandboxStore(data_store, scope="workspace_agent").read_active(chat_id) is None
    assert SandboxStore(data_store, scope="sandbox_plot").read_active(chat_id) is None


async def test_second_invoke_restores_staged_workspace_before_checkpoint(sandbox_service, runner):
    chat_id = "chat-restore-manifest"
    result_1 = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="tool-call-1", prompt="First prompt")
    result_2 = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="tool-call-2", prompt="Second prompt")

    assert result_2.workspace_changed is True
    assert result_2.manifest_id != result_1.manifest_id
    assert len(runner.calls) == 2
    prior_manifest = runner.calls[-1].prior_manifest
    assert prior_manifest is not None
    from lib.sandbox_store import manifest_id_for
    assert manifest_id_for(prior_manifest) == result_1.manifest_id
    assert prior_manifest.workspace is not None
    assert runner.calls[-1].prior_assets[prior_manifest.workspace.sha256].decode("utf-8") == "> First prompt\n"


async def test_staged_workspaces_are_isolated_between_chats(sandbox_service, runner):
    await sandbox_service.invoke(chat_id="chat-a", tool_call_id="a-1", prompt="alpha")
    await sandbox_service.invoke(chat_id="chat-b", tool_call_id="b-1", prompt="bravo")
    await sandbox_service.invoke(chat_id="chat-a", tool_call_id="a-2", prompt="again")

    prior_manifest = runner.calls[-1].prior_manifest
    assert prior_manifest is not None and prior_manifest.workspace is not None
    assert runner.calls[-1].prior_assets[prior_manifest.workspace.sha256].decode("utf-8") == "> alpha\n"


async def test_idempotent_invoke_same_tool_call_id(sandbox_service, runner):
    chat_id = "chat-idempotent"
    tool_call_id = "tool-call-same"
    prompt = "Same prompt twice"

    result_1 = await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt=prompt)
    result_2 = await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt=prompt)

    assert len(runner.calls) == 1
    assert result_1.to_json() == result_2.to_json()


async def test_runner_failure_degrades_gracefully(sandbox_service, runner):
    chat_id = "chat-runner-fail"
    tool_call_id = "tool-call-fail"
    prompt = "This will fail"

    async def failing_run(invocation):
        raise RuntimeError("boom")

    runner.run = failing_run

    result = await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt=prompt)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error == "runner_error"
    assert result.manifest_id is None


async def test_empty_prompt_raises(sandbox_service):
    chat_id = "chat-empty-prompt"
    tool_call_id = "tool-call-empty"

    with pytest.raises(ValueError, match="prompt must not be empty"):
        await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt="")

    with pytest.raises(ValueError, match="prompt must not be empty"):
        await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt="   \n\t  ")


async def test_overlong_prompt_raises(sandbox_service):
    chat_id = "chat-long-prompt"
    tool_call_id = "tool-call-long"
    prompt = "x" * (PROMPT_MAX_LEN + 1)

    with pytest.raises(ValueError, match=f"prompt exceeds {PROMPT_MAX_LEN} characters"):
        await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt=prompt)


async def test_delete_chat_state_removes_active_pointer_and_assets(sandbox_service, asset_store, data_store):
    chat_id = "chat-to-delete"
    tool_call_id = "tool-call-delete"
    prompt = "Create then delete"

    result = await sandbox_service.invoke(chat_id=chat_id, tool_call_id=tool_call_id, prompt=prompt)
    assert result.manifest_id is not None

    asset_paths = []
    for output in result.outputs:
        for _media_type, ref in output.mime_bundle.items():
            asset_paths.append(ref["asset_path"])

    for path in asset_paths:
        asset = asset_store.read(path)
        assert asset.content is not None

    await sandbox_service.delete_chat_state(chat_id=chat_id)

    store = SandboxStore(data_store)
    active = store.read_active(chat_id)
    assert active is None

    for path in asset_paths:
        with pytest.raises(StorageNotFoundError):
            asset_store.read(path)


async def test_delete_chat_state_preserves_shared_assets(sandbox_service, asset_store, data_store):
    chat_id_1 = "chat-shared-1"
    chat_id_2 = "chat-shared-2"
    tool_call_id = "tool-call-shared"
    prompt = "Identical prompt"

    result_1 = await sandbox_service.invoke(chat_id=chat_id_1, tool_call_id=tool_call_id, prompt=prompt)
    assert result_1.manifest_id is not None

    result_2 = await sandbox_service.invoke(chat_id=chat_id_2, tool_call_id=tool_call_id, prompt=prompt)
    assert result_2.manifest_id is not None

    asset_paths_1 = set()
    for output in result_1.outputs:
        for _media_type, ref in output.mime_bundle.items():
            asset_paths_1.add(ref["asset_path"])

    asset_paths_2 = set()
    for output in result_2.outputs:
        for _media_type, ref in output.mime_bundle.items():
            asset_paths_2.add(ref["asset_path"])

    assert asset_paths_1 == asset_paths_2

    await sandbox_service.delete_chat_state(chat_id=chat_id_1)

    store = SandboxStore(data_store)
    assert store.read_active(chat_id_2) is None

    for path in asset_paths_1:
        asset = asset_store.read(path)
        assert asset.content is not None
