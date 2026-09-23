from __future__ import annotations

import hashlib
import os
from uuid import uuid4

from agent_sandbox import SandboxManifest, SandboxRun
from agent_sandbox.manifest import Execution, Outputs, WorkspaceRef, to_yaml
import pytest

from config import TEST_MODEL
from lib.agent_sandbox_adapter import FakeSandboxRunner
from lib.asset_store import AssetStore, StorageNotFoundError
from lib.data_store import DataStorePath
from lib.json_io import safe_write_json
from lib.postgres_data_store import PostgresDataStore
from lib.sandbox_service import PROMPT_MAX_LEN, SandboxRuntimeState, SandboxService
from lib.sandbox_store import ActivePointer, AssetRef, SandboxStore
from lib.storage_namespace import sandbox_asset, sandbox_manifest

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
def runtime_state() -> SandboxRuntimeState:
    return SandboxRuntimeState()


@pytest.fixture
def sandbox_service(data_store, asset_store, runner, runtime_state) -> SandboxService:
    return SandboxService(
        data_store=data_store,
        asset_store=asset_store,
        runner=runner,
        runtime_state=runtime_state,
        model=SANDBOX_MODEL,
        profile="gigachad",
        slot_prefix="test-prefix",
    )


async def test_workspace_is_staged_until_checkpoint_while_outputs_are_durable(
    sandbox_service, asset_store, data_store, runtime_state, runner
):
    chat_id = "chat-first-invoke"
    result = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="tool-call-1", prompt="Create a workspace")

    staged = runtime_state.get("test-prefix:workspace:chat-first-invoke")
    assert staged is not None
    assert staged.manifest is not None
    assert staged.manifest.workspace is not None
    assert result.status == "completed"
    assert result.manifest_id is not None
    output = result.outputs[0]
    assert asset_store.read(output.mime_bundle["text/plain"]["asset_path"]).content == output.text.encode("utf-8")
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

    resumed = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="tool-call-2", prompt="Resume the workspace")
    prior_manifest = runner.calls[-1].active_manifest
    assert prior_manifest is not None
    assert prior_manifest.manifest_id == result.manifest_id
    assert resumed.manifest_id != result.manifest_id


async def test_legacy_manifest_resumes_with_its_embedded_aib_record(sandbox_service, asset_store, data_store, runner):
    chat_id = "chat-legacy-resume"
    workspace_bytes = b"> legacy workspace\n"
    workspace_sha256 = hashlib.sha256(workspace_bytes).hexdigest()
    manifest = SandboxManifest(
        schema_version=1,
        manifest_id="legacy-native-manifest",
        profile="gigachad",
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=sandbox_asset(workspace_sha256), sha256=workspace_sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={},
        execution=Execution(run_id="legacy-run", status="completed", exit_code=0, created_at=""),
    )
    legacy_manifest_id = "legacy-wrapper-id"
    safe_write_json(
        DataStorePath(data_store, sandbox_manifest(chat_id, legacy_manifest_id)),
        {
            "version": 1,
            "workspace": AssetRef(workspace_sha256, "application/x-tar", len(workspace_bytes)).to_json(),
            "omp_session": None,
            "extra": {"agent_sandbox_manifest": to_yaml(manifest)},
        },
    )
    asset_store.write("sandbox", sandbox_asset(workspace_sha256), workspace_bytes, mime="application/x-tar")
    SandboxStore(data_store).write_active(chat_id, ActivePointer(manifest_id=legacy_manifest_id, updated_at=1.0))

    result = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="legacy-call", prompt="Resume it")

    assert runner.calls[-1].active_manifest == manifest
    assert result.manifest_id is not None


async def test_tool_scopes_keep_staged_and_persisted_workspaces_independent(sandbox_service, data_store, runner):
    chat_id = "chat-scopes"
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="shared-call", prompt="workspace", scope="workspace_agent")
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="shared-call", prompt="plot", scope="sandbox_plot")
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="workspace-next", prompt="again", scope="workspace_agent")
    await sandbox_service.invoke(chat_id=chat_id, tool_call_id="plot-next", prompt="again", scope="sandbox_plot")

    workspace_prior = runner.calls[-2].active_manifest
    plot_prior = runner.calls[-1].active_manifest
    assert workspace_prior is not None and workspace_prior.workspace is not None
    assert plot_prior is not None and plot_prior.workspace is not None

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
    prior_manifest = runner.calls[-1].active_manifest
    assert prior_manifest is not None
    assert prior_manifest.manifest_id == result_1.manifest_id
    assert prior_manifest.workspace is not None


async def test_staged_workspaces_are_isolated_between_chats(sandbox_service, runner):
    await sandbox_service.invoke(chat_id="chat-a", tool_call_id="a-1", prompt="alpha")
    await sandbox_service.invoke(chat_id="chat-b", tool_call_id="b-1", prompt="bravo")
    await sandbox_service.invoke(chat_id="chat-a", tool_call_id="a-2", prompt="again")

    prior_manifest = runner.calls[-1].active_manifest
    assert prior_manifest is not None and prior_manifest.workspace is not None


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


async def test_notebook_staged_only_until_checkpoint(
    sandbox_service, asset_store, data_store, runtime_state, runner
):
    chat_id = "chat-notebook-staged"
    source = '# %% [markdown]\n# Title\n\n# %%\nprint("hi")\n'
    outputs = {"cells": [{"outputs": [{"text": "hi"}]}]}

    await sandbox_service.stage_notebook(chat_id, "rev-1", source, outputs)

    # Staged reads work and nothing is durable yet.
    notebook = await sandbox_service.read_notebook(chat_id)
    assert notebook == {"revision_id": "rev-1", "source": source, "outputs": outputs}
    assert SandboxStore(data_store, scope="notebook").read_notebook_pointer(chat_id) is None
    assert SandboxStore(data_store, scope="notebook").read_run(chat_id, "rev-1") is None
    staged_slot = "test-prefix:notebook:" + chat_id
    staged = runtime_state.get(staged_slot)
    assert staged is not None
    for sha256 in staged.output_shas:
        with pytest.raises(StorageNotFoundError):
            asset_store.read(sandbox_asset(sha256))

    # A fresh service over the same stores (restart simulation) sees nothing.
    fresh = SandboxService(
        data_store=data_store,
        asset_store=asset_store,
        runner=runner,
        runtime_state=SandboxRuntimeState(),
        model=SANDBOX_MODEL,
        profile="gigachad",
        slot_prefix="test-prefix",
    )
    assert await fresh.read_notebook(chat_id) is None

    await sandbox_service.checkpoint(chat_id=chat_id)
    assert await fresh.read_notebook(chat_id) == {"revision_id": "rev-1", "source": source, "outputs": outputs}



async def test_notebook_delete_chat_state_removes_assets(sandbox_service, asset_store, data_store):
    chat_id = "chat-notebook-delete"
    source = "# %%\nprint('x')\n"
    outputs = {"cells": [{"outputs": []}]}

    await sandbox_service.stage_notebook(chat_id, "rev-3", source, outputs)
    await sandbox_service.checkpoint(chat_id=chat_id)

    store = SandboxStore(data_store, scope="notebook")
    record = store.read_run(chat_id, "rev-3")
    assert record is not None and record.result is not None
    asset_paths = [
        ref["asset_path"]
        for output in record.result["outputs"]
        for ref in output["mime_bundle"].values()
    ]
    assert len(asset_paths) == 2
    for path in asset_paths:
        assert asset_store.read(path).content is not None

    await sandbox_service.delete_chat_state(chat_id=chat_id)

    assert store.read_notebook_pointer(chat_id) is None
    assert store.read_run(chat_id, "rev-3") is None
    for path in asset_paths:
        with pytest.raises(StorageNotFoundError):
            asset_store.read(path)


async def test_notebook_does_not_disturb_workspace_scopes(sandbox_service, data_store, runner):
    chat_id = "chat-notebook-mixed"

    result = await sandbox_service.invoke(chat_id=chat_id, tool_call_id="ws-call", prompt="Build a workspace")
    await sandbox_service.stage_notebook(chat_id, "rev-4", "# %%\nprint(1)\n", {"cells": []})
    await sandbox_service.checkpoint(chat_id=chat_id)

    assert SandboxStore(data_store).read_active(chat_id) is not None
    assert result.manifest_id is not None
    manifest = SandboxStore(data_store).read_manifest(chat_id, result.manifest_id)
    assert manifest.workspace is not None
    assert SandboxStore(data_store, scope="notebook").read_notebook_pointer(chat_id) is not None


class FakeSeededRunner:
    """Return a capture containing a caller-named result file, recording each invocation."""

    def __init__(self, result_name: str, result_bytes: bytes) -> None:
        self.result_name = result_name
        self.result_bytes = result_bytes
        self.calls: list = []
        self.seed_archives: list[bytes] = []

    async def run_script(self, script: str, *, profile: str, interpreter: str, timeout: float) -> str:
        raise AssertionError("run_seeded never runs scripts")

    async def run(self, invocation) -> "SandboxRun":
        import io as _io
        import os as _os
        import tarfile as _tarfile
        import tempfile as _tempfile

        self.calls.append(invocation)
        # Snapshot the seed archive while the service still exposes it on disk.
        with open(invocation.workspace_archive_path, "rb") as seed_file:
            self.seed_archives.append(seed_file.read())
        archive_bytes = _io.BytesIO()
        with _tarfile.open(fileobj=archive_bytes, mode="w") as archive:
            member = _tarfile.TarInfo(self.result_name)
            member.size = len(self.result_bytes)
            archive.addfile(member, _io.BytesIO(self.result_bytes))
        fd, capture_path = _tempfile.mkstemp(prefix="test-seeded-", suffix=".tar")
        with _os.fdopen(fd, "wb") as capture:
            capture.write(archive_bytes.getvalue())
        manifest = SandboxManifest(
            schema_version=1,
            manifest_id=f"seeded-{invocation.run_id}",
            profile=invocation.profile_name,
            runtime_fingerprint="fake",
            workspace=WorkspaceRef(snapshot_asset_id=None, sha256=hashlib.sha256(archive_bytes.getvalue()).hexdigest()),
            outputs=Outputs(events_asset_id=None, artifacts=()),
            omp_sessions={},
            execution=Execution(run_id=invocation.run_id, status="completed", exit_code=0, created_at=""),
        )
        return SandboxRun(
            status="completed",
            summary="edited the notebook",
            next_manifest=manifest,
            outputs=(),
            workspace_changed=True,
            capture_path=capture_path,
        )


def _seed_files(seed_archives: list[bytes]) -> dict[str, bytes]:
    """Extract {name: content} from the first seeded tar snapshot."""
    import io as _io
    import tarfile as _tarfile

    files: dict[str, bytes] = {}
    with _tarfile.open(fileobj=_io.BytesIO(seed_archives[0]), mode="r:") as archive:
        for member in archive:
            source = archive.extractfile(member)
            assert source is not None
            files[member.name] = source.read()
    return files


async def test_run_seeded_builds_a_fresh_notebook_agent_run(sandbox_service, runtime_state):
    """The invocation is seed-only: no manifest, a correct sha, and notebook-scoped slot."""
    import hashlib as _hashlib
    import os as _os

    seeded_runner = FakeSeededRunner("notebook.py", b"# %%\nprint('new')\n")
    sandbox_service._runner = seeded_runner  # noqa: SLF001 - swap in the seeded fake
    chat_id = "chat-run-seeded"
    files = {
        "notebook.py": b"# %%\nprint('old')\n",
        "conversation.md": b"# Conversation (reference only)\n",
    }

    summary, result_bytes = await sandbox_service.run_seeded(
        chat_id=chat_id,
        tool_call_id="tool-call-seed",
        files=files,
        prompt="add a cell",
        append_system="# rules",
    )

    assert summary == "edited the notebook"
    assert result_bytes == b"# %%\nprint('new')\n"
    (invocation,) = seeded_runner.calls
    assert invocation.active_manifest is None
    assert invocation.run_id == "tool-call-seed"
    assert invocation.slot_key == f"test-prefix:notebook:{chat_id}"
    assert invocation.append_system == "# rules"
    assert invocation.thinking == "low"
    assert invocation.lean is True
    assert invocation.model == SANDBOX_MODEL

    # The seed archive matches the declared sha and carries exactly the requested files.
    archive_bytes = seeded_runner.seed_archives[0]
    assert invocation.workspace_archive_sha256 == _hashlib.sha256(archive_bytes).hexdigest()
    assert _seed_files(seeded_runner.seed_archives) == files

    # No workspace manifest is staged and no durable pointer appears: only bytes matter.
    assert runtime_state.get(f"test-prefix:notebook:{chat_id}") is None
    assert not _os.path.exists(invocation.workspace_archive_path)



async def test_run_seeded_missing_result_file_raises(sandbox_service):
    """A capture without the named result file surfaces as a contained error."""
    seeded_runner = FakeSeededRunner("other.py", b"unrelated")
    sandbox_service._runner = seeded_runner  # noqa: SLF001

    with pytest.raises(Exception):
        await sandbox_service.run_seeded(
            chat_id="chat-seeded-missing",
            tool_call_id="tool-call-missing",
            files={"notebook.py": b"# %%\n"},
            prompt="edit",
        )

    (invocation,) = seeded_runner.calls
    import os as _os
    assert not _os.path.exists(invocation.workspace_archive_path)
