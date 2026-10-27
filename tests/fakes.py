"""In-memory stand-ins for the stores, sandbox and LLM client that tests share."""

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace
from typing import Any

from agent_sandbox import SandboxManifest
from agent_sandbox.manifest import Execution, Outputs, WorkspaceRef

from config import TEST_MODEL
from lib import sandbox_plot, tools
from lib.agent_sandbox_adapter import SandboxScriptError
from lib.data_store import Entry, Revision, StorageNotFoundError
from lib.fake_sandbox_runner import FakeSandboxRunner
from lib.sandbox_service import SandboxRuntimeState, SandboxService, SandboxToolResult
from lib.tool_context import ToolOptions


class MemoryDataStore:
    """Document store over a dict: no revisions are enforced and listing ignores depth."""

    def __init__(self) -> None:
        self._documents: dict[str, bytes] = {}

    def read_bytes(self, key: str) -> tuple[bytes, Revision]:
        try:
            content = self._documents[key]
        except KeyError as exc:
            raise StorageNotFoundError(key) from exc
        return content, Revision(hashlib.sha256(content).hexdigest())

    def write_bytes(self, key: str, content: bytes, *, expected: Revision | None = None) -> Revision:
        del expected
        self._documents[key] = content
        return Revision(hashlib.sha256(content).hexdigest())

    def apply(self, writes: list[tuple[str, bytes, Revision | None]], deletes: list[str]) -> list[Revision]:
        revisions = [self.write_bytes(key, content, expected=expected) for key, content, expected in writes]
        for key in deletes:
            self.delete(key)
        return revisions

    def list(self, prefix: str = "", *, recursive: bool = False) -> list[Entry]:
        del recursive
        scope = f"{prefix}/" if prefix else ""
        return [Entry(key=key, is_dir=False, size=len(value)) for key, value in self._documents.items() if key.startswith(scope)]

    def exists(self, key: str) -> bool:
        return key in self._documents

    def mkdir(self, key: str) -> None:
        del key

    def delete(self, key: str, *, recursive: bool = False) -> None:
        del recursive
        for existing in [k for k in self._documents if k == key or k.startswith(f"{key}/")]:
            del self._documents[existing]

    def move(self, source: str, destination: str) -> None:
        self._documents[destination] = self._documents.pop(source)


@dataclass
class StoredAsset:
    logical_path: str
    content: bytes
    kind: str
    mime: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


class MemoryAssetStore:
    """Asset store over a dict keyed by logical path, recording every key it is asked to read."""

    def __init__(self) -> None:
        self.assets: dict[str, StoredAsset] = {}
        self.read_keys: list[str] = []

    def read(self, key: str) -> StoredAsset:
        self.read_keys.append(key)
        try:
            return self.assets[key]
        except KeyError as exc:
            raise StorageNotFoundError(key) from exc

    def stat(self, key: str) -> StoredAsset | None:
        return self.assets.get(key)

    def list(self, prefix: str = "", *, kind: str | None = None) -> list[StoredAsset]:
        return [asset for key, asset in self.assets.items() if key.startswith(prefix) and kind in (None, asset.kind)]

    def write(self, kind: str, key: str, content: bytes, *, mime: str | None = None) -> StoredAsset:
        self.assets[key] = StoredAsset(key, content, kind, mime)
        return self.assets[key]

    def delete(self, key: str) -> None:
        self.assets.pop(key, None)

    def mirror(self, asset: StoredAsset, root: Path) -> None:
        """Nextcloud mirroring is out of scope for in-memory tests."""


class SandboxServiceStub:
    """Sandbox facade that answers tool runs from canned output and records every call.

    ``events`` is shared with sibling stubs so a test can assert the order of saves and checkpoints."""

    def __init__(
        self,
        stdout: str = "",
        script_stdout: str | None = None,
        invoke_result: SandboxToolResult | None = None,
        events: list[tuple[str, str | None]] | None = None,
    ) -> None:
        self.model = TEST_MODEL
        self.stdout = stdout
        self.script_stdout = script_stdout
        self.invoke_result = invoke_result
        self.events = events if events is not None else []
        self.calls: list[dict[str, Any]] = []
        self.scripts: list[str] = []

    async def invoke(self, **kwargs: Any) -> SandboxToolResult:
        self.calls.append(kwargs)
        return self.invoke_result or SandboxToolResult(
            status="completed", summary="hidden transcript", manifest_id="manifest", workspace_changed=True
        )

    async def run_script(self, script: str, **_kwargs: Any) -> str:
        self.scripts.append(script)
        if self.script_stdout is None:
            raise SandboxScriptError("ModuleNotFoundError: no such module")
        return self.script_stdout

    def output_texts(self, _result: SandboxToolResult, *, media_type: str = "text/plain") -> tuple[str, ...]:
        assert media_type == sandbox_plot.PLOTLY_MEDIA_TYPE
        return (self.stdout,)

    async def checkpoint(self, *, chat_id: str) -> None:
        self.events.append(("checkpoint", chat_id))


def memory_sandbox_service(
    runner: Any = None,
    *,
    data_store: MemoryDataStore | None = None,
    assets: MemoryAssetStore | None = None,
    runtime: SandboxRuntimeState | None = None,
    slot_prefix: str = "",
) -> SandboxService:
    """A real ``SandboxService`` over in-memory stores, running ``FakeSandboxRunner`` unless given a runner."""
    return SandboxService(
        data_store=data_store or MemoryDataStore(),
        asset_store=assets or MemoryAssetStore(),
        runner=runner or FakeSandboxRunner(),
        runtime_state=runtime or SandboxRuntimeState(),
        model=TEST_MODEL,
        slot_prefix=slot_prefix,
    )


def tar_archive(content: bytes) -> bytes:
    """A tar holding one ``workspace.txt`` with *content*."""
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("workspace.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return archive_bytes.getvalue()


def native_manifest(
    manifest_id: str,
    sha256: str,
    *,
    run_id: str | None = None,
    snapshot_asset_id: str | None = None,
    profile: str = "gigachad",
) -> SandboxManifest:
    """A completed native manifest whose workspace has *sha256* and that has no outputs."""
    return SandboxManifest(
        schema_version=1,
        manifest_id=manifest_id,
        profile=profile,
        runtime_fingerprint="test",
        workspace=WorkspaceRef(snapshot_asset_id=snapshot_asset_id, sha256=sha256),
        outputs=Outputs(events_asset_id=None, artifacts=()),
        omp_sessions={},
        execution=Execution(run_id=run_id or manifest_id, status="completed", exit_code=0, created_at=""),
    )


class ClientStub:
    """Only the two client seams the tool loop borrows: routing and payload construction."""

    def _resolve_routing(self, model: str) -> tuple[str, None, None]:
        return model, None, None

    def _construct_message_payload(
        self,
        user_msg: str | None = None,
        user_msg_history: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        img: Any = None,  # noqa: ARG002 - mirrors the client signature
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(user_msg_history or [])
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        return messages


def text_chunk(text: str) -> SimpleNamespace:
    """A streamed provider chunk carrying answer text."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))], usage=None)


def tool_chunk(index: int, call_id: str, name: str | None, arguments: str) -> SimpleNamespace:
    """A streamed provider chunk carrying (a fragment of) one tool call."""
    call = SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call]))], usage=None)


def usage_chunk(total: int) -> SimpleNamespace:
    """A streamed provider chunk carrying only token usage."""
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=total, completion_tokens=0, total_tokens=total), choices=[])


async def collect_tool_turn(**overrides: Any) -> list[tuple[str, Any]]:
    """Run one ``stream_chat_with_tools`` turn against ``ClientStub`` and return its events."""
    kwargs: dict[str, Any] = {
        "client": ClientStub(),
        "model": TEST_MODEL,
        "user_msg": "who won?",
        "history": [],
        "system_prompt": "be terse",
        "enabled": ["web_search"],
        "opts": ToolOptions(),
    }
    kwargs.update(overrides)
    return [event async for event in tools.stream_chat_with_tools(**kwargs)]
