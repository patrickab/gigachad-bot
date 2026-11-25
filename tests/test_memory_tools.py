"""The chat's memory tools: `list_memories`, `read_memories`, `apply_memory_ops` over the login's root, off the event loop."""

from pathlib import Path
import threading
from typing import Any

from memory_store import store
import pytest

from lib import tools
from lib.tool_context import ToolContext, ToolOptions
from lib.tools import BUILTIN_TOOLS, offered_tools

MEMORY_TOOLS = ["list_memories", "read_memories", "apply_memory_ops"]


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    store.create_memory_store("user", root=tmp_path)
    return ToolContext(model="test", opts=ToolOptions(), memory_root=tmp_path)


async def run(name: str, args: dict[str, Any], context: ToolContext) -> Any:
    return await BUILTIN_TOOLS.execute(name, args, context, enabled=MEMORY_TOOLS)


def test_apply_memory_ops_tells_the_model_to_read_ids_first_and_update_instead_of_adding() -> None:
    [spec] = BUILTIN_TOOLS.specs(["apply_memory_ops"])

    assert "Read the memory first for the fact ids" in spec["function"]["description"]
    assert "an update of it, not a new add" in spec["function"]["description"]


@pytest.mark.parametrize("host_access", [True, False])
def test_the_memory_toggle_expands_to_the_three_tools_for_every_login(host_access: bool) -> None:
    assert offered_tools(["memory"], host_access=host_access) == MEMORY_TOOLS
    assert [spec["function"]["name"] for spec in BUILTIN_TOOLS.specs(MEMORY_TOOLS)] == MEMORY_TOOLS


@pytest.mark.asyncio
async def test_apply_read_and_list_round_trip(context: ToolContext) -> None:
    ops = [{"op": "add", "path": "profile", "description": "Who the user is", "text": "Named Ada"}]

    applied = await run("apply_memory_ops", {"memory_store": "user", "ops": ops}, context)
    read = await run("read_memories", {"memory_store": "user", "paths": ["profile"], "ids": False}, context)
    listed = await run("list_memories", {"memory_store": "user"}, context)

    assert applied.error is None and applied.content.startswith("ok add profile ")
    assert applied.summary == "1/1 applied"
    assert read.content == "## profile — Who the user is\n- Named Ada\n"
    assert "profile — Who the user is · 1 facts" in listed.content


@pytest.mark.asyncio
async def test_a_failed_op_is_a_result_line_not_an_error(context: ToolContext) -> None:
    outcome = await run("apply_memory_ops", {"memory_store": "user", "ops": [{"op": "retire", "path": "x", "id": "zz"}]}, context)

    assert outcome.content.startswith("failed retire x: not_found:")
    assert outcome.summary == "0/1 applied"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("list_memories", {"memory_store": "nope"}, "memory_store_missing"),
        ("read_memories", {"memory_store": "nope", "paths": ["x"]}, "memory_store_missing"),
        ("apply_memory_ops", {"memory_store": "../x", "ops": [{"op": "add", "path": "x", "text": "t"}]}, "invalid_path"),
    ],
)
async def test_store_errors_become_tool_results(context: ToolContext, name: str, args: dict[str, Any], code: str) -> None:
    outcome = await run(name, args, context)

    assert outcome.content.startswith(f"error: {code}: ")
    assert outcome.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("apply_memory_ops", {"memory_store": "user", "ops": "add a fact"}),
        ("apply_memory_ops", {"memory_store": "user", "ops": []}),
        ("apply_memory_ops", {"memory_store": "user", "ops": ["add"]}),
        ("read_memories", {"memory_store": "user", "paths": "profile"}),
        ("read_memories", {"memory_store": "user", "paths": ["profile"], "ids": "no"}),
    ],
)
async def test_malformed_arguments_are_rejected(context: ToolContext, name: str, args: dict[str, Any]) -> None:
    outcome = await run(name, args, context)

    assert outcome.content.startswith("Tool call rejected:")
    assert outcome.summary == "Rejected"


@pytest.mark.asyncio
async def test_without_a_memory_root_the_tools_report_unavailable() -> None:
    outcome = await run("list_memories", {"memory_store": "user"}, ToolContext(model="test", opts=ToolOptions()))

    assert outcome.content.startswith("error: unavailable: ")


@pytest.mark.asyncio
async def test_the_store_runs_off_the_event_loop(context: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[bool] = []
    original = store.list_memories

    def recorded(*args: Any, **kwargs: Any) -> dict:
        threads.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(tools.memory_store, "list_memories", recorded)

    await run("list_memories", {"memory_store": "user"}, context)

    assert threads == [False]
