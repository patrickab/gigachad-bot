"""Tests for the notebook execution harness (lib.sandbox_notebook)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from lib.sandbox_notebook import (
    NOTEBOOK_OUTPUT_LINE_PREFIX,
    NotebookNotFoundError,
    cell_output_key,
    run_notebook_cells,
    split_cells,
)


def _run_harness(code_cells: list[str]) -> str:
    """Execute the container harness locally and return its stdout."""
    from lib.sandbox_notebook import _harness_script

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(_harness_script(code_cells), "<harness>", "exec"), {"__name__": "__main__"})
    return buffer.getvalue()


class FakeSandboxService:
    """Stand-in service that executes the harness locally, no Docker."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.staged: list[tuple[str, str, dict]] = []
        self.run_scripts: list[str] = []

    async def read_notebook(self, chat_id: str) -> dict | None:
        return {"revision_id": "base", "source": self.source, "outputs": {}}

    async def run_script(self, script: str, *, timeout: float) -> str:
        self.run_scripts.append(script)
        # Extract the cells embedded in the script, then genuinely run the harness.
        cells = json.loads(script.split("CELLS = ", 1)[1].split("\n", 1)[0])
        return _run_harness(cells)

    async def stage_notebook(self, chat_id: str, revision_id: str, source: str, outputs: dict, *, expected_revision=None) -> None:
        assert expected_revision == "base"
        self.staged.append((revision_id, source, outputs))


async def test_markdown_cells_are_skipped_for_execution():
    source = (
        "# %% [markdown]\n"
        "# # Title\n"
        "\n"
        "# %%\n"
        "print('hello')\n"
    )
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    staged_cells = json.loads(service.run_scripts[0].split("CELLS = ", 1)[1].split("\n", 1)[0])
    assert staged_cells == ["print('hello')"]
    assert result["outputs"][cell_output_key("print('hello')")] == [{"type": "stdout", "text": "hello\n"}]


async def test_stdout_and_result_records():
    source = (
        "# %%\n"
        "print('hello')\n"
        "total = 1 + 1\n"
        "\n"
        "# %%\n"
        "total\n"
    )
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    stdout_records = result["outputs"][cell_output_key("print('hello')\ntotal = 1 + 1")]
    assert stdout_records == [{"type": "stdout", "text": "hello\n"}]
    result_records = result["outputs"][cell_output_key("total")]
    assert result_records == [{"type": "result", "text": "2"}]


async def test_error_cell_stops_run_and_keeps_earlier_outputs():
    source = (
        "# %%\n"
        "print('before')\n"
        "\n"
        "# %%\n"
        "1 / 0\n"
        "\n"
        "# %%\n"
        "print('after')\n"
    )
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 3)
    outputs = result["outputs"]
    assert outputs[cell_output_key("print('before')")] == [{"type": "stdout", "text": "before\n"}]
    error = outputs[cell_output_key("1 / 0")][0]
    assert error["type"] == "error"
    assert "ZeroDivisionError" in error["text"]
    assert "after" not in json.dumps(outputs)


async def test_outputs_keyed_by_source_sha256():
    source = "# %%\nprint('hello')\n"
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 1)
    assert list(result["outputs"].keys()) == [cell_output_key("print('hello')")]


async def test_upto_bounds_markdown_and_code_cells():
    source = (
        "# %% [markdown]\n"
        "# prose\n"
        "\n"
        "# %%\n"
        "print('one')\n"
        "\n"
        "# %%\n"
        "print('two')\n"
    )
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    assert list(result["outputs"].keys()) == [cell_output_key("print('one')")]
    with pytest.raises(ValueError):
        await run_notebook_cells(service, "chat-1", 4)
    with pytest.raises(ValueError):
        await run_notebook_cells(service, "chat-1", 0)


async def test_missing_notebook_raises():
    class EmptyService(FakeSandboxService):
        async def read_notebook(self, chat_id: str) -> dict | None:
            return None

    with pytest.raises(NotebookNotFoundError):
        await run_notebook_cells(EmptyService(""), "chat-1", 1)


async def test_revision_is_staged_with_source_unchanged():
    source = "# %%\nprint('hello')\n"
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 1)
    revision_id, staged_source, outputs = service.staged[0]
    assert staged_source == source
    assert revision_id == result["revision_id"]
    assert outputs == result["outputs"]


async def test_plotly_figure_record():
    pytest.importorskip("plotly")
    source = (
        "# %%\n"
        "import plotly.express as px\n"
        "fig = px.line(x=[1, 2], y=[3, 4])\n"
        "\n"
        "# %%\n"
        "fig\n"
    )
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    figure_cell = "fig"
    record = result["outputs"][cell_output_key(figure_cell)][0]
    assert record["type"] == "plotly"
    assert isinstance(record["figure"]["data"], list)
    # The verbose default template must not be inlined.
    assert "template" not in record["figure"].get("layout", {})


async def test_merge_outputs_rejects_stray_prefixed_lines():
    from lib.sandbox_notebook import _merge_outputs

    lines = (NOTEBOOK_OUTPUT_LINE_PREFIX + json.dumps({"stdout": "one\n"})) + "\n" + (
        NOTEBOOK_OUTPUT_LINE_PREFIX + json.dumps({"stdout": "two\n"})
    )
    with pytest.raises(ValueError):
        _merge_outputs(lines, ran_cells=["print('one')"], prior_outputs={})

async def test_unserializable_trailing_expression_degrades_to_repr():
    # A trailing `sys` (module) is not JSON-serializable; the run must not die.
    source = "# %%\nimport sys\n\n# %%\nsys\n"
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    record = result["outputs"][cell_output_key("sys")][0]
    assert record["type"] == "result"
    assert "<module" in record["text"]



async def test_dataframe_trailing_cell_is_not_mistaken_for_a_figure():
    pytest.importorskip("pandas")
    source = "# %%\nimport pandas as pd\n\n# %%\npd.DataFrame({'a': [1, 2]})\n"
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    record = result["outputs"][cell_output_key("pd.DataFrame({'a': [1, 2]})")][0]
    # Serialized as its repr, never as a bogus Plotly figure.
    assert record["type"] == "result"
    assert "a" in record["text"]

async def test_stale_namespace_fig_is_not_attached_to_later_cells():
    pytest.importorskip("plotly")
    source = (
        "# %%\n"
        "import plotly.express as px\n"
        "fig = px.line(x=[1], y=[1])\n"
        "\n"
        "# %%\n"
        "1 + 1\n"
    )
    service = FakeSandboxService(source)
    result = await run_notebook_cells(service, "chat-1", 2)
    record = result["outputs"][cell_output_key("1 + 1")][0]
    assert record == {"type": "result", "text": "2"}


async def test_run_with_concurrent_edit_conflicts_instead_of_reverting():
    class RacingService(FakeSandboxService):
        # Another writer lands between read and stage.
        async def read_notebook(self, chat_id: str) -> dict | None:
            return {"revision_id": "base", "source": self.source, "outputs": {}}

        async def stage_notebook(self, chat_id: str, revision_id: str, source: str, outputs: dict, *, expected_revision=None) -> None:
            from lib.sandbox_service import NotebookRevisionConflict

            assert expected_revision == "base"
            raise NotebookRevisionConflict("base")

    service = RacingService("# %%\nprint('hi')\n")
    from lib.sandbox_service import NotebookRevisionConflict

    with pytest.raises(NotebookRevisionConflict):
        await run_notebook_cells(service, "chat-1", 1)