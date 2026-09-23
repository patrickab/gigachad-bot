"""Execute chat-scoped percent-format notebooks through the throwaway sandbox.

A notebook is a string of `# %%`-separated cells (percent format, matching
src/frontend/lib/notebook.ts). Markdown cells annotate prose and never run;
code cells share one namespace and report stdout, a trailing expression value,
a Plotly figure, or an error. An erroring cell stops the run while everything
captured before it is kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = [
    "NOTEBOOK_OUTPUT_LINE_PREFIX",
    "NOTEBOOK_RUN_TIMEOUT_SECONDS",
    "NotebookNotFoundError",
    "cell_output_key",
    "run_notebook_cells",
    "split_cells",
]

# One container run covers every code cell up to `upto`.
NOTEBOOK_RUN_TIMEOUT_SECONDS = 120.0

# Harness result lines carry a prefix so stray container output cannot
# impersonate cell results.
NOTEBOOK_OUTPUT_LINE_PREFIX = "## note:out:"

_MARKDOWN_MARKER = "# %% [markdown]"
_MARKER_RE = re.compile(r"^\s*#\s*%%")

_MAX_INLINE_TEXT_CHARS = 8000
_MAX_FIGURE_JSON_CHARS = 20000


class NotebookNotFoundError(LookupError):
    """Raised when a chat has no notebook to run."""


def split_cells(source: str) -> list[tuple[str, str]]:
    """Split percent format into (kind, source) cells; kind is code|markdown."""
    lines = source.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        # Mirror the frontend parser exactly: trimmed-line equality, so titled
        # markers and `#%%` stay inside cell sources on both sides.
        if line.strip() == _MARKDOWN_MARKER:
            starts.append((index, "markdown"))
        elif line.strip() == "# %%":
            starts.append((index, "code"))
    if not starts:
        return [("code", source.strip("\n"))]
    cells: list[tuple[str, str]] = []
    # Non-blank content before the first marker is an implicit leading code cell.
    if any(line.strip() for line in lines[: starts[0][0]]):
        cells.append(("code", "\n".join(lines[: starts[0][0]]).strip("\n")))
    for position, (start, kind) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        cells.append((kind, "\n".join(lines[start + 1 : end]).strip("\n")))
    return cells


def cell_output_key(source: str) -> str:
    """Sidecar key: sha256 of the cell source."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# The harness runs inside the container via `python -c <script>` with no
# stdin, so the cell sources are spliced in as a JSON literal between the
# two halves below (JSON list literals are valid Python).
_HARNESS_PRE = """\
import ast, io, json
from contextlib import redirect_stdout

CELLS = """

_HARNESS_POST = '''
PREFIX = %r
ns = {}
for cell in CELLS:
    result = {}
    try:
        tree = ast.parse(cell, mode="exec")
        body = tree.body
        expr = body.pop() if body and isinstance(body[-1], ast.Expr) else None
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(compile(tree, "<cell>", "exec"), ns)  # noqa: S102
            if expr is not None:
                result["result"] = eval(compile(ast.Expression(expr.value), "<expr>", "eval"), ns)  # noqa: S102
        stdout = buffer.getvalue()
        if stdout:
            result["stdout"] = stdout
        # Only the cell's own trailing figure expression counts, and only a real
        # Plotly type: duck-typed `to_json` misclassifies e.g. a pandas DataFrame.
        value = result.get("result")
        if type(value).__module__.startswith("plotly."):
            figure = json.loads(value.to_json())
            # Plotly ships a huge default template; the UI re-applies its own.
            figure.get("layout", {}).pop("template", None)
            result["plotly"] = figure
            result.pop("result", None)
    except BaseException as exc:  # noqa: BLE001 - cell code may raise anything
        result.clear()
        result["error"] = {"ename": type(exc).__name__, "evalue": str(exc)}
    # Unserializable results degrade to their repr, never a dead run.
    try:
        line = json.dumps(result)
    except (TypeError, ValueError):
        if "result" in result:
            result["result"] = repr(result["result"])
        line = json.dumps(result)
    print(PREFIX + line, flush=True)
    if "error" in result:
        break
''' % NOTEBOOK_OUTPUT_LINE_PREFIX


def _harness_script(code_cells: list[str]) -> str:
    """Build the container script with the code cells embedded as JSON."""
    return _HARNESS_PRE + json.dumps(code_cells) + _HARNESS_POST


def _trim(text: str) -> str:
    return text if len(text) <= _MAX_INLINE_TEXT_CHARS else text[:_MAX_INLINE_TEXT_CHARS] + "\u2026 [truncated]"


def _cell_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Shape one harness result into small UI-renderable output records."""
    if error := result.get("error"):
        return [{"type": "error", "text": f"{error['ename']}: {error['evalue']}"}]
    records: list[dict[str, Any]] = []
    if stdout := result.get("stdout"):
        records.append({"type": "stdout", "text": _trim(str(stdout))})
    if figure := result.get("plotly"):
        if not isinstance(figure, dict) or len(json.dumps(figure)) > _MAX_FIGURE_JSON_CHARS:
            return [{"type": "error", "text": "cell figure could not be rendered"}]
        records.append({"type": "plotly", "figure": figure})
    elif "result" in result and result["result"] is not None:
        records.append({"type": "result", "text": _trim(repr(result["result"]))})
    return records


def _merge_outputs(stdout: str, ran_cells: list[str], prior_outputs: dict[str, Any]) -> dict[str, Any]:
    """Fold prefixed harness result lines into the sidecar keyed by cell sha256."""
    outputs = dict(prior_outputs)
    lines = [line for line in stdout.splitlines() if line.startswith(NOTEBOOK_OUTPUT_LINE_PREFIX)]
    if len(lines) > len(ran_cells):
        raise ValueError("notebook harness emitted more results than cells")
    for line, source in zip(lines, ran_cells, strict=False):
        try:
            result = json.loads(line[len(NOTEBOOK_OUTPUT_LINE_PREFIX) :])
        except json.JSONDecodeError:
            raise ValueError("notebook harness emitted an unreadable result line") from None
        records = _cell_records(result)
        key = cell_output_key(source)
        if records:
            outputs[key] = records
        else:
            outputs.pop(key, None)
    return outputs


async def run_notebook_cells(
    sandbox_service: Any,
    chat_id: str,
    upto: int,
    *,
    timeout: float = NOTEBOOK_RUN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the first `upto` code cells in one fresh container; stage a revision.

    Markdown cells are skipped. A failing cell stops the run, records its
    error, and leaves later cells' previous outputs untouched. Returns
    {"revision_id": <opaque id>, "outputs": <sidecar keyed by cell sha256>}.
    """
    notebook = await sandbox_service.read_notebook(chat_id)
    if notebook is None:
        raise NotebookNotFoundError(chat_id)

    cells = split_cells(notebook["source"])
    if upto < 1:
        raise ValueError("upto must be a positive cell number")
    if upto > len(cells):
        raise ValueError(f"upto={upto} exceeds the {len(cells)} cells in the notebook")
    # `upto` indexes all cells; markdown cells annotate prose and are skipped.
    ran_cells = [source for kind, source in cells[:upto] if kind == "code"]

    outputs = dict(notebook.get("outputs") or {})
    if ran_cells:
        stdout = await sandbox_service.run_script(_harness_script(ran_cells), timeout=timeout)
        outputs = _merge_outputs(stdout, ran_cells, outputs)

    revision_id = uuid4().hex
    await sandbox_service.stage_notebook(
        chat_id=chat_id,
        revision_id=revision_id,
        source=notebook["source"],
        outputs=outputs,
        # A concurrent edit that landed while the container ran must not be reverted.
        expected_revision=notebook["revision_id"],
    )
    logger.info("ran %d notebook cells for chat %s", len(ran_cells), chat_id)
    return {"revision_id": revision_id, "outputs": outputs}