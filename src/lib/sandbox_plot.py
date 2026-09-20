"""Create interactive Plotly figures through the fast or persistent sandbox path."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import tomllib
from typing import Any, Protocol

from agent_sandbox import PromptImage
from llm_baseclient.client import LLMClient

from lib.agent_sandbox_adapter import SandboxScriptError
from lib.llm_resilience import api_query_resilient
from lib.sandbox_service import SandboxService
from lib.toolcalling import ToolOutcome

__all__ = ["PLOTLY_MEDIA_TYPE", "PlotContext", "create_sandbox_plot"]

logger = logging.getLogger(__name__)

PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"


class PlotContext(Protocol):
    client: LLMClient
    model: str
    small_model: str
    sandbox_service: SandboxService | None
    chat_id: str
    tool_call_id: str
    prompt_images: tuple[PromptImage, ...]


def _sandbox_plot_dependencies() -> str:
    """Read the packages installed by the sandbox-plot dependency extra."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject:
        project = tomllib.load(pyproject)["project"]
    dependencies = project["optional-dependencies"]["sandbox-plot"]
    return ", ".join(re.split(r"[\s<>=!~\[@]", dependency, maxsplit=1)[0] for dependency in dependencies)


_SANDBOX_PLOT_APPEND_SYSTEM = f"""\
# Plot workspace protocol
Work iteratively in the sandbox workspace. Create or update `plot.py` for this request; on later
calls, revise that same file rather than starting elsewhere. Use the dependencies below and run the
script to inspect and correct the chart before finishing.

# Available Dependencies
{_sandbox_plot_dependencies()}

# Figure requirements
- Use numpy as np, plotly.express as px, and plotly.graph_objects as go.
- Assign the final Plotly figure to `fig`.
- Use appropriate colours, labels, and a concise legend. Prefer Plasma for continuous colour scales.
- Use a clear title and labelled axes.
- Design for a narrow chat card. Do not set figure width or height or use three or more side-by-side panels.

# Required plot artifact
Write the final figure JSON to `plot.json` in the workspace with `fig.to_json()`. Do not print figure
JSON or implementation details in your final response. For example:
```
with open("plot.json", "w") as output:
    output.write(fig.to_json())
```
Once `plot.json` is written, stop. Reply with one short sentence and no further tool calls.
"""

_FAST_PLOT_SYSTEM = f"""\
# Task
Write one Python script that builds a single Plotly figure for the chart brief.

# Available dependencies
{_sandbox_plot_dependencies()}

# Requirements
- Assign the final Plotly figure to `fig`.
- Use a clear title, labelled axes, and a concise legend. Prefer Plasma for continuous colour scales.
- Do not set figure width, height, or autosize: the chart renders in a narrow chat card.
- The script runs offline with no network and no input files. Derive or define any data it needs.
- End with `print(fig.to_json())` and print nothing else.

Reply with the script only: no prose, no explanation, no code fences.
"""

_SCRIPT_FENCE = re.compile(r"\A```(?:python)?\s*\n(?P<body>.*?)\n?```\s*\Z", re.DOTALL)


def _plot_script(text: str) -> str:
    """Take the script out of a model reply that may still be fenced."""
    match = _SCRIPT_FENCE.match(text.strip())
    return (match.group("body") if match else text).strip()


def _generate_plot_script(context: PlotContext, brief: str, previous_error: str = "") -> str:
    request = brief if not previous_error else f"{brief}\n\n# Previous attempt failed\n{previous_error}\n\nReturn a corrected script."
    response = api_query_resilient(
        context.client,
        user_msg=request,
        user_msg_history=[],
        system_prompt=_FAST_PLOT_SYSTEM,
        model=context.small_model or context.model,
    )
    if isinstance(response, Exception):
        raise response
    return _plot_script(response.choices[0].message.content or "")


def _figure_from_sandbox_outputs(outputs: tuple[str, ...]) -> dict[str, Any] | None:
    for output in outputs:
        try:
            figure = json.loads(output)
        except json.JSONDecodeError:
            continue
        if isinstance(figure, dict):
            return figure
    return None


async def _fast_plot(context: PlotContext, brief: str) -> tuple[dict[str, Any], str] | None:
    """Generate and execute a disposable plot, or hand off to the persistent agent."""
    error = ""
    for _attempt in range(2):
        try:
            script = await asyncio.to_thread(_generate_plot_script, context, brief, error)
            figure = json.loads(await context.sandbox_service.run_script(script))
            if not isinstance(figure, dict) or not isinstance(figure.get("data"), list):
                raise ValueError("script did not print a Plotly figure")
            return figure, script
        except (SandboxScriptError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)[-2000:]
            logger.info("fast plot attempt failed: %s", error)
        except Exception:
            logger.exception("fast plot generation failed")
            return None
    return None


async def create_sandbox_plot(brief: str, context: PlotContext) -> ToolOutcome:
    """Create a Plotly figure, preferring the fast path for self-contained briefs."""
    if context.sandbox_service is None:
        return ToolOutcome(
            content="The sandbox plot tool is not configured.",
            summary="Unavailable",
            error="sandbox_service_unavailable",
        )

    script = ""
    # Route prompt images through the agent, which alone can see them.
    fast = None if context.prompt_images else await _fast_plot(context, brief)
    if fast is not None:
        figure, script = fast
    else:
        result = await context.sandbox_service.invoke(
            chat_id=context.chat_id,
            tool_call_id=context.tool_call_id,
            prompt=brief,
            prompt_images=context.prompt_images,
            append_system=_SANDBOX_PLOT_APPEND_SYSTEM,
            scope="sandbox_plot",
            thinking="low",
            lean=True,
        )
        figure = (
            _figure_from_sandbox_outputs(context.sandbox_service.output_texts(result, media_type=PLOTLY_MEDIA_TYPE))
            if result.status == "completed"
            else None
        )

    if figure is None:
        return ToolOutcome(
            content="The sandbox plot did not produce a valid figure. Explain that briefly or try a clearer chart brief.",
            summary="Failed",
            error="invalid_figure",
        )

    traces = len(figure.get("data", [])) if isinstance(figure.get("data"), list) else 0
    return ToolOutcome(
        content=f"Rendered an interactive plot with {traces} trace(s). The user can already see and interact with it.",
        summary=f"{traces} trace{'s' if traces != 1 else ''}",
        detail={"figure": figure, "brief": brief, "script": script},
    )
