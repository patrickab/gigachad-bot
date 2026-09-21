# ruff: noqa
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

__all__ = ["PLOTLY_MEDIA_TYPE", "PlotContext", "create_sandbox_plot", "sandbox_plot_dependencies"]

logger = logging.getLogger(__name__)

PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"


class PlotContext(Protocol):
    client: LLMClient
    model: str
    user_msg: str
    sandbox_service: SandboxService | None
    chat_id: str
    tool_call_id: str
    prompt_images: tuple[PromptImage, ...]
    history: tuple[dict[str, Any], ...]


def sandbox_plot_dependencies() -> str:
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
{sandbox_plot_dependencies()}

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
Once `plot.json` is written, stop. Reply with a concise, layered, skimmable Markdown explanation: lead
with the essential takeaway, then add only the few details needed to support quick understanding. When
math helps, use proper Markdown LaTeX: `$...$` inline and `$$...$$` for display equations. Make no
further tool calls.
"""

_FAST_PLOT_SYSTEM = f"""\

# Task

The user is a visual learner and wants to understand a new concept. Optimize for visual understanding

- Write a self-contained Python script that creates a visually appealing didactic Plotly figure for the user's request.
- Prefer well-behaved, nontrivial examples with meaningful spatial structure. This creates illustrative examples.
- Never use unnecessary symmetry, singularities, extreme values, or discontinuities in illustrative examples unless they are essential to the concept.

- Optimize the plot for visual understanding and visual appeal.

# Available dependencies
{sandbox_plot_dependencies()}

# Figure design

- Use color schemes consistently across related panels.
- For unrelated panels, use distinct, complementary colors.
- Center titles and give the main title slightly more emphasis.
- Ensure sufficient spacing between text and plots, avoid excessive whitespace.
- For central variables, formulas, and values use clean, MathJax-safe LaTeX.

- For related views, use consistent ranges, aspect ratios, and visual encoding.
- Use concise titles/descriptions and readable axis labels with units where relevant.
- Use subplots when they make a comparison/concept-decomposition clearer.

- Do not mix unrelated color scales unless required by the data.
- Never hardcode the page background, because the host provides theme-aware background and typography.

# Script and output
- Assign the final figure to fig.
- Add concise imperative comments where appropriate for improving understandability for the user.
- End with print(fig.to_json()) and print nothing else.
- Return only the script, without prose or Markdown fences.
"""

_REPAIR_SYSTEM = """\
Fix the broken Plotly script. Keep the chart it was meant to draw, assign the figure to `fig`, and
end with `print(fig.to_json())` printing nothing else. Reply with the corrected script only: no
prose, no explanation, no code fences.
"""

_SCRIPT_FENCE = re.compile(r"```(?:python)?[^\S\n]*\n(?P<body>.*?)\n?```", re.DOTALL)


def _plot_script(text: str) -> str:
    """Take the fenced script when the model fences it, otherwise the whole reply."""
    match = _SCRIPT_FENCE.search(text)
    return match.group("body").strip() if match else text.strip()


def _completion_text(context: PlotContext, user_msg: str, system_prompt: str, **kwargs: object) -> str:
    """Run one blocking completion, surfacing a returned Exception as a raised one."""
    response = api_query_resilient(
        context.client,
        user_msg=user_msg,
        system_prompt=system_prompt,
        model=context.model,
        **kwargs,
    )
    if isinstance(response, Exception):
        raise response
    return response.choices[0].message.content or ""


def _generate_plot(context: PlotContext) -> str:
    """Ask the chat model for the plot script, guided by the styling prompt."""
    return _plot_script(
        _completion_text(
            context,
            context.user_msg,
            _FAST_PLOT_SYSTEM,
            user_msg_history=list(context.history),
        )
    )


def _repair_plot_script(context: PlotContext, script: str, error: str) -> str:
    """Ask for a corrected script, given the broken one and what the sandbox reported."""
    request = f"# Script\n{script}\n\n# Error\n{error}"
    return _plot_script(_completion_text(context, request, _REPAIR_SYSTEM))


def _figure_from_sandbox_outputs(outputs: tuple[str, ...]) -> dict[str, Any] | None:
    for output in outputs:
        try:
            figure = json.loads(output)
        except json.JSONDecodeError:
            continue
        if isinstance(figure, dict):
            return figure
    return None


async def _fast_plot(context: PlotContext) -> tuple[dict[str, Any], str] | None:
    """Generate and execute a disposable plot, repairing a broken script once."""
    try:
        script = await asyncio.to_thread(_generate_plot, context)
        for attempt in range(2):
            try:
                figure = json.loads(await context.sandbox_service.run_script(script))
                if not isinstance(figure, dict) or not isinstance(figure.get("data"), list):
                    raise ValueError("script did not print a Plotly figure")
                return figure, script
            except (SandboxScriptError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)[-2000:]
                logger.info("fast plot attempt failed: %s", error)
            if attempt == 0:
                script = await asyncio.to_thread(_repair_plot_script, context, script, error)
    except Exception:
        logger.exception("fast plot generation failed")
    return None


async def create_sandbox_plot(context: PlotContext) -> ToolOutcome:
    """Create a Plotly figure, then hand the answer round the code that drew it."""
    if context.sandbox_service is None:
        return ToolOutcome(
            content="The sandbox plot tool is not configured.",
            summary="Unavailable",
            error="sandbox_service_unavailable",
        )

    script = ""
    # Route prompt images through the agent, which alone can see them.
    fast = None if context.prompt_images else await _fast_plot(context)
    if fast is not None:
        figure, script = fast
    else:
        result = await context.sandbox_service.invoke(
            chat_id=context.chat_id,
            tool_call_id=context.tool_call_id,
            prompt=context.user_msg,
            prompt_images=context.prompt_images,
            append_system=_SANDBOX_PLOT_APPEND_SYSTEM,
            scope="sandbox_plot",
            thinking="low",
            lean=True,
        )
        if result.status != "completed":
            return ToolOutcome(
                content="The sandbox could not run: it failed before producing a figure. Explain that briefly.",
                summary=result.summary or "Failed",
                error=result.error or "sandbox_runner_error",
            )
        figure = _figure_from_sandbox_outputs(context.sandbox_service.output_texts(result, media_type=PLOTLY_MEDIA_TYPE))

    if figure is None:
        return ToolOutcome(
            content="The sandbox plot did not produce a valid figure. Explain that briefly or offer a simpler chart.",
            summary="Failed",
            error="invalid_figure",
        )

    traces = len(figure.get("data", [])) if isinstance(figure.get("data"), list) else 0
    # The code is the model's only view of the chart, so give it a focused explanation brief.
    seen = (
        f"The chart rendered and the user can already see it ({traces} trace(s)). "
        "Give an extremely concise companion explanation: explain how the"
        "underlying concept works and how the graphic encodes that mechanism. Make the graphic easier to grasp."
        "Do not explain every little detail - focus on whats important for the user to understand."
        "Use layered, skimmable layout with clean markdown-flavored latex.\n\n"
        f"```python\n{script}\n```"
        if script
        else f"Rendered an interactive plot with {traces} trace(s). The user can already see and interact with it. "
        "Give an extremely concise companion explanation of how the concept works and how the graphic encodes it. "
        "Do not explain code or restate the visible chart."
    )
    return ToolOutcome(
        content=seen,
        summary=f"{traces} trace{'s' if traces != 1 else ''}",
        detail={"figure": figure, "script": script},
    )
