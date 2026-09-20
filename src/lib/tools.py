"""Bind Gigachad's built-in tools to the generic one-call runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agent_sandbox import PromptImage
from llm_baseclient.client import LLMClient
from pydantic import BaseModel, Field

from lib.research import run_deep_research
from lib.sandbox_plot import create_sandbox_plot
from lib.sandbox_service import SandboxService
from lib.toolcalling import ToolCatalog, ToolDefinition, ToolOutcome, stream_tool_turn
from lib.web_search import brave_sources, evidence_prompt, plan_query, source_label

TOOL_GUIDANCE = (
    "Call tools yourself when a request needs them; never ask permission and never claim you cannot "
    "browse. A short, vague, or nonsense message is not automatically a lookup request — respond "
    "normally, or ask what the user means, instead of guessing at what it 'means' via search."
)


class ToolOptions(BaseModel):
    """Hold the per-request settings the browser already owns."""

    search_system_instructions: str = ""
    search_domain: str = ""
    research_fast_model: str = ""
    research_smart_model: str = ""
    research_strategic_model: str = ""
    research_depth: int = Field(default=2, ge=1, le=5)
    research_breadth: int = Field(default=4, ge=1, le=10)
    research_reasoning: str | None = None
    research_report_type: str = "deep"


@dataclass(frozen=True)
class ToolContext:
    """Provide one tool call with request-owned dependencies."""

    client: LLMClient
    model: str
    opts: ToolOptions
    chat_id: str = ""
    sandbox_service: SandboxService | None = None
    tool_call_id: str = ""
    prompt_images: tuple[PromptImage, ...] = ()
    small_model: str = ""


def _single_text_parameter(name: str, description: str, *, max_length: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "string", "minLength": 1, "description": description}
    if max_length is not None:
        value["maxLength"] = max_length
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [name],
        "properties": {name: value},
    }


async def _web_search(args: dict[str, Any], context: ToolContext) -> ToolOutcome:
    planner_model = context.small_model or context.model
    search_query, profile = await asyncio.to_thread(plan_query, context.client, args["query"], planner_model)
    domain = context.opts.search_domain.strip()
    if domain:
        search_query = f"{domain} {search_query}"
    sources = await brave_sources(search_query, profile)
    if not sources:
        return ToolOutcome(
            content="The web search returned no usable source content. Say so instead of guessing.",
            summary="No usable sources",
            error="Brave found no usable source content for this query.",
            detail={"search_query": search_query},
        )
    return ToolOutcome(
        content="\n\n".join(filter(None, [context.opts.search_system_instructions.strip(), evidence_prompt(sources)])),
        summary=f"{len(sources)} sources",
        sources=sources,
        detail={"search_query": search_query},
    )


async def _deep_research(args: dict[str, Any], context: ToolContext) -> ToolOutcome:
    report, urls, costs = await run_deep_research(
        args["query"],
        fast_model=context.opts.research_fast_model,
        smart_model=context.opts.research_smart_model,
        strategic_model=context.opts.research_strategic_model,
        depth=context.opts.research_depth,
        breadth=context.opts.research_breadth,
        reasoning_effort=context.opts.research_reasoning,
        report_type=context.opts.research_report_type,
        fallback_model=context.model,
    )
    return ToolOutcome(
        content=report or "Deep research produced no report. Say so instead of guessing.",
        summary=f"{len(urls)} sources",
        sources=[{"label": source_label(url, set()), "url": url, "title": "", "content": ""} for url in urls],
        detail={"costs": costs, "report": report},
    )

async def _sandbox_plot(args: dict[str, Any], context: ToolContext) -> ToolOutcome:
    return await create_sandbox_plot(args["brief"], context)


async def _workspace_agent(args: dict[str, Any], context: ToolContext) -> ToolOutcome:
    if context.sandbox_service is None:
        return ToolOutcome(
            content="The workspace agent tool is not configured.",
            summary="Unavailable",
            error="sandbox_service_unavailable",
        )
    result = await context.sandbox_service.invoke(
        chat_id=context.chat_id,
        tool_call_id=context.tool_call_id,
        prompt=args["prompt"],
        prompt_images=context.prompt_images,
        scope="workspace_agent",
    )
    return ToolOutcome(
        content=result.summary,
        summary=result.summary,
        sandbox={
            "status": result.status,
            "manifest_id": f"sha256:{result.manifest_id}" if result.manifest_id else None,
            "workspace_changed": result.workspace_changed,
            "outputs": [output.to_json() for output in result.outputs],
        },
        error=result.error,
    )


BUILTIN_TOOLS: ToolCatalog[ToolContext] = ToolCatalog(
    (
        ToolDefinition(
            "web_search",
            "Search the live web and return labelled source evidence. Use for current events, fast-moving "
            "facts, specific documentation, and any claim that should be cited. Do not use this to guess "
            "at the meaning of a short, vague, or one-word message like a greeting or test message — only "
            "search when the user's request clearly needs outside information.",
            _single_text_parameter("query", "What to look up, phrased as a self-contained search request."),
            _web_search,
        ),
        ToolDefinition(
            "deep_research",
            "Run a multi-step research agent that browses many sources and writes a cited report. Takes "
            "minutes. Use only for broad or multi-part investigations, never for a single fact.",
            _single_text_parameter("query", "The research question, stated as a full topic rather than keywords."),
            _deep_research,
        ),
        ToolDefinition(
            "sandbox_plot",
            "Create or revise an interactive Plotly chart in a persistent, chat-scoped sandbox workspace. "
            "Use when the user asks for a chart, graph, or visual data analysis. Give a concise chart brief.",
            _single_text_parameter(
                "brief",
                "Concise, self-contained instructions for the chart to create or revise.",
                max_length=12000,
            ),
            _sandbox_plot,
        ),
        ToolDefinition(
            "workspace_agent",
            "Perform work in this chat's persistent code workspace: write and edit files, run code, "
            "and produce charts, tables, or other rich artifacts. The workspace persists between calls "
            "within this chat. Give one clear natural-language instruction describing the work to do.",
            _single_text_parameter(
                "prompt",
                "Describe the work to perform in the chat workspace.",
                max_length=12000,
            ),
            _workspace_agent,
        ),
    )
)


async def stream_chat_with_tools(
    *,
    client: LLMClient,
    model: str,
    user_msg: str,
    history: list[dict[str, Any]],
    system_prompt: str,
    img: Any = None,
    enabled: list[str],
    opts: ToolOptions,
    chat_id: str = "",
    sandbox_service: SandboxService | None = None,
    prompt_images: tuple[PromptImage, ...] = (),
    small_model: str = "",
    **kwargs: Any,
) -> AsyncIterator[tuple[str, Any]]:
    """Bind request dependencies, then stream one optional tool call and the answer."""

    def context_for_call(tool_call_id: str) -> ToolContext:
        return ToolContext(
            client=client,
            model=model,
            opts=opts,
            chat_id=chat_id,
            sandbox_service=sandbox_service,
            tool_call_id=tool_call_id,
            prompt_images=prompt_images,
            small_model=small_model,
        )

    async for event in stream_tool_turn(
        client=client,
        model=model,
        user_msg=user_msg,
        history=history,
        system_prompt=system_prompt,
        img=img,
        enabled=enabled,
        catalog=BUILTIN_TOOLS,
        context_for_call=context_for_call,
        guidance=TOOL_GUIDANCE,
        **kwargs,
    ):
        yield event
