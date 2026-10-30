"""Bind Gigachad's built-in tools to the generic one-call runtime."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from agent_sandbox import PromptImage
from llm_baseclient.client import LLMClient

from lib import learnharness
from lib.data_store import StorageConflictError
from lib.llm_resilience import completion_text
from lib.prompts.non_user_prompts import SYS_DIAGRAM_MERMAID, SYS_OCR_TEXT_EXTRACTION, SYS_STUDY_MINDMAP
from lib.research import run_deep_research
from lib.sandbox_plot import create_sandbox_plot
from lib.sandbox_service import PROMPT_MAX_LEN, WORKSPACE_AGENT_SCOPE, SandboxService
from lib.tool_context import ToolContext, ToolOptions
from lib.toolcalling import Stage, ToolCatalog, ToolDefinition, ToolOutcome, sandbox_unavailable, stream_tool_turn
from lib.upstream import UpstreamContractError, UpstreamRequestFailed, UpstreamUnavailable
from lib.web_search import brave_sources, evidence_prompt, plan_query, source_label

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from lib.project_store import ProjectStore

TOOL_GUIDANCE = (
    "Call tools yourself when a request needs them; never ask permission and never claim you cannot "
    "browse. A short, vague, or nonsense message is not automatically a lookup request — respond "
    "normally, or ask what the user means, instead of guessing at what it 'means' via search."
)

# Tools that act on the machine owner's host, offered only to logins with the host capability.
HOST_TOOLS = frozenset({"start_learn_session"})


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


_NO_ARGUMENTS: dict[str, Any] = {"type": "object", "additionalProperties": False, "properties": {}}


async def _web_search(args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    stage("Planning search")
    planner_model = context.small_model or context.model
    search_query, profile = await asyncio.to_thread(plan_query, context.client, args["query"], planner_model)
    # Domain scoping is a plain keyword prefix on the planned query.
    domain = context.opts.search_domain.strip()
    if domain:
        search_query = f"{domain} {search_query}"
    stage("Searching sources")
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


async def _deep_research(args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    stage("Researching sources")
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
        # Research returns bare URLs, so sources carry no title or content.
        sources=[{"label": source_label(url, set()), "url": url, "title": "", "content": ""} for url in urls],
        detail={"costs": costs, "report": report},
    )


async def _sandbox_plot(_args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    stage("Generating chart")
    if context.sandbox_service is None:
        return sandbox_unavailable("sandbox plot")
    return await create_sandbox_plot(context, context.sandbox_service)


async def _workspace_agent(args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    if context.sandbox_service is None:
        return sandbox_unavailable("workspace agent")
    stage("Running workspace agent")
    result = await context.sandbox_service.invoke(
        chat_id=context.chat_id,
        tool_call_id=context.tool_call_id,
        prompt=args["prompt"],
        prompt_images=context.prompt_images,
        scope=WORKSPACE_AGENT_SCOPE,
    )
    return ToolOutcome(content=result.summary, summary=result.summary, sandbox=result.chat_payload(), error=result.error)


# Whole-reply fenced block, capturing the body without the fence lines.
_FENCED_CODE = re.compile(r"```[^\n]*\n(?P<body>.*?)\n?```", re.DOTALL)


async def _transcript_completion(context: ToolContext, instruction: str, system_prompt: str) -> str:
    """Run one completion over the whole conversation so far and return the stripped reply."""
    transcript = "\n".join(
        f"[{message.get('role', 'user')}]: {message.get('content', '')}"
        for message in (*context.history, {"role": "user", "content": context.user_msg})
    )
    text = await asyncio.to_thread(
        completion_text,
        context.client,
        model=context.model,
        user_msg=f"{instruction}\n\n<transcript>\n{transcript}\n</transcript>",
        user_msg_history=[],
        system_prompt=system_prompt,
        img=None,
        stream=False,
    )
    return text.strip()


async def _diagram(_args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    stage("Creating diagram")
    content = await _transcript_completion(context, "Create a Mermaid diagram from this conversation.", SYS_DIAGRAM_MERMAID)
    # Strip a Markdown fence if the model wrapped the source despite the prompt.
    if match := _FENCED_CODE.fullmatch(content):
        content = match.group("body").strip()
    if not content:
        raise ValueError("The diagram generator returned no Mermaid source.")
    # Tells the model not to repeat the diagram it can't see, since the UI renders `detail["mermaid"]`.
    seen = (
        "The Mermaid diagram rendered above this response and the user can already see it. Do not create, "
        "return, or repeat Mermaid or any other diagram.\n\n"
        f"```mermaid\n{content}\n```"
    )
    return ToolOutcome(content=seen, summary="Diagram ready", detail={"mermaid": content})


async def _mindmap(_args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    stage("Creating mind map")
    mindmap = await _transcript_completion(context, "Produce a mind map from this conversation.", SYS_STUDY_MINDMAP)
    return ToolOutcome(content=mindmap, summary="Mind map ready", detail={"mindmap": mindmap})


async def _latex_ocr(_args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    if context.img is None:
        return ToolOutcome(
            content="LaTeX OCR needs an attached image.",
            summary="No image attached",
            error="Attach an image before requesting LaTeX OCR.",
        )
    stage("Extracting text and LaTeX")
    text = await asyncio.to_thread(
        completion_text,
        context.client,
        model=context.vision_model or context.model,
        user_msg="Extract all text and LaTeX from this image.",
        user_msg_history=[],
        system_prompt=SYS_OCR_TEXT_EXTRACTION,
        img=context.img,
        stream=False,
    )
    text = text.strip()
    return ToolOutcome(content=text, summary="Text extracted", detail={"text": text})


# A goal's first sentence titles its learn session, LearnHarness shortens it further.
_FIRST_SENTENCE = re.compile(r"(?<=[.!?])\s")


async def _start_learn_session(args: dict[str, Any], context: ToolContext, stage: Stage) -> ToolOutcome:
    goal = args["goal"]
    stage("Creating learn session")
    session = await learnharness.create_session(_FIRST_SENTENCE.split(goal, maxsplit=1)[0])
    store, project_slug = context.project_store, context.project_slug
    ungrouped = ""
    if store is None or project_slug is None:
        project_slug = None
    else:
        try:
            await asyncio.to_thread(store.assign_learn_session, session.id, project_slug)
        except (FileNotFoundError, StorageConflictError) as exc:
            # The session exists either way: link it ungrouped (it shows under Learn) and start it, never orphan it.
            ungrouped = f"It could not join the chat's project ({exc}), so it is listed under Learn on its own. "
            project_slug = None
    detail = {"session_id": session.id, "title": session.title, "project_slug": project_slug}
    stage("Starting the tutor")
    try:
        await learnharness.submit_text_turn(session.id, goal)
    except (UpstreamUnavailable, UpstreamRequestFailed, UpstreamContractError) as exc:
        return ToolOutcome(
            content=f'The learn session "{session.title}" was created, but its tutor did not start: {exc}. {ungrouped}'
            "The user can open it from the link above and send the goal again.",
            summary="Tutor not started",
            detail=detail,
            error=str(exc),
        )
    return ToolOutcome(
        content=f'Started the learn session "{session.title}": its tutor is already working on the goal. {ungrouped}The user '
        "opens it from the link shown above this reply, so write no link or URL. Say so in one or two sentences, do not start "
        "teaching here.",
        summary="Session started" if not ungrouped else "Session started outside the project",
        detail=detail,
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
            "Use when the user asks for a chart, graph, or visual data analysis. Takes no arguments: the "
            "conversation already carries the request.",
            _NO_ARGUMENTS,
            _sandbox_plot,
        ),
        ToolDefinition(
            "diagram",
            "Create a Mermaid diagram directly in the chat. Use whenever the user asks for a diagram, "
            "flowchart, sequence diagram, process map, architecture diagram, or visual relationship map. "
            "It may also be used whenever a diagram would significantly reduce the text needed to answer "
            "the user's request. Takes no arguments because the conversation already supplies the request.",
            _NO_ARGUMENTS,
            _diagram,
        ),
        ToolDefinition(
            "mindmap",
            "Create a visual mind map of the current conversation when the user asks to organize, map, or visualize its ideas. "
            "Takes no arguments because the conversation already supplies the material.",
            _NO_ARGUMENTS,
            _mindmap,
            # The mind map card is the whole reply, so no answer round follows.
            answers=False,
        ),
        ToolDefinition(
            "latex_ocr",
            "Extract exact Markdown text and LaTeX from the attached image. Use only when an image is attached and the user "
            "asks to transcribe, read, or extract equations from it. Takes no arguments.",
            _NO_ARGUMENTS,
            _latex_ocr,
        ),
        ToolDefinition(
            "workspace_agent",
            "Perform work in this chat's persistent code workspace: write and edit files, run code, "
            "and produce charts, tables, or other rich artifacts. The workspace persists between calls "
            "within this chat. Give one clear natural-language instruction describing the work to do.",
            _single_text_parameter("prompt", "Describe the work to perform in the chat workspace.", max_length=PROMPT_MAX_LEN),
            _workspace_agent,
        ),
        ToolDefinition(
            "start_learn_session",
            "Start a learn session in Learn mode, where a tutor probes what the user knows, plans and teaches the topic "
            "with quizzes and demos. Use when the user asks to learn, study, or master a topic properly or in depth, or to "
            "be taught, tutored, or quizzed on it, rather than to get an answer here.",
            _single_text_parameter(
                "goal",
                "The learning goal in a few sentences, condensed from the conversation: the topic, what the user already "
                "knows or struggles with, and what they want to be able to do. Written as the user, addressed to the tutor.",
            ),
            _start_learn_session,
        ),
    )
)


def offered_tools(enabled: list[str], *, host_access: bool) -> list[str]:
    """The tools of *enabled* this login may be offered: `HOST_TOOLS` only with the host capability."""
    return enabled if host_access else [name for name in enabled if name not in HOST_TOOLS]


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
    vision_model: str = "",
    host_access: bool = False,
    project_slug: str | None = None,
    project_store: ProjectStore | None = None,
    **kwargs: Any,
) -> AsyncIterator[tuple[str, Any]]:
    """Bind request dependencies, then stream one optional tool call and the answer.

    *host_access*: the login holds the host capability, without it `HOST_TOOLS` are never offered.
    """

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
            history=tuple(history),
            user_msg=user_msg,
            img=img,
            vision_model=vision_model,
            project_slug=project_slug,
            project_store=project_store,
        )

    async for event in stream_tool_turn(
        client=client,
        model=model,
        user_msg=user_msg,
        history=history,
        system_prompt=system_prompt,
        img=img,
        enabled=offered_tools(enabled, host_access=host_access),
        catalog=BUILTIN_TOOLS,
        context_for_call=context_for_call,
        guidance=TOOL_GUIDANCE,
        **kwargs,
    ):
        yield event
