"""Offer, parse, dispatch, and answer with at most one tool call per turn."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, TypedDict
from uuid import uuid4

import litellm
from llm_baseclient.client import LLMClient

from lib.async_utils import aiter_in_thread
from lib.llm_json import extract_json_from_llm

__all__ = [
    "Source",
    "Stage",
    "ToolCatalog",
    "ToolDefinition",
    "ToolOutcome",
    "sandbox_unavailable",
    "stream_tool_turn",
]

type Stage = Callable[[str], None]


class Source(TypedDict):
    """One cited web source; `content` is the evidence text the model sees, never the browser."""

    label: str
    url: str
    title: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model, native or recovered from text."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolOutcome:
    """Keep model-visible content separate from browser-only metadata."""

    content: str
    summary: str = ""
    sources: list[Source] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    sandbox: dict[str, Any] | None = None

    def result_event(self, call: ToolCall) -> dict[str, Any]:
        """The browser's `tool_result` payload: everything except model content and source evidence."""
        return {
            "id": call.id,
            "name": call.name,
            "summary": self.summary,
            "sources": [{key: value for key, value in source.items() if key != "content"} for source in self.sources],
            "detail": self.detail,
            "error": self.error,
            "sandbox": self.sandbox,
        }


def sandbox_unavailable(tool: str) -> ToolOutcome:
    """Report that a sandbox-backed tool has no sandbox service configured."""
    return ToolOutcome(content=f"The {tool} tool is not configured.", summary="Unavailable", error="sandbox_service_unavailable")


@dataclass(frozen=True)
class ToolProgress:
    """A snapshot of a tool's stage timeline, safe to send over SSE."""

    stages: tuple[dict[str, Any], ...]


class StageTimeline:
    """Record a tool's stages, closing the running one whenever the next begins or the tool ends."""

    def __init__(self) -> None:
        self._stages: list[dict[str, Any]] = []

    def _close(self, status: str, now: float) -> None:
        if self._stages and self._stages[-1]["status"] == "running":
            self._stages[-1]["status"] = status
            self._stages[-1]["duration"] = now - self._stages[-1]["started_at"]

    def _snapshot(self) -> ToolProgress:
        return ToolProgress(stages=tuple(dict(stage) for stage in self._stages))

    def begin(self, label: str) -> ToolProgress:
        """Finish the running stage and start *label*."""
        now = time.time()
        self._close("done", now)
        self._stages.append(
            {"id": f"stage-{len(self._stages) + 1}", "label": label, "status": "running", "started_at": now, "duration": 0.0}
        )
        return self._snapshot()

    def finish(self, *, error: bool) -> ToolProgress:
        """Close the running stage as failed or done."""
        self._close("error" if error else "done", time.time())
        return self._snapshot()


def _no_stage(_label: str) -> None:
    """Discard stage reports when no timeline listens."""


@dataclass(frozen=True)
class ToolDefinition[ContextT]:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any], ContextT, Stage], Awaitable[ToolOutcome]]
    # False when the tool result is the whole reply, so no answer round follows.
    answers: bool = True

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _argument_error(parameters: dict[str, Any], args: dict[str, Any]) -> tuple[str, str] | None:
    """Check a call against its declared schema: required, unknown, string type, declared length."""
    properties: dict[str, Any] = parameters.get("properties") or {}
    missing = [key for key in parameters.get("required", ()) if not str(args.get(key, "")).strip()]
    if missing:
        detail = ", ".join(f"`{key}`" for key in missing)
        return f"Tool call rejected: {detail} required.", f"Missing argument: {detail}."
    unknown = [key for key in args if key not in properties] if parameters.get("additionalProperties") is False else []
    if unknown:
        detail = ", ".join(f"`{key}`" for key in unknown)
        return f"Tool call rejected: {detail} not accepted.", f"Unknown argument: {detail}."
    for key, value in args.items():
        schema = properties.get(key)
        if not isinstance(schema, dict) or schema.get("type") != "string":
            continue
        if not isinstance(value, str):
            return f"Tool call rejected: `{key}` must be text.", f"Invalid argument: `{key}` must be a string."
        limit = schema.get("maxLength")
        # Handlers see stripped strings, so the limit applies to what they will actually receive.
        if isinstance(limit, int) and len(value.strip()) > limit:
            return f"Tool call rejected: `{key}` is longer than {limit} characters.", f"Argument too long: `{key}`."
    return None


class ToolCatalog[ContextT]:
    """Offer and execute a closed set of tool definitions."""

    def __init__(self, definitions: Iterable[ToolDefinition[ContextT]]) -> None:
        self.definitions = tuple(definitions)
        self._by_name = {definition.name: definition for definition in self.definitions}

    def get(self, name: str) -> ToolDefinition[ContextT] | None:
        return self._by_name.get(name)

    def specs(self, names: list[str]) -> list[dict[str, Any]]:
        """Drop unknown names so callers cannot widen the offered tool surface."""
        return [self._by_name[name].spec for name in dict.fromkeys(names) if name in self._by_name]

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: ContextT,
        *,
        enabled: Iterable[str] | None = None,
        stage: Stage = _no_stage,
    ) -> ToolOutcome:
        """Contain gating, validation and handler failures so the stream can continue."""
        definition = self._by_name.get(name)
        if definition is None:
            message = f"Unknown tool `{name}`."
            return ToolOutcome(content=message, summary="Unknown tool", error=message)
        # `enabled` is the turn's offered set; registration alone must never authorise execution.
        if enabled is not None and name not in set(enabled):
            message = f"Tool `{name}` was not enabled for this turn."
            return ToolOutcome(content=message, summary="Not enabled", error=message)

        rejection = _argument_error(definition.parameters, args)
        if rejection is not None:
            content, error = rejection
            return ToolOutcome(content=content, summary="Rejected", error=error)
        clean_args = {key: value.strip() if isinstance(value, str) else value for key, value in args.items()}

        try:
            return await definition.run(clean_args, context, stage)
        except Exception as exc:  # noqa: BLE001
            message = str(exc) or exc.__class__.__name__
            return ToolOutcome(content=f"The {name} tool failed: {message}", summary="Failed", error=message)

    async def execute_with_progress(
        self, name: str, args: dict[str, Any], context: ContextT, *, enabled: Iterable[str] | None = None
    ) -> AsyncIterator[ToolProgress | ToolOutcome]:
        """Execute a tool, yielding a timeline snapshot per reported stage, then the outcome."""
        timeline = StageTimeline()
        # None marks the end of the tool task.
        progress: asyncio.Queue[ToolProgress | None] = asyncio.Queue()

        def stage(label: str) -> None:
            progress.put_nowait(timeline.begin(label))

        stage("Preparing")
        task = asyncio.create_task(self.execute(name, args, context, enabled=enabled, stage=stage))
        task.add_done_callback(lambda _task: progress.put_nowait(None))
        while (snapshot := await progress.get()) is not None:
            yield snapshot
        outcome = await task
        yield timeline.finish(error=bool(outcome.error))
        yield outcome

    def __contains__(self, name: object) -> bool:
        return name in self._by_name


def _accumulate_tool_calls(pending: dict[int, dict[str, Any]], deltas: list[Any]) -> None:
    """Join provider fragments by call index."""
    for delta in deltas:
        slot = pending.setdefault(getattr(delta, "index", 0) or 0, {"id": "", "name": "", "arguments": ""})
        if getattr(delta, "id", None):
            slot["id"] = delta.id
        function = getattr(delta, "function", None)
        if function is None:
            continue
        if getattr(function, "name", None):
            slot["name"] = function.name
        if getattr(function, "arguments", None):
            slot["arguments"] += function.arguments


def _tool_call_from_text(text: str, catalog: ToolCatalog[Any]) -> ToolCall | None:
    """Parse a text-mode tool call, accepting only names registered in *catalog*."""
    try:
        payload = extract_json_from_llm(text)
    except ValueError:
        return None
    if isinstance(payload.get("function"), dict):
        payload = payload["function"]
    # Models without native tool calling emit `{"name": ..., "arguments": ...}` or variants as text.
    name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
    if not isinstance(name, str) or name not in catalog:
        return None
    args = payload.get("arguments")
    if args is None:
        args = payload.get("parameters") or payload.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return ToolCall(f"text_{name}_{uuid4().hex[:8]}", name, args if isinstance(args, dict) else {})


def _start_stream(
    client: LLMClient,
    model: str,
    messages: list[dict[str, Any]],
    specs: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
) -> Iterator[Any]:
    """Use LiteLLM directly to retain tool deltas and the client's routing."""
    model_id, api_base, custom_provider = client._resolve_routing(model)  # noqa: SLF001
    return litellm.completion(
        model=model_id,
        messages=messages,
        stream=True,
        api_base=api_base,
        custom_llm_provider=custom_provider,
        stream_options={"include_usage": True},
        **({"tools": specs, "tool_choice": "auto"} if specs else {}),
        **kwargs,
    )


@dataclass
class _Round:
    text: str = ""
    call: ToolCall | None = None
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


async def _run_round(
    *,
    client: LLMClient,
    model: str,
    messages: list[dict[str, Any]],
    specs: list[dict[str, Any]] | None,
    catalog: ToolCatalog[Any],
    kwargs: dict[str, Any],
    out: _Round,
) -> AsyncIterator[tuple[str, Any]]:
    """Stream one completion, yielding `token` events and filling *out* with text, call and usage."""
    stream = await asyncio.to_thread(_start_stream, client, model, messages, specs, kwargs)
    pending: dict[int, dict[str, Any]] = {}
    in_reasoning = False
    # Buffer leading text only when tools are offered, to detect text-mode calls before streaming.
    held: str | None = "" if specs else None

    async for chunk in aiter_in_thread(stream):
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            for key in out.usage:
                out.usage[key] += getattr(usage, key, 0) or 0
            continue
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_reasoning:
                in_reasoning = True
                yield ("token", "<thought>\n")
            yield ("token", reasoning)
        if delta.content:
            if in_reasoning:
                in_reasoning = False
                yield ("token", "\n</thought>")
            out.text += delta.content
            if held is None:
                yield ("token", delta.content)
            else:
                held += delta.content
                # Text opening with JSON or a fence may be a text-mode tool call, so keep buffering.
                if not held.lstrip().startswith(("{", "[", "`")):
                    yield ("token", held)
                    held = None
        if getattr(delta, "tool_calls", None):
            _accumulate_tool_calls(pending, delta.tool_calls)

    if in_reasoning:
        yield ("token", "\n</thought>")

    # Only the first named call runs, matching the one-call-per-turn contract.
    native = next((call for call in (pending[index] for index in sorted(pending)) if call["name"]), None)
    if native:
        try:
            args = json.loads(native["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        out.call = ToolCall(native["id"] or f"call_{uuid4().hex[:8]}", native["name"], args if isinstance(args, dict) else {})
    # No native call: the held text may be a JSON tool call emitted as plain content.
    elif held:
        out.call = _tool_call_from_text(held, catalog)
        if out.call is None:
            yield ("token", held)


async def stream_tool_turn[ContextT](
    *,
    client: LLMClient,
    model: str,
    user_msg: str,
    history: list[dict[str, Any]],
    system_prompt: str,
    img: Any = None,
    enabled: list[str],
    catalog: ToolCatalog[ContextT],
    context_for_call: Callable[[str], ContextT],
    guidance: str,
    **kwargs: Any,
) -> AsyncIterator[tuple[str, Any]]:
    """Offer tools once, dispatch at most one call, then force an answer."""
    specs = catalog.specs(enabled)
    prompt = "\n\n".join(filter(None, [system_prompt, guidance])) if specs else system_prompt
    messages = client._construct_message_payload(  # noqa: SLF001
        user_msg=user_msg,
        user_msg_history=history,
        system_prompt=prompt,
        img=img,
    )

    first = _Round()
    async for event in _run_round(
        client=client,
        model=model,
        messages=messages,
        specs=specs or None,
        catalog=catalog,
        kwargs=kwargs,
        out=first,
    ):
        yield event
    usage_total = dict(first.usage)

    if first.call:
        call = first.call
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                ],
            }
        )
        yield ("tool_call", asdict(call))
        async for execution_event in catalog.execute_with_progress(
            call.name, call.arguments, context_for_call(call.id), enabled=enabled
        ):
            if isinstance(execution_event, ToolProgress):
                yield ("tool_progress", {"id": call.id, "stages": execution_event.stages})
                continue
            outcome = execution_event
        yield ("tool_result", outcome.result_event(call))
        definition = catalog.get(call.name)
        if definition is None or definition.answers:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})

            answer = _Round()
            async for event in _run_round(
                client=client,
                model=model,
                messages=messages,
                specs=None,
                catalog=catalog,
                kwargs=kwargs,
                out=answer,
            ):
                yield event
            for key, value in answer.usage.items():
                usage_total[key] += value

    if any(usage_total.values()):
        yield ("usage", usage_total)
