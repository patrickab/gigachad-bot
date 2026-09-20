"""Offer, parse, dispatch, and answer with at most one tool call per turn."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
import json
from typing import Any, Generic, TypeVar
from uuid import uuid4

import litellm
from llm_baseclient.client import LLMClient

from lib.llm_json import extract_json_from_llm

__all__ = ["ToolCatalog", "ToolDefinition", "ToolOutcome", "stream_tool_turn"]

ContextT = TypeVar("ContextT")
_SENTINEL = object()


@dataclass
class ToolOutcome:
    """Keep model-visible content separate from browser-only metadata."""

    content: str
    summary: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    sandbox: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolDefinition(Generic[ContextT]):
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any], ContextT], Awaitable[ToolOutcome]]

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


class ToolCatalog(Generic[ContextT]):
    """Offer and execute a closed set of tool definitions."""

    def __init__(self, definitions: Iterable[ToolDefinition[ContextT]]) -> None:
        self.definitions = tuple(definitions)
        self._by_name = {definition.name: definition for definition in self.definitions}

    def specs(self, names: list[str]) -> list[dict[str, Any]]:
        """Drop unknown names so callers cannot widen the offered tool surface."""
        return [self._by_name[name].spec for name in dict.fromkeys(names) if name in self._by_name]

    async def execute(
        self, name: str, args: dict[str, Any], context: ContextT, *, enabled: Iterable[str] | None = None
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
            return await definition.run(clean_args, context)
        except Exception as exc:  # noqa: BLE001
            message = str(exc) or exc.__class__.__name__
            return ToolOutcome(content=f"The {name} tool failed: {message}", summary="Failed", error=message)

    def _contains(self, name: str) -> bool:
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


def _tool_call_from_text(text: str, catalog: ToolCatalog[Any]) -> dict[str, Any] | None:
    try:
        payload = extract_json_from_llm(text)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("function"), dict):
        payload = payload["function"]
    name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
    if not isinstance(name, str) or not catalog._contains(name):
        return None
    args = payload.get("arguments")
    if args is None:
        args = payload.get("parameters") or payload.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {"id": f"text_{name}_{uuid4().hex[:8]}", "name": name, "arguments": args if isinstance(args, dict) else {}}


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
    call: dict[str, Any] | None = None
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
    stream = await asyncio.to_thread(_start_stream, client, model, messages, specs, kwargs)
    iterator = iter(stream)
    pending: dict[int, dict[str, Any]] = {}
    in_reasoning = False
    held: str | None = "" if specs else None

    while True:
        chunk = await asyncio.to_thread(next, iterator, _SENTINEL)
        if chunk is _SENTINEL:
            break
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
                if not held.lstrip().startswith(("{", "[", "`")):
                    yield ("token", held)
                    held = None
        if getattr(delta, "tool_calls", None):
            _accumulate_tool_calls(pending, delta.tool_calls)

    if in_reasoning:
        yield ("token", "\n</thought>")

    native = next((call for call in (pending[index] for index in sorted(pending)) if call["name"]), None)
    if native:
        try:
            args = json.loads(native["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        out.call = {
            "id": native["id"] or f"call_{uuid4().hex[:8]}",
            "name": native["name"],
            "arguments": args if isinstance(args, dict) else {},
        }
    elif held:
        out.call = _tool_call_from_text(held, catalog)
        if out.call is None:
            yield ("token", held)


async def stream_tool_turn(
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
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                    }
                ],
            }
        )
        yield ("tool_call", call)
        outcome = await catalog.execute(call["name"], call["arguments"], context_for_call(call["id"]), enabled=enabled)
        yield (
            "tool_result",
            {
                "id": call["id"],
                "name": call["name"],
                "summary": outcome.summary,
                "sources": [{key: value for key, value in source.items() if key != "content"} for source in outcome.sources],
                "detail": outcome.detail,
                "error": outcome.error,
                "sandbox": outcome.sandbox,
            },
        )
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": outcome.content})

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
