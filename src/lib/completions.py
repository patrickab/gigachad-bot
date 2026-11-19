"""Chat completions through pi-host's OpenAI-compatible gateway, the one path every backend model call takes.

- Consumes `pi_host.gateway_url()` with bearer `pi_host.token()`, a stored model selector becomes the gateway's id
  through `pi_models.gateway_model_id`
- Provides raw chunk streaming (tools, `thinking_blocks`), a text stream with `<thought>` wrapping and a final usage
  dict, a non-streaming text call, and the message and image helpers the callers build requests with
- A model the gateway does not serve raises `ModelNotServed` naming the stored selector
"""

from __future__ import annotations

import base64
from functools import lru_cache
import io
import math
from typing import TYPE_CHECKING, Any

import filetype
from PIL import Image, ImageOps

from lib import pi_host
from lib.pi_models import GatewayUnavailable, gateway_model_id

if TYPE_CHECKING:
    from collections.abc import Iterator

    import openai
    from openai.types.chat import ChatCompletionChunk

type Usage = dict[str, int]
# Raw bytes, a `data:image/...` URI or an http(s) URL.
type ImageInput = bytes | str

# The vision patch grid `downscale_img` budgets tokens on: one token per 28 px patch plus one per patch row.
_PATCH_PIXELS = 28
_JPEG_QUALITY = 85


class ModelNotServed(RuntimeError):
    """The gateway serves no model under the selector's gateway id."""


@lru_cache(maxsize=1)
def _client(base_url: str, api_key: str) -> openai.OpenAI:
    """One SDK client per gateway URL and token, rebuilt when pi-host writes a new token."""
    import openai  # ~0.5 s, so it loads on the first model call, not when the server starts

    return openai.OpenAI(base_url=base_url, api_key=api_key)


def _create(selector: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """`chat.completions.create` for a stored selector, gateway failures translated to errors naming it."""
    import openai

    model = gateway_model_id(selector)
    url = pi_host.gateway_url()
    try:
        return _client(url, pi_host.token()).chat.completions.create(model=model, messages=messages, **kwargs)
    except openai.NotFoundError as exc:
        detail = exc.body.get("message") if isinstance(exc.body, dict) else None
        raise ModelNotServed(f"Model {selector} is not served by pi-host's gateway ({detail or exc.message}).") from exc
    except openai.APIConnectionError as exc:
        raise GatewayUnavailable(f"pi-host's gateway at {url} does not answer ({exc}).") from exc


def stream(
    model: str, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, **kwargs: Any
) -> Iterator[ChatCompletionChunk]:
    """Stream a completion's raw chunks, usage last. *tools* are offered with `tool_choice` auto."""
    offered = {"tools": tools, "tool_choice": "auto"} if tools else {}
    return _create(model, messages, stream=True, stream_options={"include_usage": True}, **offered, **kwargs)


def stream_text(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str | Usage]:
    """Stream answer text with reasoning wrapped in `<thought>` tags, then one usage dict."""
    in_reasoning = False
    for chunk in stream(model, messages, **kwargs):
        if chunk.usage is not None:
            yield {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
            continue
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if reasoning := getattr(delta, "reasoning_content", None):
            if not in_reasoning:
                in_reasoning = True
                yield "<thought>\n"
            yield reasoning
        if delta.content:
            if in_reasoning:
                in_reasoning = False
                yield "\n</thought>"
            yield delta.content
    if in_reasoning:
        yield "\n</thought>"


def text(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """One non-streaming completion's answer text."""
    return _create(model, messages, **kwargs).choices[0].message.content or ""


def _image_url(img: ImageInput) -> str:
    """A data URI for raw bytes, an image URL or data URI as given."""
    if isinstance(img, str):
        return img
    kind = filetype.guess(img)
    return f"data:{kind.mime if kind else 'image/jpeg'};base64,{base64.b64encode(img).decode()}"


def build_messages(
    user_msg: str | None = None,
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    img: ImageInput | list[ImageInput] | tuple[ImageInput, ...] | None = None,
) -> list[dict[str, Any]]:
    """The system prompt, *history*, then the user turn carrying *img* as image parts."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or [])
    images = [] if img is None else [_image_url(i) for i in (img if isinstance(img, (list, tuple)) else [img])]
    if images:
        text_part = [{"type": "text", "text": user_msg}] if user_msg else []
        messages.append({"role": "user", "content": text_part + [{"type": "image_url", "image_url": {"url": url}} for url in images]})
    elif user_msg:
        messages.append({"role": "user", "content": user_msg})
    return messages


def _patch_tokens(width: int, height: int) -> int:
    columns, rows = math.ceil(width / _PATCH_PIXELS), math.ceil(height / _PATCH_PIXELS)
    return columns * rows + rows


def downscale_img(img: bytes | str, max_tokens: int) -> str:
    """Shrink an image (bytes or data URI) to *max_tokens* vision patches, aspect kept, as a JPEG data URI."""
    raw = base64.b64decode(img.partition(",")[2]) if isinstance(img, str) else img
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))  # re-encoding drops the EXIF orientation tag
    width, height = image.size
    scale = min(1.0, math.sqrt(max_tokens / _patch_tokens(width, height)))
    fit_width = max(_PATCH_PIXELS, round(width * scale / _PATCH_PIXELS) * _PATCH_PIXELS)
    fit_height = max(_PATCH_PIXELS, round(height * scale / _PATCH_PIXELS) * _PATCH_PIXELS)
    while _patch_tokens(fit_width, fit_height) > max_tokens and (fit_width > _PATCH_PIXELS or fit_height > _PATCH_PIXELS):
        if fit_width > fit_height:
            fit_width -= _PATCH_PIXELS
        else:
            fit_height -= _PATCH_PIXELS
    if (fit_width, fit_height) != (width, height):
        image = image.resize((fit_width, fit_height), Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
