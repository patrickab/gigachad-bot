"""The gateway client: request shape, SDK passthrough of gateway extras, `<thought>` wrapping, errors and images."""

import base64
from dataclasses import dataclass, field
import io
import json
from typing import Any

import httpx
import openai
from PIL import Image
import pytest

from lib import completions, pi_host

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _sse(*chunks: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + b"data: [DONE]\n\n"


def _chunk(**delta: Any) -> dict[str, Any]:
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": delta}]}


def _usage(total: int) -> dict[str, Any]:
    usage = {"prompt_tokens": total, "completion_tokens": 0, "total_tokens": total}
    return {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [], "usage": usage}


@dataclass
class Gateway:
    """Scripted gateway replies, and the request bodies the SDK sent."""

    replies: list[httpx.Response] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer gateway-token"
        self.requests.append(json.loads(request.content))
        return self.replies.pop(0)


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Gateway:
    """Serve scripted gateway replies through the real SDK."""
    scripted = Gateway()

    def client(base_url: str, api_key: str) -> openai.OpenAI:
        transport = httpx.MockTransport(scripted.handle)
        return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, http_client=httpx.Client(transport=transport))

    monkeypatch.setattr(pi_host, "token", lambda: "gateway-token")
    monkeypatch.setattr(completions, "_client", client)
    return scripted


def test_stream_sends_the_gateway_id_tools_and_usage_option_and_keeps_thinking_blocks(gateway: Gateway) -> None:
    """Signed thinking blocks and reasoning text are gateway extras the SDK must hand through untouched."""
    block = {"type": "thinking", "thinking": "hm", "signature": "sig"}
    gateway.replies.append(
        httpx.Response(200, content=_sse(_chunk(reasoning_content="hm", thinking_blocks=[block]), _chunk(content="hi"), _usage(3)))
    )
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

    messages = [{"role": "user", "content": "q"}]
    chunks = list(completions.stream("gemini/gemini-3.5-flash", messages, tools=tools, reasoning_effort="low"))

    (body,) = gateway.requests
    assert body["model"] == "google/gemini-3.5-flash"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert (body["tools"], body["tool_choice"], body["reasoning_effort"]) == (tools, "auto", "low")
    assert chunks[0].choices[0].delta.reasoning_content == "hm"
    assert chunks[0].choices[0].delta.thinking_blocks == [block]
    assert chunks[-1].usage.total_tokens == 3


def test_text_returns_the_answer_and_names_an_unserved_model(gateway: Gateway) -> None:
    gateway.replies.append(
        httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}
        ]})
    )
    gateway.replies.append(
        httpx.Response(404, json={"error": {"message": "unknown model nope/foo", "type": "gateway_error", "code": 404}})
    )

    assert completions.text("litellm_proxy/anthropic/claude-haiku-5-5", []) == "{}"
    unserved = r"Model nope/foo is not served by pi-host's gateway \(unknown model nope/foo\)"
    with pytest.raises(completions.ModelNotServed, match=unserved):
        completions.text("nope/foo", [])
    assert [body["model"] for body in gateway.requests] == ["anthropic/claude-haiku-5-5", "nope/foo"]


def test_a_mid_stream_gateway_error_raises(gateway: Gateway) -> None:
    failure = {"error": {"message": "overloaded", "type": "error"}}
    gateway.replies.append(httpx.Response(200, content=_sse(_chunk(content="a"), failure)))

    with pytest.raises(openai.APIError, match="overloaded"):
        list(completions.stream_text("anthropic/claude-haiku-5-5", []))


def test_stream_text_wraps_reasoning_and_ends_with_usage(gateway: Gateway) -> None:
    gateway.replies.append(
        httpx.Response(200, content=_sse(_chunk(reasoning_content="think"), _chunk(content="answer"), _usage(5)))
    )

    assert list(completions.stream_text("anthropic/claude-haiku-5-5", [])) == [
        "<thought>\n",
        "think",
        "\n</thought>",
        "answer",
        {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    ]


def test_build_messages_puts_images_after_the_text_as_data_uris() -> None:
    assert completions.build_messages("hi", [{"role": "assistant", "content": "a"}], "sys") == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "hi"},
    ]

    (user,) = completions.build_messages("read", img=[PNG, "data:image/jpeg;base64,AAAA"])

    assert user["content"] == [
        {"type": "text", "text": "read"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(PNG).decode()}"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]


def test_downscale_img_fits_the_token_budget_on_the_patch_grid() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (2000, 1000)).save(buffer, format="PNG")

    uri = completions.downscale_img(buffer.getvalue(), max_tokens=256)

    header, _, data = uri.partition(",")
    assert header == "data:image/jpeg;base64"
    width, height = Image.open(io.BytesIO(base64.b64decode(data))).size
    assert width % 28 == 0 and height % 28 == 0
    assert completions._patch_tokens(width, height) <= 256  # noqa: SLF001
