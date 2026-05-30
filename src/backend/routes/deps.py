import base64
from contextlib import contextmanager
import re
from typing import Any, Iterator

from sse_starlette.sse import EventSourceResponse

from llm_client import LLMClient
from llm_config import MODEL_CONFIGS

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(model_configs=MODEL_CONFIGS)
    return _client


def shutdown_client() -> None:
    global _client
    if _client is not None:
        _client.kill_inference_engines()
        _client = None


@contextmanager
def request_client() -> Iterator[LLMClient]:
    c = get_client()
    c.reset_history()
    try:
        yield c
    finally:
        c.reset_history()


def decode_image(base64_data: str | None) -> bytes | None:
    if not base64_data:
        return None
    match = re.match(r"data:image/\w+;base64,(.+)", base64_data)
    if match:
        return base64.b64decode(match.group(1))
    return base64.b64decode(base64_data)


def sse_event_stream(chunks: Iterator[str]) -> EventSourceResponse:
    async def event_stream() -> Any:
        try:
            for chunk in chunks:
                yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_stream())