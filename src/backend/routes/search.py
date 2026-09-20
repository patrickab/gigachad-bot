"""Citation-grounded web search backed by Brave LLM Context.

The standalone Web Search mode shares planning, retrieval, source labelling, and
evidence construction with the chat tool through `lib.web_search`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel

from backend.routes.deps import get_model_provider_store, request_client
from lib.llm_resilience import api_query_resilient
from lib.model_provider_store import ModelProviderStore
from lib.web_search import brave_sources, evidence_prompt, plan_query

router = APIRouter(prefix="/api", tags=["search"])


class WebSearchRequest(BaseModel):
    query: str
    system_instructions: str = ""
    model: str = ""


def _plan(raw_query: str, model: str) -> tuple[str, dict[str, int]]:
    with request_client() as client:
        return plan_query(client, raw_query, model)


def _event(event_type: str, **data: object) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


@router.post("/web-search")
async def web_search(
    req: WebSearchRequest,
    store: Annotated[ModelProviderStore, Depends(get_model_provider_store)],
) -> StreamingResponse:
    if not req.model:
        async def missing_model() -> AsyncIterator[str]:
            yield _event("error", data="No chat model selected for web search.")
        return StreamingResponse(missing_model(), media_type="text/event-stream")

    async def event_stream() -> AsyncIterator[str]:
        try:
            planner_model = store.load_defaults()["small_model"]
            search_query, search_profile = await asyncio.to_thread(_plan, req.query, planner_model)
            sources = await brave_sources(search_query, search_profile)
            if not sources:
                yield _event("error", data="Brave found no usable source content for this query.")
                return
            yield _event("sources", sources=sources)
            with request_client() as client:
                chunks = api_query_resilient(
                    client,
                    model=req.model,
                    user_msg=req.query,
                    user_msg_history=[],
                    system_prompt="\n\n".join(filter(None, [req.system_instructions, evidence_prompt(sources)])),
                    stream=True,
                )
                for chunk in chunks:
                    if isinstance(chunk, str) and chunk:
                        yield _event("text", text=chunk)
            yield _event("done")
        except httpx.HTTPStatusError as exc:
            yield _event("error", data=f"Brave search failed ({exc.response.status_code}).")
        except Exception as exc:  # noqa: BLE001 - safely surface search and model failures to the UI
            yield _event("error", data=str(exc))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
