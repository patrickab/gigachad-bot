import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lib.prompts.non_user_prompts import SYS_STUDY_ARTICLE, SYS_STUDY_MINDMAP, SYS_STUDY_OVERVIEW

from .deps import request_client

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/study", tags=["study"])


class StudyProcessRequest(BaseModel):
    markdown: str
    filename: str
    model: str


class StudyProcessResponse(BaseModel):
    filename: str
    mindmap: str
    overview: str
    article: str


def _build_user_msg(markdown: str) -> str:
    return (
        "Below is the raw markdown extracted from a PDF. Produce the requested artifact as specified in the system prompt.\n\n"
        "<raw_markdown>\n" + markdown + "\n</raw_markdown>"
    )


def _sync_llm_call(model: str, system_prompt: str, user_msg: str) -> str:
    with request_client() as c:
        response = c.api_query(
            model=model,
            user_msg=user_msg,
            user_msg_history=[],
            system_prompt=system_prompt,
            img=None,
            stream=False,
        )
        if isinstance(response, Exception):
            raise response
        return response.choices[0].message.content or ""


async def _llm_call(model: str, system_prompt: str, user_msg: str) -> str:
    try:
        return await asyncio.to_thread(_sync_llm_call, model, system_prompt, user_msg)
    except Exception:
        log.exception("LLM call failed (model=%s, sys_prompt_len=%d)", model, len(system_prompt))
        raise HTTPException(status_code=500, detail="LLM call failed")


@router.post("/process", response_model=StudyProcessResponse)
async def process_pdf(req: StudyProcessRequest) -> StudyProcessResponse:
    """Accept markdown extracted from a PDF, run three parallel LLM calls (learning goals, overview, article)."""
    user_msg = _build_user_msg(req.markdown)

    async def run_mindmap() -> str:
        content = await _llm_call(req.model, SYS_STUDY_MINDMAP, user_msg)
        return content.strip()

    async def run_overview() -> str:
        content = await _llm_call(req.model, SYS_STUDY_OVERVIEW, user_msg)
        return content.strip()

    async def run_article() -> str:
        content = await _llm_call(req.model, SYS_STUDY_ARTICLE, user_msg)
        return content.strip()

    try:
        mindmap, overview, article = await asyncio.gather(
            run_mindmap(), run_overview(), run_article(),
        )
    except HTTPException:
        raise
    except Exception:
        log.exception("Study artifact generation failed")
        raise HTTPException(status_code=500, detail="Artifact generation failed")

    return StudyProcessResponse(
        filename=req.filename,
        mindmap=mindmap,
        overview=overview,
        article=article,
    )


class MindmapRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str
    prompt: str = ""


class MindmapResponse(BaseModel):
    mindmap: str


@router.post("/mindmap", response_model=MindmapResponse)
async def generate_mindmap(req: MindmapRequest) -> MindmapResponse:
    transcript = "\n".join(f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in req.messages)
    user_msg = (
        "Below is a conversation transcript. Produce the requested mind map as specified in the system prompt.\n\n"
        "<transcript>\n" + transcript + "\n</transcript>"
    )
    sys_prompt = SYS_STUDY_MINDMAP
    if req.prompt:
        sys_prompt += f"\n\n# Additional instructions from the user\n{req.prompt}"

    content = await _llm_call(req.model, sys_prompt, user_msg)
    return MindmapResponse(mindmap=content.strip())
