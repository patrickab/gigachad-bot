import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lib.llm_json import extract_json_from_llm
from lib.naming import slugify
from lib.prompts.non_user_prompts import SYS_STUDY_ARTICLE, SYS_STUDY_LEARNING_GOALS, SYS_STUDY_OVERVIEW

from .deps import request_client

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/study", tags=["study"])


class StudyTopic(BaseModel):
    id: str
    label: str
    anchor: str


class StudyProcessRequest(BaseModel):
    markdown: str
    filename: str
    model: str


class StudyProcessResponse(BaseModel):
    filename: str
    topics: list[StudyTopic]
    overview: str
    article: str


def _coerce_topics(raw: object) -> list[StudyTopic]:
    if not isinstance(raw, list):
        return []
    out: list[StudyTopic] = []
    seen_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        anchor = str(item.get("anchor", "")).strip() or label
        if not label:
            continue
        base_id = slugify(str(item.get("id", "") or label), fallback="topic")
        cand = base_id
        n = 2
        while cand in seen_ids:
            cand = f"{base_id}-{n}"
            n += 1
        seen_ids.add(cand)
        out.append(StudyTopic(id=cand, label=label, anchor=anchor))
    return out


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

    async def run_topics() -> list[StudyTopic]:
        content = await _llm_call(req.model, SYS_STUDY_LEARNING_GOALS, user_msg)
        try:
            data = extract_json_from_llm(content)
        except (ValueError, Exception) as e:
            log.warning("learning_goals: JSON extraction failed: %s (raw=%r)", e, content[:300])
            return []
        return _coerce_topics(data.get("topics"))

    async def run_overview() -> str:
        content = await _llm_call(req.model, SYS_STUDY_OVERVIEW, user_msg)
        return content.strip()

    async def run_article() -> str:
        content = await _llm_call(req.model, SYS_STUDY_ARTICLE, user_msg)
        return content.strip()

    try:
        topics, overview, article = await asyncio.gather(
            run_topics(), run_overview(), run_article(),
        )
    except HTTPException:
        raise
    except Exception:
        log.exception("Study artifact generation failed")
        raise HTTPException(status_code=500, detail="Artifact generation failed")

    return StudyProcessResponse(
        filename=req.filename,
        topics=topics,
        overview=overview,
        article=article,
    )
