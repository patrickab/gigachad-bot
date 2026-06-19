from fastapi import APIRouter
from llm_baseclient.config import (
    AVAILABLE_MODELS,
    MODELS_DEEPSEEK,
    MODELS_GEMINI,
    MODELS_OLLAMA,
    # MODELS_OPENAI
)

from lib.prompts import (
    SYS_ADVISOR,
    SYS_ARTICLE,
    SYS_CONCEPT_IN_DEPTH,
    SYS_CONCEPTUAL_OVERVIEW,
    SYS_EMPTY_PROMPT,
    SYS_QUICK_OVERVIEW,
    SYS_TUTOR,
)

PROMPT_MAP: dict[str, str] = {
    "Quick Overview": SYS_QUICK_OVERVIEW,
    "Advisor": SYS_ADVISOR,
    "Tutor": SYS_TUTOR,
    "Concept - High-Level": SYS_CONCEPTUAL_OVERVIEW,
    "Concept - In-Depth": SYS_CONCEPT_IN_DEPTH,
    "Concept - Article": SYS_ARTICLE,
    "<empty prompt>": SYS_EMPTY_PROMPT,
}

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def get_models() -> dict:
    return {
        "all": AVAILABLE_MODELS,
        "ollama": MODELS_OLLAMA,
        "gemini": MODELS_GEMINI,
        "deepseek": MODELS_DEEPSEEK,
        # "openai": MODELS_OPENAI
    }


@router.get("/prompts")
async def get_prompts() -> dict[str, dict[str, str]]:
    return {"prompts": PROMPT_MAP}
