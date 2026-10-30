"""Request-owned settings and dependencies handed to one built-in tool call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_sandbox import PromptImage
    from llm_baseclient.client import LLMClient

    from lib.project_store import ProjectStore
    from lib.sandbox_service import SandboxService


class ToolOptions(BaseModel):
    """Hold the per-request settings the browser already owns."""

    search_system_instructions: str = ""
    search_domain: str = ""
    research_fast_model: str = ""
    research_smart_model: str = ""
    research_strategic_model: str = ""
    research_depth: int = Field(default=2, ge=1, le=5)
    research_breadth: int = Field(default=4, ge=1, le=10)
    research_reasoning: str | None = None
    research_report_type: str = "deep"


@dataclass(frozen=True)
class ToolContext:
    """Provide one tool call with request-owned dependencies."""

    client: LLMClient
    model: str
    opts: ToolOptions
    chat_id: str = ""
    sandbox_service: SandboxService | None = None
    tool_call_id: str = ""
    prompt_images: tuple[PromptImage, ...] = ()
    small_model: str = ""
    history: tuple[dict[str, Any], ...] = ()
    user_msg: str = ""
    img: Any = None
    vision_model: str = ""
    # The chat's project (None outside projects) and the store its learn sessions are grouped in.
    project_slug: str | None = None
    project_store: ProjectStore | None = None
