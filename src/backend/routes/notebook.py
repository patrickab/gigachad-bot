"""Notebook routes — the HTTP face of the chat-scoped notebook.

The notebook source is stored in the percent format the frontend speaks
(`# %%` code cells, `# %% [markdown]` cells); these routes treat it as an
opaque string and only arbitrate revisions. PUT is optimistic-concurrency
gated (`base_revision` must match the current revision), and POST /run
delegates execution to the sandbox-notebook harness.
"""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.routes.deps import get_chat_store, get_sandbox_service
from lib.chat_store import ChatStore
from lib.sandbox_service import NOTEBOOK_SCOPE, NotebookRevisionConflict, SandboxService
from lib.sandbox_notebook import run_notebook_cells

router = APIRouter(prefix="/api/notebook", tags=["notebook"])

SandboxServiceDep = Annotated[SandboxService, Depends(get_sandbox_service)]

ChatStoreDep = Annotated[ChatStore, Depends(get_chat_store)]


async def _checkpoint_saved_notebook(chat_id: str, chats: ChatStore, sandbox: SandboxService) -> None:
    if chats.find_by_chat_id(chat_id) is not None:
        await sandbox.checkpoint(chat_id=chat_id, scope=NOTEBOOK_SCOPE)


class NotebookResponse(BaseModel):
    source: str
    outputs: dict
    revision_id: str


class PutNotebookRequest(BaseModel):
    source: str
    outputs: dict = {}
    base_revision: str



class RunNotebookRequest(BaseModel):
    upto: int = Field(ge=1)


class RevisionResponse(BaseModel):
    revision_id: str


class RunResponse(BaseModel):
    revision_id: str
    outputs: dict



@router.get("/{chat_id}")
async def get_notebook(chat_id: str, sandbox: SandboxServiceDep) -> NotebookResponse:
    notebook = await sandbox.read_notebook(chat_id)
    if notebook is None:
        raise HTTPException(status_code=404, detail="No notebook for this chat")
    return NotebookResponse(source=notebook["source"], outputs=notebook["outputs"], revision_id=notebook["revision_id"])


@router.put("/{chat_id}")
async def put_notebook(
    chat_id: str, req: PutNotebookRequest, sandbox: SandboxServiceDep, chats: ChatStoreDep
) -> RevisionResponse:
    revision_id = uuid4().hex
    try:
        await sandbox.stage_notebook(chat_id, revision_id, req.source, req.outputs, expected_revision=req.base_revision)
    except NotebookRevisionConflict:
        raise HTTPException(status_code=409, detail="Stale notebook revision") from None
    await _checkpoint_saved_notebook(chat_id, chats, sandbox)
    return RevisionResponse(revision_id=revision_id)


@router.post("/{chat_id}/run")
async def run_notebook(chat_id: str, req: RunNotebookRequest, sandbox: SandboxServiceDep, chats: ChatStoreDep) -> RunResponse:
    try:
        result = await run_notebook_cells(sandbox, chat_id, req.upto)
    except LookupError:
        raise HTTPException(status_code=404, detail="No notebook for this chat") from None
    except NotebookRevisionConflict:
        raise HTTPException(status_code=409, detail="Stale notebook revision") from None
    except ValueError as exc:
        # An out-of-range `upto` is a routine client race, not a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from None
    await _checkpoint_saved_notebook(chat_id, chats, sandbox)
    return RunResponse(revision_id=result["revision_id"], outputs=result["outputs"])