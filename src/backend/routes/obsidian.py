"""Obsidian vault routes — the API owns every filesystem access.

The UI never touches the vault directly: it lists notes (`/files`), previews a
note (`/file`), and materialises a chosen note into the chat's upload directory
(`/attach`) so it behaves exactly like any other text attachment.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.routes.deps import get_obsidian_vault
from config import chat_upload_dir
from lib.naming import dedup_filename
from lib.obsidian_vault import ObsidianVault

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/obsidian", tags=["obsidian"])


class ObsidianFile(BaseModel):
    path: str
    name: str


class ObsidianListResponse(BaseModel):
    enabled: bool
    files: list[ObsidianFile]


class ObsidianNode(BaseModel):
    name: str
    path: str
    type: str
    children: list["ObsidianNode"] | None = None


class ObsidianTreeResponse(BaseModel):
    enabled: bool
    tree: list[ObsidianNode]


class RootBody(BaseModel):
    path: str


class WriteBody(BaseModel):
    path: str
    content: str


class ObsidianFileContent(BaseModel):
    path: str
    content: str


class AttachResult(BaseModel):
    name: str
    mime: str
    content: str


@router.get("/files", response_model=ObsidianListResponse)
async def list_files(vault: ObsidianVault = Depends(get_obsidian_vault)) -> ObsidianListResponse:
    return ObsidianListResponse(enabled=vault.enabled, files=vault.list_markdown())


@router.get("/tree", response_model=ObsidianTreeResponse)
async def list_tree(vault: ObsidianVault = Depends(get_obsidian_vault)) -> ObsidianTreeResponse:
    return ObsidianTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.post("/roots", response_model=ObsidianTreeResponse)
async def add_root(body: RootBody, vault: ObsidianVault = Depends(get_obsidian_vault)) -> ObsidianTreeResponse:
    try:
        vault.add_root(body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ObsidianTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.delete("/roots", response_model=ObsidianTreeResponse)
async def remove_root(path: str = Query(...), vault: ObsidianVault = Depends(get_obsidian_vault)) -> ObsidianTreeResponse:
    vault.remove_root(path)
    return ObsidianTreeResponse(enabled=vault.enabled, tree=vault.tree())


class MountpointBody(BaseModel):
    path: str


@router.post("/mountpoints", response_model=ObsidianTreeResponse)
async def add_mountpoint(
    body: MountpointBody,
    vault_path: str = Query(..., alias="vault"),
    vault: ObsidianVault = Depends(get_obsidian_vault),
) -> ObsidianTreeResponse:
    try:
        vault.add_mountpoint(vault_path, body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ObsidianTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.delete("/mountpoints", response_model=ObsidianTreeResponse)
async def remove_mountpoint(
    vault_path: str = Query(..., alias="vault"),
    path: str = Query(...),
    vault: ObsidianVault = Depends(get_obsidian_vault),
) -> ObsidianTreeResponse:
    vault.remove_mountpoint(vault_path, path)
    return ObsidianTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.get("/file", response_model=ObsidianFileContent)
async def read_file(path: str = Query(...), vault: ObsidianVault = Depends(get_obsidian_vault)) -> ObsidianFileContent:
    try:
        return ObsidianFileContent(path=path, content=vault.read(path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/file")
async def write_file(body: WriteBody, vault: ObsidianVault = Depends(get_obsidian_vault)) -> dict[str, bool]:
    try:
        vault.write(body.path, body.content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/rendered", response_model=ObsidianFileContent)
async def read_rendered(path: str = Query(...), vault: ObsidianVault = Depends(get_obsidian_vault)) -> ObsidianFileContent:
    try:
        raw = vault.read(path)
        return ObsidianFileContent(path=path, content=vault.resolve_wiki_content(raw, path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/attach", response_model=AttachResult)
async def attach_file(
    path: str = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    vault: ObsidianVault = Depends(get_obsidian_vault),
) -> AttachResult:
    try:
        content = vault.read(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    chat_dir = chat_upload_dir(chat_id, slug)
    chat_dir.mkdir(parents=True, exist_ok=True)
    name = dedup_filename(chat_dir, Path(path).name)
    (chat_dir / name).write_text(content, encoding="utf-8")

    return AttachResult(name=name, mime="text/markdown", content=content)
