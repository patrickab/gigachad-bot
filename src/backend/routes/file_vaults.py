"""File-vault routes — the API owns every filesystem access.

The UI never touches a vault directly: it lists files (`/files`), previews one
(`/file`), and *attaches* a chosen file (`/attach`). Attach is a **live
reference** — nothing is copied; the returned `path` is stored on the
Attachment and content is read from (and written back to) the actual file.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.routes.deps import get_file_vault
from config import DIRECTORY_OUTPUT_MINERU
from lib import document_library as lib_docs
from lib import extract_queue
from lib.file_vault import FileVault

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/filevaults", tags=["filevaults"])


class VaultFile(BaseModel):
    path: str
    name: str


class VaultListResponse(BaseModel):
    enabled: bool
    files: list[VaultFile]


class VaultNode(BaseModel):
    name: str
    path: str
    type: str
    project: str | None = None
    children: list["VaultNode"] | None = None


class VaultTreeResponse(BaseModel):
    enabled: bool
    tree: list[VaultNode]


class RootBody(BaseModel):
    path: str
    project: str | None = None


class WriteBody(BaseModel):
    path: str
    content: str


class VaultFileContent(BaseModel):
    path: str
    content: str


class AttachResult(BaseModel):
    name: str
    mime: str
    path: str
    content: str | None = None
    parsedMd: str | None = None


@router.get("/files", response_model=VaultListResponse)
async def list_files(vault: FileVault = Depends(get_file_vault)) -> VaultListResponse:
    return VaultListResponse(enabled=vault.enabled, files=vault.list_files())


@router.get("/tree", response_model=VaultTreeResponse)
async def list_tree(vault: FileVault = Depends(get_file_vault)) -> VaultTreeResponse:
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.post("/roots", response_model=VaultTreeResponse)
async def add_root(body: RootBody, vault: FileVault = Depends(get_file_vault)) -> VaultTreeResponse:
    try:
        vault.add_root(body.path, project=body.project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.delete("/roots", response_model=VaultTreeResponse)
async def remove_root(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> VaultTreeResponse:
    vault.remove_root(path)
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


class MountpointBody(BaseModel):
    path: str


@router.post("/mountpoints", response_model=VaultTreeResponse)
async def add_mountpoint(
    body: MountpointBody,
    vault_path: str = Query(..., alias="vault"),
    vault: FileVault = Depends(get_file_vault),
) -> VaultTreeResponse:
    try:
        vault.add_mountpoint(vault_path, body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.delete("/mountpoints", response_model=VaultTreeResponse)
async def remove_mountpoint(
    vault_path: str = Query(..., alias="vault"),
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
) -> VaultTreeResponse:
    vault.remove_mountpoint(vault_path, path)
    return VaultTreeResponse(enabled=vault.enabled, tree=vault.tree())


@router.get("/file", response_model=VaultFileContent)
async def read_file(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> VaultFileContent:
    try:
        return VaultFileContent(path=path, content=vault.read(path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/file")
async def write_file(body: WriteBody, vault: FileVault = Depends(get_file_vault)) -> dict[str, bool]:
    try:
        vault.write(body.path, body.content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/rendered", response_model=VaultFileContent)
async def read_rendered(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> VaultFileContent:
    try:
        raw = vault.read(path)
        return VaultFileContent(path=path, content=vault.resolve_wiki_content(raw, path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/attach", response_model=AttachResult)
async def attach_file(
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
) -> AttachResult:
    """Attach as a live reference: validate the path, return current content.

    Idempotent and copy-free — the frontend calls this again at send time to
    refresh content (and pick up a finished PDF extraction).
    """
    try:
        resolved = vault.resolve(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    mime = lib_docs.mime_for(resolved)
    result = AttachResult(name=resolved.name, mime=mime, path=str(resolved))

    if resolved.suffix.lower() == ".pdf":
        cached_md = DIRECTORY_OUTPUT_MINERU / f"{resolved.stem}.md"
        if cached_md.is_file():
            result.parsedMd = cached_md.read_text(encoding="utf-8")
        else:
            # ponytail: enqueue for background extraction; attach returns without parsedMd
            extract_queue.enqueue(resolved)
    else:
        try:
            result.content = vault.read(path)
        except UnicodeDecodeError:
            result.content = None

    return result
