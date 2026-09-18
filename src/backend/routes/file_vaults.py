"""File-vault routes — the API owns every filesystem access.

The UI never touches a vault directly: it lists files (`/files`), previews one
(`/rendered`), writes one (`/file`), and *attaches* a chosen file (`/attach`).
Attach is a **live reference** — nothing is copied; the returned `path` is
stored on the Attachment and content is read from (and written back to) the
actual file.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.routes.deps import get_asset_store, get_file_vault
from backend.routes.schemas import AttachResult, FileContent, FileListResponse, FileMeta
from lib import document_library as lib_docs
from lib.asset_store import AssetStore
from lib.attachment_materialize import materialize, store_library_pdf
from lib.data_store import StorageNotFoundError
from lib.file_vault import FileVault
from lib.storage_namespace import PDFS

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


@router.get("/files", response_model=VaultListResponse)
async def list_files(vault: FileVault = Depends(get_file_vault)) -> VaultListResponse:
    return VaultListResponse(enabled=vault.enabled, files=vault.list_files())


@router.get("/project-documents", response_model=FileListResponse)
async def project_documents(
    slug: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
    assets: AssetStore = Depends(get_asset_store),
) -> FileListResponse:
    """Files from vaults mounted to *slug*, shaped like project documents.

    Surfaced so the chat sidebar can list mounted-vault files alongside library
    documents. They attach as live references (no copy). A vault PDF whose
    cloud-library copy already exists is reported at the *library* key — the
    parsed cloud copy takes precedence over the vault original, so the sidebar
    never lists the same PDF twice (vault row + library row).
    """
    docs: list[FileMeta] = []
    for f in vault.list_files_for_project(slug):
        p = Path(f["path"])
        if p.suffix.lower() == ".pdf":
            library_key = f"{PDFS}/{p.name}"
            if assets.list(library_key):
                docs.append(FileMeta(**lib_docs.document_meta(library_key)))
                continue
        docs.append(FileMeta(path=str(p), name=p.name, mime=lib_docs.mime_for(p)))
    return FileListResponse(documents=docs)


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


@router.post("/file")
async def write_file(body: FileContent, vault: FileVault = Depends(get_file_vault)) -> dict[str, bool]:
    try:
        vault.write(body.path, body.content)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/rendered", response_model=FileContent)
async def read_rendered(path: str = Query(...), vault: FileVault = Depends(get_file_vault)) -> FileContent:
    try:
        raw = vault.read(path)
        return FileContent(path=path, content=vault.resolve_wiki_content(raw, path))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/attach", response_model=AttachResult)
async def attach_file(
    path: str = Query(...),
    vault: FileVault = Depends(get_file_vault),
    assets: AssetStore = Depends(get_asset_store),
) -> AttachResult:
    """Attach as a live reference: validate the path, return current content.

    Idempotent and copy-free — the frontend calls this again at send time to
    refresh content (and pick up a finished PDF extraction).

    PDFs are the one exception to the copy-free rule: a vault PDF is *promoted*
    into the shared Postgres library (``PDFs/<name>``) on first attach and the
    library copy takes precedence thereafter. This dedupes against any
    same-named PDF already parsed by MinerU, so a vault never shadows the
    canonical library copy. The library's logical key is also accepted here
    (it is what this endpoint itself returns), so the send-time refresh — which
    re-calls this endpoint with whatever path it was last given — keeps working
    for promoted PDFs. Non-PDF vault files stay pure live references.
    """
    mime = lib_docs.mime_for(path)

    if Path(path).suffix.lower() == ".pdf":
        if Path(path).is_absolute():
            try:
                vault_resolved = vault.resolve(path)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            name = vault_resolved.name
            # Cloud copy takes precedence if it already exists (don't clobber a
            # possibly-parsed library PDF with a same-named vault file). Otherwise
            # promote the vault PDF into the library — filename is identity, so
            # this is an overwrite-by-name, never a "<name> (n).pdf" dupe.
            if assets.list(f"{PDFS}/{name}"):
                content = assets.read(f"{PDFS}/{name}").content or b""
            else:
                content = vault_resolved.read_bytes()
                store_library_pdf(assets, name, content)
        else:
            # Send-time refresh: `path` is the logical library key this
            # endpoint returned on first attach.
            name = Path(path).name
            try:
                content = assets.read(f"{PDFS}/{name}").content or b""
            except StorageNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        parsed = materialize(name, content, assets, enqueue_on_miss=True)
        return AttachResult(name=name, mime=mime, path=f"{PDFS}/{name}", parsedMd=parsed)

    try:
        vault_resolved = vault.resolve(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = AttachResult(name=vault_resolved.name, mime=mime, path=str(vault_resolved))
    try:
        result.content = vault.read(path)
    except UnicodeDecodeError:
        result.content = None
    return result
