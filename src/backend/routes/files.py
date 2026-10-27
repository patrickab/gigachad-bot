from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.http_plumbing import http_errors
from backend.routes.deps import get_asset_store
from lib.asset_store import AssetStore
from lib.attachments import AttachResult, ChatUploads
from lib.data_store import InvalidStorageKey

router = APIRouter(prefix="/api/files", tags=["files"])

# A filename that would leave the chat's upload prefix.
_INVALID_FILENAME = (InvalidStorageKey, 400, "Invalid filename")


@router.post("/upload", response_model=AttachResult)
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    overwrite: bool = Query(default=False),
    assets: AssetStore = Depends(get_asset_store),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    content = await file.read()
    uploads = ChatUploads(assets, chat_id, slug)
    with http_errors(_INVALID_FILENAME):
        return uploads.store(file.filename, content, file.content_type or "application/octet-stream", overwrite=overwrite)


@router.post("/parse", response_model=list[AttachResult])
async def parse_attachments(
    filenames: list[str] = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    assets: AssetStore = Depends(get_asset_store),
):
    with http_errors(_INVALID_FILENAME):
        return await ChatUploads(assets, chat_id, slug).parse(filenames)


@router.delete("/chat/{chat_id}/att/{filename:path}")
async def delete_single_file(
    chat_id: str,
    filename: str,
    slug: str | None = Query(default=None),
    assets: AssetStore = Depends(get_asset_store),
):
    uploads = ChatUploads(assets, chat_id, slug)
    with http_errors(_INVALID_FILENAME):
        deleted = uploads.delete(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "ok"}
