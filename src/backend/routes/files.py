import json
import logging
import re
import shutil
from pathlib import Path
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from config import DIRECTORY_CHAT_HISTORIES, DIRECTORY_CHAT_UPLOADS

from backend.routes.mineru import should_cancel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


class UploadResult(BaseModel):
    name: str
    mime: str
    content: str | None = None


class ParsedAttachment(BaseModel):
    name: str
    parsedMd: str | None = None


def delete_chat_upload_dir(chat_id: str) -> None:
    path = DIRECTORY_CHAT_UPLOADS / chat_id
    if path.exists():
        shutil.rmtree(path)


def _dedup_name(dest_dir: Path, filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    target = dest_dir / filename
    if not target.exists():
        return filename
    n = 1
    while True:
        candidate = f"{stem} ({n}){suffix}"
        if not (dest_dir / candidate).exists():
            return candidate
        n += 1


def _classify_mime(mime: str) -> str:
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("text/"):
        return "text"
    return "other"


@router.post("/upload", response_model=UploadResult)
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Query(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    chat_dir = DIRECTORY_CHAT_UPLOADS / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)

    deduped = _dedup_name(chat_dir, file.filename)
    dest = chat_dir / deduped
    content_b = await file.read()
    dest.write_bytes(content_b)

    mime = file.content_type or "application/octet-stream"
    kind = _classify_mime(mime)

    result = UploadResult(name=deduped, mime=mime)

    if kind == "text":
        try:
            result.content = content_b.decode("utf-8")
        except UnicodeDecodeError:
            result.content = None

    return result


@router.post("/parse", response_model=list[ParsedAttachment])
async def parse_attachments(
    filenames: list[str] = Query(...),
    chat_id: str = Query(...),
):
    chat_dir = DIRECTORY_CHAT_UPLOADS / chat_id
    if not chat_dir.exists():
        raise HTTPException(status_code=404, detail="Chat upload directory not found")

    results: list[ParsedAttachment] = []
    for filename in filenames:
        if should_cancel():
            log.info("Parse cancelled, stopping batch")
            break

        file_path = chat_dir / filename
        if not file_path.exists():
            results.append(ParsedAttachment(name=filename, parsedMd=None))
            continue

        mime = _classify_mime(filename)
        if file_path.suffix.lower() == ".pdf":
            mime = "pdf"

        if mime == "pdf":
            from backend.routes.mineru import _parse_pdf

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / filename
                tmp_path.write_bytes(file_path.read_bytes())
                try:
                    md_path, _images_dir = await _parse_pdf(tmp_path, chat_dir)
                    results.append(ParsedAttachment(name=filename, parsedMd=md_path.read_text(encoding="utf-8")))
                except Exception:
                    log.exception("MinerU parse failed for %s", filename)
                    results.append(ParsedAttachment(name=filename, parsedMd=None))
        else:
            results.append(ParsedAttachment(name=filename, parsedMd=None))

    return results


@router.delete("/chat/{chat_id}")
async def delete_chat_files(chat_id: str):
    delete_chat_upload_dir(chat_id)
    return {"status": "ok"}


@router.delete("/chat/{chat_id}/att/{filename:path}")
async def delete_single_file(chat_id: str, filename: str):
    path = DIRECTORY_CHAT_UPLOADS / chat_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"status": "ok"}


@router.post("/gc")
async def gc_orphan_uploads():
    known_chat_ids: set[str] = set()
    if DIRECTORY_CHAT_HISTORIES.exists():
        for f in DIRECTORY_CHAT_HISTORIES.rglob("*.json"):
            if "archived" in f.parts:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("chat_id"):
                    known_chat_ids.add(data["chat_id"])
            except Exception:
                continue

    cleaned = 0
    if DIRECTORY_CHAT_UPLOADS.exists():
        for d in DIRECTORY_CHAT_UPLOADS.iterdir():
            if d.is_dir() and d.name not in known_chat_ids:
                shutil.rmtree(d)
                cleaned += 1

    return {"cleaned": cleaned}