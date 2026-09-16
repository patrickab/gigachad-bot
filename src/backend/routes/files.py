import logging
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.routes.schemas import AttachResult
from config import chat_upload_dir
from lib.attachment_materialize import materialize
from lib.document_library import mime_for, organize_file
from lib.image_paths import delete_downscaled
from lib.mineru import parse_pdf, should_cancel
from lib.naming import dedup_filename

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


def delete_chat_upload_dir(chat_id: str, slug: str | None = None) -> None:
    path = chat_upload_dir(chat_id, slug)
    if path.exists():
        shutil.rmtree(path)


_TEXT_EXTS = {".csv", ".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf", ".log", ".md", ".rst", ".svg"}


def _classify_mime(mime: str, filename: str | None = None) -> str:
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("text/"):
        return "text"
    if filename and Path(filename).suffix.lower() in _TEXT_EXTS:
        return "text"
    return "other"


@router.post("/upload", response_model=AttachResult)
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    overwrite: bool = Query(default=False),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    chat_dir = chat_upload_dir(chat_id, slug)
    chat_dir.mkdir(parents=True, exist_ok=True)

    deduped = file.filename if overwrite else dedup_filename(chat_dir, file.filename)
    dest = chat_dir / deduped
    content_b = await file.read()
    dest.write_bytes(content_b)

    mime = file.content_type or "application/octet-stream"
    kind = _classify_mime(mime, deduped)

    result = AttachResult(name=deduped, mime=mime)

    if kind == "text":
        try:
            result.content = content_b.decode("utf-8")
        except UnicodeDecodeError:
            result.content = None

    return result


@router.post("/parse", response_model=list[AttachResult])
async def parse_attachments(
    filenames: list[str] = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
):
    chat_dir = chat_upload_dir(chat_id, slug)
    if not chat_dir.exists():
        raise HTTPException(status_code=404, detail="Chat upload directory not found")

    results: list[AttachResult] = []
    for filename in filenames:
        if should_cancel():
            log.info("Parse cancelled, stopping batch")
            break

        file_mime = mime_for(filename)
        file_path = chat_dir / filename
        if not file_path.exists():
            results.append(AttachResult(name=filename, mime=file_mime))
            continue

        kind = _classify_mime("", filename)
        if file_path.suffix.lower() == ".pdf":
            kind = "pdf"

        if kind == "pdf":
            cached = materialize(file_path, enqueue_on_miss=False)
            if cached.parsed_md is not None:
                try:
                    organize_file(file_path)
                except Exception:
                    log.exception("Failed to organize %s into document library", filename)
                results.append(AttachResult(name=filename, mime=file_mime, parsedMd=cached.parsed_md))
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / filename
                    tmp_path.write_bytes(file_path.read_bytes())
                    try:
                        md_path, _images_dir = await parse_pdf(tmp_path, chat_dir)
                        try:
                            organize_file(file_path)
                        except Exception:
                            log.exception("Failed to organize %s into document library", filename)
                        results.append(
                            AttachResult(name=filename, mime=file_mime, parsedMd=md_path.read_text(encoding="utf-8"))
                        )
                    except Exception:
                        log.exception("MinerU parse failed for %s", filename)
                        results.append(AttachResult(name=filename, mime=file_mime))
        elif kind == "text":
            try:
                results.append(AttachResult(name=filename, mime=file_mime, parsedMd=file_path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                results.append(AttachResult(name=filename, mime=file_mime))
        else:
            results.append(AttachResult(name=filename, mime=file_mime))

    return results


@router.delete("/chat/{chat_id}/att/{filename:path}")
async def delete_single_file(chat_id: str, filename: str, slug: str | None = Query(default=None)):
    path = chat_upload_dir(chat_id, slug) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    delete_downscaled(chat_upload_dir(chat_id, slug), filename)
    md_path = path.with_name(path.stem + ".md")
    if md_path.exists():
        md_path.unlink()
    return {"status": "ok"}
