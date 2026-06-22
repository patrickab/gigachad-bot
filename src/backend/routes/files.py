import json
import logging
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.routes.mineru import should_cancel
from config import DIRECTORY_CHAT_HISTORIES, chat_upload_dir, uploads_dir_for
from lib.image_paths import delete_downscaled
from lib.naming import dedup_filename

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


class UploadResult(BaseModel):
    name: str
    mime: str
    content: str | None = None


class ParsedAttachment(BaseModel):
    name: str
    parsedMd: str | None = None


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


@router.post("/upload", response_model=UploadResult)
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
    slug: str | None = Query(default=None),
):
    chat_dir = chat_upload_dir(chat_id, slug)
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

        mime = _classify_mime("", filename)
        if file_path.suffix.lower() == ".pdf":
            mime = "pdf"

        if mime == "pdf":
            from config import DIRECTORY_OUTPUT_MINERU
            from lib.document_library import organize_file

            cached_md = DIRECTORY_OUTPUT_MINERU / f"{file_path.stem}.md"
            if cached_md.is_file():
                try:
                    organize_file(file_path)
                except Exception:
                    log.exception("Failed to organize %s into document library", filename)
                results.append(ParsedAttachment(name=filename, parsedMd=cached_md.read_text(encoding="utf-8")))
            else:
                from backend.routes.mineru import _parse_pdf

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / filename
                    tmp_path.write_bytes(file_path.read_bytes())
                    try:
                        md_path, _images_dir = await _parse_pdf(tmp_path, chat_dir)
                        try:
                            organize_file(file_path)
                        except Exception:
                            log.exception("Failed to organize %s into document library", filename)
                        results.append(ParsedAttachment(name=filename, parsedMd=md_path.read_text(encoding="utf-8")))
                    except Exception:
                        log.exception("MinerU parse failed for %s", filename)
                        results.append(ParsedAttachment(name=filename, parsedMd=None))
        elif mime == "text":
            try:
                results.append(ParsedAttachment(name=filename, parsedMd=file_path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                results.append(ParsedAttachment(name=filename, parsedMd=None))
        else:
            results.append(ParsedAttachment(name=filename, parsedMd=None))

    return results


@router.delete("/chat/{chat_id}")
async def delete_chat_files(chat_id: str, slug: str | None = Query(default=None)):
    delete_chat_upload_dir(chat_id, slug)
    return {"status": "ok"}


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


@router.post("/gc")
async def gc_orphan_uploads():
    known_chat_ids: set[tuple[str | None, str]] = set()
    if DIRECTORY_CHAT_HISTORIES.exists():
        for f in DIRECTORY_CHAT_HISTORIES.rglob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("chat_id"):
                    rel = f.relative_to(DIRECTORY_CHAT_HISTORIES)
                    slug = rel.parts[0] if len(rel.parts) > 1 else None
                    if slug in ("_uploads", "projects-meta.json"):
                        continue
                    if slug and not (DIRECTORY_CHAT_HISTORIES / slug).is_dir():
                        slug = None
                    known_chat_ids.add((slug, data["chat_id"]))
            except Exception:
                continue

    cleaned = 0
    for uploads_root in [uploads_dir_for(None)] + [
        DIRECTORY_CHAT_HISTORIES / slug / "_uploads"
        for slug_dir in (DIRECTORY_CHAT_HISTORIES.iterdir() if DIRECTORY_CHAT_HISTORIES.exists() else [])
        if slug_dir.is_dir() and (slug_dir / "_uploads").is_dir()
        for slug in [slug_dir.name]
    ]:
        if not uploads_root.exists():
            continue
        slug = None
        rel = uploads_root.relative_to(DIRECTORY_CHAT_HISTORIES)
        if len(rel.parts) > 1 and rel.parts[0] not in ("_uploads",):
            slug = rel.parts[0]
        for d in uploads_root.iterdir():
            if d.is_dir() and (slug, d.name) not in known_chat_ids:
                shutil.rmtree(d)
                cleaned += 1

    return {"cleaned": cleaned}
