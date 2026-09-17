import logging
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.routes.deps import get_asset_store
from backend.routes.schemas import AttachResult
from config import DOCUMENTS, chat_upload_dir
from lib.asset_store import AssetStore
from lib.attachment_materialize import materialize
from lib.data_store import InvalidStorageKey, StorageNotFoundError, validate_key
from lib.document_library import mime_for, organize_file
from lib.image_paths import delete_downscaled
from lib.mineru import parse_pdf, should_cancel
from lib.naming import dedup_filename

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


def upload_prefix(chat_id: str, slug: str | None) -> str:
    """Logical asset prefix for one chat's uploads, relative to ``Documents``."""
    return chat_upload_dir(chat_id, slug).relative_to(DOCUMENTS).as_posix()


def _asset_key(prefix: str, filename: str) -> str:
    """Join a client-supplied filename onto a prefix, rejecting traversal at the boundary."""
    try:
        return validate_key(f"{prefix}/{filename}")
    except InvalidStorageKey as exc:
        raise HTTPException(status_code=400, detail="Invalid filename") from exc


def _sidecar_key(key: str) -> str:
    """Parsed-markdown sibling of an attachment: ``a/foo.pdf`` -> ``a/foo.md``, as on disk."""
    path = PurePosixPath(key)
    return str(path.with_name(f"{path.stem}.md"))


def _dedup_asset_name(assets: AssetStore, prefix: str, filename: str) -> str:
    """``dedup_filename`` semantics against stored asset names instead of a directory listing."""
    taken = {PurePosixPath(asset.logical_path).name for asset in assets.list(prefix)}
    if filename not in taken:
        return filename
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 1
    while f"{stem} ({n}){suffix}" in taken:
        n += 1
    return f"{stem} ({n}){suffix}"


def _asset_exists(assets: AssetStore, key: str) -> bool:
    return any(asset.logical_path == key for asset in assets.list(key))


def delete_chat_upload_dir(chat_id: str, slug: str | None = None, assets: AssetStore | None = None) -> None:
    if assets is not None:
        for asset in assets.list(upload_prefix(chat_id, slug)):
            assets.delete(asset.logical_path)
        return
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
    assets: AssetStore | None = Depends(get_asset_store),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    content_b = await file.read()
    mime = file.content_type or "application/octet-stream"

    if assets is None:
        chat_dir = chat_upload_dir(chat_id, slug)
        chat_dir.mkdir(parents=True, exist_ok=True)
        deduped = file.filename if overwrite else dedup_filename(chat_dir, file.filename)
        (chat_dir / deduped).write_bytes(content_b)
    else:
        prefix = upload_prefix(chat_id, slug)
        deduped = file.filename if overwrite else _dedup_asset_name(assets, prefix, file.filename)
        assets.write("upload", _asset_key(prefix, deduped), content_b, mime=mime)

    kind = _classify_mime(mime, deduped)

    result = AttachResult(name=deduped, mime=mime)

    if kind == "text":
        try:
            result.content = content_b.decode("utf-8")
        except UnicodeDecodeError:
            result.content = None

    return result


def _promote_pdf(assets: AssetStore, pdf_path: Path) -> None:
    """Postgres counterpart of ``organize_file``: the ``pdf`` asset row is authoritative,
    the ``PDFs/`` copy is only a mirror, so a mirror failure never undoes the write."""
    try:
        asset = assets.write("pdf", f"PDFs/{pdf_path.name}", pdf_path.read_bytes(), mime="application/pdf")
    except Exception:
        log.exception("Failed to store %s in the PDF library", pdf_path.name)
        return
    try:
        assets.mirror(asset, DOCUMENTS)
    except OSError:
        log.exception("Failed to mirror PDF asset %s", asset.logical_path)


async def _parse_assets(assets: AssetStore, filenames: list[str], chat_id: str, slug: str | None) -> list[AttachResult]:
    """Postgres branch of ``/parse``: bytes come from the store, MinerU works in a temp dir."""
    prefix = upload_prefix(chat_id, slug)
    results: list[AttachResult] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        for filename in filenames:
            if should_cancel():
                log.info("Parse cancelled, stopping batch")
                break

            file_mime = mime_for(filename)
            key = _asset_key(prefix, filename)
            try:
                asset = assets.read(key)
            except StorageNotFoundError:
                results.append(AttachResult(name=filename, mime=file_mime))
                continue

            kind = "pdf" if Path(filename).suffix.lower() == ".pdf" else _classify_mime("", filename)
            if kind == "pdf":
                pdf_path = work / Path(filename).name
                pdf_path.write_bytes(asset.content or b"")
                cached = materialize(pdf_path, enqueue_on_miss=False)
                if cached.parsed_md is not None:
                    _promote_pdf(assets, pdf_path)
                    results.append(AttachResult(name=filename, mime=file_mime, parsedMd=cached.parsed_md))
                    continue
                try:
                    md_path, _images_dir = await parse_pdf(pdf_path, work / "mineru")
                except Exception:
                    log.exception("MinerU parse failed for %s", filename)
                    results.append(AttachResult(name=filename, mime=file_mime))
                    continue
                parsed_md = md_path.read_text(encoding="utf-8")
                assets.write("upload", _sidecar_key(key), parsed_md.encode("utf-8"), mime="text/markdown")
                _promote_pdf(assets, pdf_path)
                results.append(AttachResult(name=filename, mime=file_mime, parsedMd=parsed_md))
            elif kind == "text":
                try:
                    parsed_md = (asset.content or b"").decode("utf-8")
                except UnicodeDecodeError:
                    results.append(AttachResult(name=filename, mime=file_mime))
                else:
                    results.append(AttachResult(name=filename, mime=file_mime, parsedMd=parsed_md))
            else:
                results.append(AttachResult(name=filename, mime=file_mime))
    return results


@router.post("/parse", response_model=list[AttachResult])
async def parse_attachments(
    filenames: list[str] = Query(...),
    chat_id: str = Query(...),
    slug: str | None = Query(default=None),
    assets: AssetStore | None = Depends(get_asset_store),
):
    if assets is not None:
        return await _parse_assets(assets, filenames, chat_id, slug)

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
async def delete_single_file(
    chat_id: str,
    filename: str,
    slug: str | None = Query(default=None),
    assets: AssetStore | None = Depends(get_asset_store),
):
    if assets is not None:
        key = _asset_key(upload_prefix(chat_id, slug), filename)
        if not _asset_exists(assets, key):
            raise HTTPException(status_code=404, detail="File not found")
        assets.delete(key)
        assets.delete(_sidecar_key(key))
        return {"status": "ok"}

    path = chat_upload_dir(chat_id, slug) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    delete_downscaled(chat_upload_dir(chat_id, slug), filename)
    md_path = path.with_name(path.stem + ".md")
    if md_path.exists():
        md_path.unlink()
    return {"status": "ok"}
