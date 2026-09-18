"""The MinerU PDF extraction engine.

Owns the parse call itself plus the lifecycle of the MinerU server processes it
spawns: a registry of live servers so shutdown can stop them, and a cancel flag
the batch parsers poll between files.

This is engine code, not route code. It lives in ``lib`` so the background
extraction queue and the ``files`` route can both drive it without importing
upward from the route layer.
"""

import logging
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading

from fastapi import HTTPException

from config import DIRECTORY_OUTPUT_MINERU, MINERU_SERVER_URL

log = logging.getLogger(__name__)

_active_servers: list[object] = []
_active_servers_lock = threading.Lock()
_cancel_event = threading.Event()


def kill_all_mineru_servers() -> None:
    _cancel_event.set()
    with _active_servers_lock:
        servers = list(_active_servers)
        _active_servers.clear()
    for srv in servers:
        try:
            srv.stop()
        except Exception:
            log.exception("Error stopping MinerU server during shutdown")


def reset_cancel() -> None:
    _cancel_event.clear()


def should_cancel() -> bool:
    return _cancel_event.is_set()


def register_mineru_server(server: object) -> None:
    with _active_servers_lock:
        _active_servers.append(server)


def unregister_mineru_server(server: object) -> None:
    with _active_servers_lock:
        try:
            _active_servers.remove(server)
        except ValueError:
            pass


def _global_cache_path(stem: str) -> Path:
    """Path to the shared, cross-request MinerU markdown cache for one PDF
    stem. Engine-internal extraction dedup only — never read by a route; the
    app-facing cache is each user's Postgres ``mineru_markdown`` asset."""
    return DIRECTORY_OUTPUT_MINERU / f"{stem}.md"


async def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    backend: str = "pipeline",
) -> tuple[Path, Path]:
    """Parse a PDF with MinerU and reorganize output.

    Returns ``(md_path, images_dir)`` where:
      - ``md_path`` is ``<output_dir>/<stem>.md``
      - ``images_dir`` is ``<output_dir>/images/``

    Images are named ``<stem>-<padded>.<ext>`` and markdown references are
    rewritten to match the new layout.
    """
    from mineru.cli import api_client

    if should_cancel():
        raise RuntimeError("MinerU parse cancelled")

    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    extracted_md = output_dir / f"{stem}.md"
    if extracted_md.exists():
        log.info("MinerU already extracted for %s", stem)
        return extracted_md, images_dir

    global_md = _global_cache_path(stem)
    if global_md.exists():
        log.info("MinerU already extracted for %s (global cache)", stem)
        shutil.copy2(global_md, extracted_md)
        global_images = DIRECTORY_OUTPUT_MINERU / "images"
        if global_images.exists():
            for img in global_images.iterdir():
                if img.is_file() and img.name.startswith(stem):
                    shutil.copy2(img, images_dir / img.name)
        return extracted_md, images_dir

    if backend == "auto":
        backend = _detect_backend()
    log.info("MinerU using backend: %s", backend)

    form_data = api_client.build_parse_request_form_data(
        lang_list=["en"],
        backend=backend,
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        start_page_id=0,
        end_page_id=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=True,
        response_format_zip=True,
        return_original_file=False,
    )
    assets = [api_client.UploadAsset(path=pdf_path, upload_name=pdf_path.name)]

    import httpx

    with tempfile.TemporaryDirectory() as tmp_extract:
        tmp_dir = Path(tmp_extract)

        local = None
        if MINERU_SERVER_URL:
            base_url = MINERU_SERVER_URL.rstrip("/")
        elif getattr(sys, "frozen", False):
            # LocalAPIServer spawns `sys.executable -m mineru.cli.fast_api`,
            # which cannot work inside a PyInstaller bundle (sys.executable is
            # the frozen sidecar itself, and the ML stack isn't bundled).
            # HTTPException so the actionable message reaches the UI.
            raise HTTPException(
                status_code=503,
                detail="PDF OCR in the desktop app requires MINERU_SERVER_URL pointing at a "
                "running MinerU server (e.g. `python -m mineru.cli.fast_api` from the repo venv).",
            )
        else:
            local = api_client.LocalAPIServer()
            register_mineru_server(local)
            base_url = local.start()
        async with httpx.AsyncClient(timeout=api_client.build_http_timeout()) as cli:
            try:
                if local is not None:
                    await api_client.wait_for_local_api_ready(cli, local)
                sub = await api_client.submit_parse_task(base_url, assets, form_data)
                await api_client.wait_for_task_result(cli, sub, task_label=pdf_path.name)
                zp = await api_client.download_result_zip(cli, sub, task_label=pdf_path.name)
                api_client.safe_extract_zip(zp, tmp_dir)
                zp.unlink(missing_ok=True)
            finally:
                if local is not None:
                    unregister_mineru_server(local)
                    local.stop()

        md_files = sorted(tmp_dir.glob("**/*.md"), key=lambda p: len(p.name))
        if not md_files:
            raise RuntimeError(f"MinerU produced no .md output in {tmp_dir}")
        md_path = md_files[0]
        md_content = md_path.read_text(encoding="utf-8")

        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}
        all_images = sorted(
            [p for p in tmp_dir.rglob("*") if p.suffix.lower() in image_exts],
            key=lambda p: p.name,
        )

        count = len(all_images)
        width = len(str(count)) if count > 0 else 1

        renaming: list[tuple[Path, Path]] = []
        for idx, img_path in enumerate(all_images, start=1):
            new_name = f"{stem}-{idx:0{width}d}{img_path.suffix.lower()}"
            new_dest = images_dir / new_name
            renaming.append((img_path, new_dest))

        for old_path, new_path in renaming:
            old_name = old_path.name
            new_name = new_path.name
            md_content = re.sub(
                r"\]\([^)]*" + re.escape(old_name) + r"\)",
                r"](images/" + new_name + r")",
                md_content,
            )

        for old_path, new_path in renaming:
            shutil.move(str(old_path), str(new_path))

        final_md_path = output_dir / f"{stem}.md"
        final_md_path.write_text(md_content, encoding="utf-8")

    if output_dir != DIRECTORY_OUTPUT_MINERU:
        global_md_path = _global_cache_path(stem)
        if not global_md_path.exists():
            # A direct-chat parse's output_dir is a temp dir, so nothing has
            # created the global Nextcloud mirror tree yet — this can be the
            # very first PDF processed since the backend started.
            global_images_dir = DIRECTORY_OUTPUT_MINERU / "images"
            global_images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_md_path, global_md_path)
            for img in images_dir.iterdir():
                if img.is_file() and img.name.startswith(stem):
                    shutil.copy2(img, global_images_dir / img.name)

    return final_md_path, images_dir


def _detect_backend() -> str:
    # nvidia-smi presence stands in for torch.cuda.is_available() so the slim
    # desktop sidecar (which excludes torch) can still pick a sensible backend.
    if shutil.which("nvidia-smi") is not None:
        return "hybrid-auto-engine"
    return "pipeline"
