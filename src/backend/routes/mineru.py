import logging
from pathlib import Path
import re
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from config import DIRECTORY_OUTPUT_MINERU, DIRECTORY_OUTPUT_PDF, SMALL_MODEL

from .deps import request_client

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mineru", tags=["mineru"])


class MineruResult(BaseModel):
    filename: str
    output_dir: str
    markdown_content: str
    answer: str | None = None
    query: str | None = None


class MineruBatchResponse(BaseModel):
    results: list[MineruResult]


async def _parse_pdf(
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

    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

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

        local = api_client.LocalAPIServer()
        base_url = local.start()
        async with httpx.AsyncClient(timeout=api_client.build_http_timeout()) as cli:
            try:
                await api_client.wait_for_local_api_ready(cli, local)
                sub = await api_client.submit_parse_task(base_url, assets, form_data)
                await api_client.wait_for_task_result(cli, sub, task_label=pdf_path.name)
                zp = await api_client.download_result_zip(cli, sub, task_label=pdf_path.name)
                api_client.safe_extract_zip(zp, tmp_dir)
                zp.unlink(missing_ok=True)
            finally:
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
            old_rel = str(old_path.relative_to(tmp_dir))
            new_rel = str(new_path.relative_to(output_dir))
            md_content = md_content.replace(old_rel, new_rel)

        for old_path, new_path in renaming:
            old_path.rename(new_path)

        final_md_path = output_dir / f"{stem}.md"
        final_md_path.write_text(md_content, encoding="utf-8")

    return final_md_path, images_dir


def _detect_backend() -> str:
    import torch

    if torch.cuda.is_available():
        return "hybrid-auto-engine"
    return "pipeline"


def _rewrite_images_for_frontend(md_content: str) -> str:
    """Rewrite relative ``images/`` refs to absolute ``/mineru/images/`` URLs."""
    return re.sub(r'\(images/([^)]+)\)', r'(/mineru/images/\1)', md_content)


@router.post("/parse", response_model=MineruResult)
async def parse_single_pdf(
    file: UploadFile = File(...),
    query: str = Form(""),
    backend: str = Form("pipeline"),
    model: str = Form(SMALL_MODEL),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        content_b = await file.read()
        tmp_path.write_bytes(content_b)

        (DIRECTORY_OUTPUT_PDF / file.filename).write_bytes(content_b)

        md_path, _images_dir = await _parse_pdf(tmp_path, DIRECTORY_OUTPUT_MINERU, backend=backend)
        md_content = md_path.read_text(encoding="utf-8")

    result = MineruResult(
        filename=file.filename,
        output_dir=str(DIRECTORY_OUTPUT_MINERU),
        markdown_content=_rewrite_images_for_frontend(md_content),
    )

    if query.strip():
        query_clean = query.strip()
        result.query = query_clean
        try:
            with request_client() as c:
                response = c.api_query(
                    model=model,
                    user_msg=md_content + "\n\n---\n\n" + query_clean,
                    system_prompt="",
                    img=None,
                    stream=False,
                )
                result.answer = response.choices[0].message.content or ""
        except Exception:
            log.exception("LLM query failed for %s", file.filename)
            result.answer = "(Failed to generate answer)"

    return result


@router.post("/parse-batch", response_model=MineruBatchResponse)
async def parse_batch_pdfs(
    files: list[UploadFile] = File(...),
    query: str = Form(""),
    backend: str = Form("pipeline"),
    model: str = Form(SMALL_MODEL),
):
    results = []
    shared_md = ""
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported, got: {file.filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / file.filename
            content_b = await file.read()
            tmp_path.write_bytes(content_b)

            (DIRECTORY_OUTPUT_PDF / file.filename).write_bytes(content_b)

            md_path, _images_dir = await _parse_pdf(tmp_path, DIRECTORY_OUTPUT_MINERU, backend=backend)
            md_raw = md_path.read_text(encoding="utf-8")
            md_content = _rewrite_images_for_frontend(md_raw)

        shared_md += f"\n\n### {file.filename}\n\n{md_raw}"
        results.append(
            MineruResult(
                filename=file.filename,
                output_dir=str(DIRECTORY_OUTPUT_MINERU),
                markdown_content=md_content,
            )
        )

    query_clean = query.strip()
    if query_clean:
        try:
            with request_client() as c:
                response = c.api_query(
                    model=model,
                    user_msg=shared_md + "\n\n---\n\n" + query_clean,
                    system_prompt="",
                    img=None,
                    stream=False,
                )
                answer = response.choices[0].message.content or ""
        except Exception:
            log.exception("LLM query failed for batch")
            answer = "(Failed to generate answer)"
        for r in results:
            r.query = query_clean
            r.answer = answer

    return MineruBatchResponse(results=results)
