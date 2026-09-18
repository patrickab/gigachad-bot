"""Regression test for ``lib.mineru.parse_pdf``'s local-output-to-global-Nextcloud handoff.

A direct-chat parse gives MinerU a scratch temp directory as ``output_dir`` (see
``backend/routes/files.py::_parse_assets``), distinct from the global
``DIRECTORY_OUTPUT_MINERU`` Nextcloud mirror tree that a prior library/queue
extraction would have created. The final handoff copies the parsed markdown
and images into that global tree — it must not depend on the tree already
existing, or the very first PDF processed since the backend started fails.
"""

from pathlib import Path

import pytest

import lib.mineru as mineru


class _FakeLocalServer:
    def start(self) -> str:
        return "http://127.0.0.1:0"

    def stop(self) -> None:
        pass


async def _noop_async(*_args, **_kwargs) -> None:
    return None


async def test_parse_pdf_creates_global_mineru_directories_before_first_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    global_dir = tmp_path / "Nextcloud" / "Mineru"
    assert not global_dir.exists()
    monkeypatch.setattr(mineru, "DIRECTORY_OUTPUT_MINERU", global_dir, raising=True)

    from mineru.cli import api_client

    monkeypatch.setattr(api_client, "build_parse_request_form_data", lambda **_kw: {})
    monkeypatch.setattr(api_client, "UploadAsset", lambda **kw: kw)
    monkeypatch.setattr(api_client, "build_http_timeout", lambda: 5)
    monkeypatch.setattr(api_client, "LocalAPIServer", _FakeLocalServer)
    monkeypatch.setattr(api_client, "wait_for_local_api_ready", _noop_async)

    async def fake_submit_parse_task(*_a, **_kw):
        return "task-id"

    monkeypatch.setattr(api_client, "submit_parse_task", fake_submit_parse_task)
    monkeypatch.setattr(api_client, "wait_for_task_result", _noop_async)

    async def fake_download_result_zip(_cli, _sub, *, task_label):  # noqa: ARG001
        return tmp_path / "result.zip"

    def fake_safe_extract_zip(_zp, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "out.md").write_text("# parsed\n![p](page.png)", encoding="utf-8")
        (dest / "page.png").write_bytes(b"\x89PNG fake")

    monkeypatch.setattr(api_client, "download_result_zip", fake_download_result_zip)
    monkeypatch.setattr(api_client, "safe_extract_zip", fake_safe_extract_zip)

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    # A scratch dir distinct from DIRECTORY_OUTPUT_MINERU, matching a direct-chat parse.
    output_dir = tmp_path / "chat-scratch" / "mineru"

    md_path, images_dir = await mineru.parse_pdf(pdf, output_dir)

    assert md_path.read_text(encoding="utf-8").startswith("# parsed")
    assert (images_dir / "report-1.png").is_file()

    global_md = global_dir / "report.md"
    assert global_md.is_file()
    assert "images/report-1.png" in global_md.read_text(encoding="utf-8")
    assert (global_dir / "images" / "report-1.png").is_file()
