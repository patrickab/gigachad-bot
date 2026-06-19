import base64
import re
from pathlib import Path

from config import chat_upload_dir
from llm_client import LLMClient

DOWNSCALED_SUBDIR = "_downscaled"
_DEFAULT_MAX_TOKENS = 2048
_DATA_URI_RE = re.compile(r"data:image/\w+;base64,(.+)")


def downscaled_path(chat_dir: Path, filename: str) -> Path:
    return chat_dir / DOWNSCALED_SUBDIR / f"{Path(filename).stem}.jpg"


def write_downscaled(client: LLMClient, src: Path, dst: Path, max_tokens: int = _DEFAULT_MAX_TOKENS) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    data_uri = client.downscale_img(str(src), max_tokens=max_tokens)
    match = _DATA_URI_RE.match(data_uri)
    if not match:
        raise ValueError("downscale_img returned invalid data URI")
    dst.write_bytes(base64.b64decode(match.group(1)))
    return dst


def resolve_chat_image_paths(
    client: LLMClient,
    chat_id: str,
    slug: str | None,
    filenames: list[str],
    downscale: bool,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[Path]:
    if not filenames:
        return []
    chat_dir = chat_upload_dir(chat_id, slug)
    paths: list[Path] = []
    for name in filenames:
        src = chat_dir / name
        if not src.is_file():
            continue
        if downscale:
            paths.append(write_downscaled(client, src, downscaled_path(chat_dir, name), max_tokens))
        else:
            paths.append(src)
    return paths


def delete_downscaled(chat_dir: Path, filename: str) -> None:
    dst = downscaled_path(chat_dir, filename)
    if dst.exists():
        dst.unlink()
