"""Images attached to a factory task's request or message, stored on the host where the coder's `read` tool opens them.

The factory core takes text only, so the frontend appends each stored image's absolute path to the text (`mention_images`
on its side). Files live under `IMAGES_DIRECTORY`, outside every repo, so nothing is committed."""

from datetime import datetime
from pathlib import Path
import re
import uuid

IMAGES_DIRECTORY = Path("~/.cache/gigachad/factory-images").expanduser()
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
_ALIASES = {".jpeg": ".jpg"}


def image_suffix(filename: str, content_type: str | None) -> str:
    """The image suffix of *filename*, else of *content_type*; ValueError for anything but png, jpg, gif and webp."""
    suffix = Path(filename).suffix.lower()
    suffix = _ALIASES.get(suffix, suffix) or IMAGE_SUFFIXES.get(content_type or "", "")
    if suffix not in IMAGE_SUFFIXES.values():
        raise ValueError("Only png, jpg, gif and webp images can be attached")
    return suffix


def store_image(filename: str, suffix: str, content: bytes, now: datetime, directory: Path | None = None) -> Path:
    """Write *content* to `<YYYYMMDD-HHMMSS>-<id>-<name><suffix>` in *directory* (default `IMAGES_DIRECTORY`), return its path.
    The client's name loses its folders and anything a path or an `@` mention splits on."""
    target = directory or IMAGES_DIRECTORY
    stem = re.sub(r"[^\w.-]+", "_", Path(filename).stem)[:60].strip("._") or "image"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}-{stem}{suffix}"
    path.write_bytes(content)
    return path
