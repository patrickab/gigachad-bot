"""Videos pasted into a Pi composer, stored in the session folder's `.pi/pastes/` where the user's Pi extension
`~/.pi/agent/extensions/video-paste.ts` (from `~/.os`) turns an `@` mention of them into frames."""

from datetime import datetime
from pathlib import Path
import re

from lib.file_roots import FileRoots

PASTES_DIRECTORY = Path(".pi") / "pastes"
MAX_PASTE_BYTES = 200 * 1024 * 1024
# The suffixes the extension previews, by the MIME types browsers give them
VIDEO_SUFFIXES = {"video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov", "video/x-matroska": ".mkv"}


class UnknownFolder(LookupError):
    """The session folder is relative, missing or outside the file roots."""


def paste_suffix(filename: str, content_type: str | None) -> str:
    """The video suffix of *filename*, else of *content_type*, ValueError for anything the extension cannot preview."""
    suffix = Path(filename).suffix.lower() or VIDEO_SUFFIXES.get(content_type or "", "")
    if suffix not in VIDEO_SUFFIXES.values():
        raise ValueError("Only mp4, webm, mov and mkv videos can be pasted")
    return suffix


def paste_mention(filename: str, suffix: str, now: datetime) -> Path:
    """`.pi/pastes/<YYYYMMDD-HHMMSS>-<name><suffix>`: the client's name loses its folders and anything an `@` mention splits on."""
    stem = re.sub(r"[^\w.-]+", "_", Path(filename).stem)[:80].strip("._") or "video"
    return PASTES_DIRECTORY / f"{now:%Y%m%d-%H%M%S}-{stem}{suffix}"


def store_paste(roots: FileRoots, folder: Path, mention: Path, content: bytes) -> Path:
    """Create `folder / mention` exclusively (FileExistsError when taken) next to a `.gitignore` of `*` (created when missing),
    so pastes are never committed. UnknownFolder for an unreachable folder, ValueError when `.pi`, `.pi/pastes` or that
    `.gitignore` is a symlink."""
    if not folder.is_absolute():
        raise UnknownFolder(f"Unknown folder {folder}")
    folder = folder.resolve()
    if not (folder.is_dir() and roots.contains(folder)):
        raise UnknownFolder(f"Unknown folder {folder}")
    pastes = folder / mention.parent
    ignore = pastes / ".gitignore"
    for part in (folder / mention.parts[0], pastes, ignore):
        if part.is_symlink():
            raise ValueError(f"{part} is a symlink")
    pastes.mkdir(parents=True, exist_ok=True)
    if not ignore.exists():
        try:
            roots.create(str(ignore), "*\n")
        except FileExistsError:
            pass
    return roots.create(str(folder / mention), content)
