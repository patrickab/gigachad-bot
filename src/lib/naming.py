"""Shared naming utilities: slug generation and filename deduplication."""

from pathlib import Path
import re


def slugify(text: str, fallback: str = "item") -> str:
    """Lowercase, strip non-alphanumeric, collapse dashes, truncate to 64 chars."""
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)
    return (s or fallback)[:64]


def dedup_filename(dest_dir: Path, filename: str) -> str:
    """Return *filename* as-is if free, otherwise append ``(n)`` until unique."""
    stem, suffix = Path(filename).stem, Path(filename).suffix
    if not (dest_dir / filename).exists():
        return filename
    n = 1
    while (dest_dir / f"{stem} ({n}){suffix}").exists():
        n += 1
    return f"{stem} ({n}){suffix}"
