"""Thumbnails of Pi tool-result images, served by `GET /api/pi/media/<sha256>/thumb`.

pi-host stores the images under their content hash (`pi-host/src/media.ts`), this module turns one into a WebP of at
most `THUMB_PX` on its long side and keeps the latest ones in memory (a hash never changes meaning, so there is nothing to invalidate).
"""

from collections import OrderedDict
import io

from PIL import Image, ImageOps

# Twice the 240px tile the transcript shows, for high-density screens.
THUMB_PX = 480
CACHE_MAX_BYTES = 64 * 1024 * 1024
HASH_PATTERN = r"^[0-9a-f]{64}$"

_cache: OrderedDict[str, bytes] = OrderedDict()
_cache_bytes = 0


def make_thumbnail(data: bytes) -> bytes:
    """*data* (PNG, JPEG, GIF or WebP) scaled down to `THUMB_PX` as WebP, smaller images keep their size. ValueError if unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            oriented.thumbnail((THUMB_PX, THUMB_PX))
            if oriented.mode not in ("RGB", "RGBA"):
                oriented = oriented.convert("RGBA" if "transparency" in oriented.info or oriented.mode == "P" else "RGB")
            out = io.BytesIO()
            oriented.save(out, format="WEBP", quality=80)
            return out.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"cannot thumbnail image: {exc}") from exc


def thumbnail_for(hash_: str, load: bytes | None = None) -> bytes | None:
    """The cached thumbnail of *hash_*, or the one made from *load* (stored), None when neither exists."""
    global _cache_bytes
    if (cached := _cache.get(hash_)) is not None:
        _cache.move_to_end(hash_)
        return cached
    if load is None:
        return None
    thumb = make_thumbnail(load)
    _cache[hash_] = thumb
    _cache_bytes += len(thumb)
    while _cache_bytes > CACHE_MAX_BYTES and len(_cache) > 1:
        _cache_bytes -= len(_cache.popitem(last=False)[1])
    return thumb


def forget(hashes: list[str]) -> None:
    """Drop the cached thumbnails of *hashes*: pi-host deleted their images with a turn."""
    global _cache_bytes
    for hash_ in hashes:
        if (thumb := _cache.pop(hash_, None)) is not None:
            _cache_bytes -= len(thumb)
