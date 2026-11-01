"""Shared naming utilities."""

import re


def slugify(text: str, fallback: str = "item") -> str:
    """Lowercase, strip non-alphanumeric, collapse dashes, truncate to 64 chars."""
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)
    return (s or fallback)[:64]


def segment(name: str, *, hidden: bool = False) -> str:
    """*name* as a single path segment. ValueError for an empty name, a separator, a control character, `.`/`..` or (unless *hidden*) a leading dot."""
    invalid = any(ord(character) < 32 or character in "/\\" for character in name)
    if not name or invalid or name in {".", ".."} or (name.startswith(".") and not hidden):
        raise ValueError(f"Invalid name: {name!r}")
    return name

