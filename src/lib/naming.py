"""Shared naming utilities."""

import re


def slugify(text: str, fallback: str = "item") -> str:
    """Lowercase, strip non-alphanumeric, collapse dashes, truncate to 64 chars."""
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)
    return (s or fallback)[:64]

