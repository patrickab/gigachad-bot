"""Shared path-traversal guard for all stores."""

from pathlib import Path


def safe_resolve(base: Path, rel: str) -> Path:
    """Resolve *rel* under *base*, raising ``ValueError`` on traversal."""
    resolved = (base / rel).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"Invalid path: {rel}") from None
    return resolved
