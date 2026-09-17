"""Classify legacy Documents files for the PostgreSQL import and verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.storage_namespace import legacy_key_to_native


@dataclass(frozen=True)
class LegacyArtifact:
    source: Path
    legacy_key: str
    key: str
    group: str
    kind: str | None


def inventory(source: Path) -> tuple[list[LegacyArtifact], Path | None]:
    """Return importable files and the special vault-root registry, if present."""
    artifacts: list[LegacyArtifact] = []
    vault_config: Path | None = None
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_dir():
            if relative.startswith("chat_history/") and not any(child.is_file() for child in path.rglob("*")):
                artifacts.append(LegacyArtifact(path, relative, legacy_key_to_native(relative), "empty_directories", None))
            continue
        if relative == "chat_history/file-vault-roots.json":
            vault_config = path
            continue
        classified = _classify(path, relative)
        if classified is not None:
            artifacts.append(classified)
    return artifacts, vault_config


def _classify(path: Path, legacy_key: str) -> LegacyArtifact | None:
    if legacy_key.startswith("chat_history/_uploads/") or "/_uploads/" in legacy_key:
        return LegacyArtifact(path, legacy_key, legacy_key_to_native(legacy_key), "upload", "upload")
    if legacy_key.startswith("PDFs/"):
        return LegacyArtifact(path, legacy_key, legacy_key_to_native(legacy_key), "pdf", "pdf")
    if legacy_key.startswith("Mineru/images/"):
        return LegacyArtifact(path, legacy_key, legacy_key_to_native(legacy_key), "mineru_image", "mineru_image")
    if legacy_key.startswith("Mineru/") and path.suffix == ".md":
        return LegacyArtifact(path, legacy_key, legacy_key_to_native(legacy_key), "mineru_markdown", "mineru_markdown")
    if legacy_key.startswith("Mineru/"):
        return None
    if legacy_key.startswith("Drawings/"):
        return LegacyArtifact(path, legacy_key, legacy_key_to_native(legacy_key), "drawing", "drawing")
    return LegacyArtifact(path, legacy_key, legacy_key_to_native(legacy_key), "documents", None)
