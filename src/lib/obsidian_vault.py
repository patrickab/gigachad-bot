"""Read-only access seam for the Obsidian vault.

Mirrors the path-traversal guard pattern used by ``ChatStore``. The vault is an
external, user-owned directory (``DIRECTORY_OBSIDIAN_VAULT``); this store only
ever reads from it. When the vault is unconfigured or missing, it reports
``enabled = False`` and every listing returns empty so the feature degrades
quietly for users without Obsidian.

Named ``ObsidianVault`` (not ``VaultStore``) because "vault" already refers to
the sidebar projects/histories tree (``VaultTree``) elsewhere in the codebase.
"""

from pathlib import Path

from config import DIRECTORY_OBSIDIAN_VAULT
from lib.safe_path import safe_resolve

# Directories that hold Obsidian internals or noise rather than notes.
_SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules"}


class ObsidianVault:
    """All read access to the Obsidian vault goes through this class."""

    def __init__(self, base_dir: Path | None = None) -> None:
        raw = base_dir if base_dir is not None else DIRECTORY_OBSIDIAN_VAULT
        self._base = Path(raw).expanduser().resolve() if str(raw) else None

    @property
    def enabled(self) -> bool:
        return self._base is not None and self._base.is_dir()

    def resolve_path(self, rel_path: str) -> Path:
        """Resolve *rel_path* under the vault, rejecting traversal."""
        if self._base is None:
            raise ValueError("Obsidian vault is not configured")
        return safe_resolve(self._base, rel_path)

    def list_markdown(self) -> list[dict[str, str]]:
        """Return every markdown note as ``{path, name}`` with posix-relative paths.

        Sorted by path so directories group naturally, letting the frontend
        reconstruct the folder tree and fuzzy-search across the flat list.
        """
        if not self.enabled or self._base is None:
            return []
        files: list[dict[str, str]] = []
        for path in self._base.rglob("*.md"):
            if not path.is_file():
                continue
            rel = path.relative_to(self._base)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            files.append({"path": rel.as_posix(), "name": path.name})
        files.sort(key=lambda f: f["path"].lower())
        return files

    def read(self, rel_path: str) -> str:
        """Return the text content of a markdown note."""
        path = self.resolve_path(rel_path)
        if not path.is_file():
            raise FileNotFoundError(rel_path)
        return path.read_text(encoding="utf-8")
