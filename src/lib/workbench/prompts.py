"""The system prompt of a workbench chat (Pi `document` session), rendered on every create and resume (G2).

- Consumes the prompt files in `templates/workbench/<kind>.md`, one per chat kind
- `{{TEMPLATES}}` in a prompt becomes the absolute folder of the LaTeX template sets, so Pi reads them where they ship
- A chat working outside the workbench folder gets the `AGENTS.md` files of its folder appended, from its repository's
  top down to the folder: Pi runs with `--no-context-files`, which drops them too
- A prompt is never an existing file path: Pi would read that file instead
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import config
from lib import latex_templates
from lib.git import repository_top

ChatKind = Literal["latex", "markdown", "pdf", "canvas", "notebook"]
CHAT_KINDS: tuple[ChatKind, ...] = get_args(ChatKind)
CONTEXT_FILE = "AGENTS.md"
# pi-host caps a prompt at 128 KiB, each appended file keeps well below.
_MAX_CONTEXT_BYTES = 32 * 1024
_TEMPLATES_TOKEN = "{{TEMPLATES}}"


def _prompts() -> Path:
    return config.TEMPLATES / "workbench"


def render(kind: ChatKind, cwd: Path | None = None) -> str:
    """The system prompt of a *kind* chat working in *cwd*, with the context files of *cwd* when it is outside the workbench."""
    prompt = (_prompts() / f"{kind}.md").read_text(encoding="utf-8").replace(_TEMPLATES_TOKEN, str(latex_templates.TEMPLATE_SETS))
    sections = [prompt.rstrip("\n")]
    if cwd is not None and not cwd.is_relative_to(config.WORKBENCH):
        sections.extend(_context_section(path) for path in context_files(cwd))
    rendered = "\n\n".join(sections) + "\n"
    if _is_file_path(rendered):
        raise ValueError(f"The {kind} prompt is a file path, Pi would read the file")
    return rendered


def _is_file_path(prompt: str) -> bool:
    try:
        return "\n" not in prompt.strip() and Path(prompt.strip()).exists()
    except OSError:
        return False


def context_files(cwd: Path) -> list[Path]:
    """The `AGENTS.md` files from the repository top holding *cwd* down to *cwd*, outermost first, else *cwd*'s own."""
    top = repository_top(cwd)
    folders = [cwd] if top is None or not cwd.is_relative_to(top) else [top, *_between(top, cwd)]
    return [path for folder in folders if (path := folder / CONTEXT_FILE).is_file()]


def _between(top: Path, cwd: Path) -> list[Path]:
    """The folders below *top* down to *cwd*, outermost first."""
    return [top.joinpath(*cwd.relative_to(top).parts[: depth + 1]) for depth in range(len(cwd.relative_to(top).parts))]


def _context_section(path: Path) -> str:
    content = path.read_bytes()[:_MAX_CONTEXT_BYTES].decode("utf-8", errors="ignore").strip()
    return f"# Project instructions from `{path}`\n\n{content}"
