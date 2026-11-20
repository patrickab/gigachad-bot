"""The system prompt of a workbench chat (Pi `document` session), rendered on every create and resume (G2).

- Consumes the prompt files in `templates/workbench/<kind>.md`, one per chat kind
- `{{TEMPLATES}}` in a prompt becomes the absolute folder of the LaTeX template sets, so Pi reads them where they ship
- pi-host appends the prompt to Pi's own, which also loads the chat folder's `AGENTS.md` files
- A prompt is never an existing file path: Pi would read that file instead
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import config
from lib import latex_templates

ChatKind = Literal["latex", "markdown", "pdf", "canvas", "notebook"]
CHAT_KINDS: tuple[ChatKind, ...] = get_args(ChatKind)
_TEMPLATES_TOKEN = "{{TEMPLATES}}"


def _prompts() -> Path:
    return config.TEMPLATES / "workbench"


def render(kind: ChatKind) -> str:
    """The system prompt of a *kind* chat."""
    prompt = (_prompts() / f"{kind}.md").read_text(encoding="utf-8").replace(_TEMPLATES_TOKEN, str(latex_templates.TEMPLATE_SETS))
    rendered = prompt.rstrip("\n") + "\n"
    if _is_file_path(rendered):
        raise ValueError(f"The {kind} prompt is a file path, Pi would read the file")
    return rendered


def _is_file_path(prompt: str) -> bool:
    try:
        return "\n" not in prompt.strip() and Path(prompt.strip()).exists()
    except OSError:
        return False


