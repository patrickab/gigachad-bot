"""The user's oh-lucy-evening scheme from `~/.config/theme/palette.json` (deployed by the ~/.os dotfiles), as CSS variables.

The file is the one source for Neovim, Pi and this app. Absent or unreadable, the app keeps the fallback copy in
`packages/chat-ui/src/scheme.css`. The variable names match that file: `--md-*` (markup roles), `--nb-syn-*` (syntax roles).
"""

import json
from pathlib import Path

PALETTE_FILE = Path.home() / ".config" / "theme" / "palette.json"

# `:root:root` beats the fallback's `html:root` whatever the stylesheet order
_SELECTORS = {"dark": "html:root:root", "light": "html.light:root:root"}


def render_theme_css(path: Path = PALETTE_FILE) -> str | None:
    """CSS overriding the fallback scheme, or None when there is no usable palette file."""
    try:
        palette = json.loads(path.read_text(encoding="utf-8"))
        blocks = []
        for mode, selector in _SELECTORS.items():
            colors = palette[mode]
            lines = [f"  --md-{role.replace('_', '-')}: {colors[key]};" for role, key in palette["markup"].items()]
            lines += [f"  --nb-syn-{role}: {colors[key]};" for role, key in palette["syntax"].items()]
            blocks.append(f"{selector} {{\n" + "\n".join(lines) + "\n}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return "\n".join(blocks) + "\n"
