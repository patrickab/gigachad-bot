"""Neovim's `tohtml` export of a notebook cell as styled lines, what the browser draws for a cell that is not being edited.

`tohtml` (Neovim's own HTML export, which knows the window's number column, signs, conceal, diagnostics and every highlight
source) writes class names and one CSS rule per highlight group. The browser never sees that markup: `parse_tohtml` turns it into
plain data, and the browser draws text nodes with a validated style, so nothing from the file or from a language server can become
markup there.

    {"styles": [{}, {"fg": "#c61365", "italic": true}, ...], "lines": [[["import", 1], [" os", 0]], ...], "base": {"fg": "#ded7d0"},
     "rows": [0, 1, ...], "height": 2}

A line is a list of `[text, style]` runs, `style` an index into `styles` (`styles[0]` is always `{}`: no colors). A style may hold
`fg`, `bg` and `sp` (`#rrggbb`), `bold`, `italic`, `strike` (`true`) and `underline` (`"line"` or `"curl"`). The number column and
the signs are text runs of the line like any other, and the text of diagnostics (virtual text) follows the line it belongs to. A
line that `virt_lines` adds (rare) is a line of its own. `base` is the `fg` and `bg` of text without a style, when the colorscheme
sets them. `rows` holds the cell line (0-based) every line shows, so that a click on a `virt_lines` line lands on the line it
stands above. `height` is the screen rows the cell takes in Neovim's window (wrapped lines, folds, virtual lines), the room the
terminal takes there too. `wrap`, only when that window wraps lines, is the column a wrapped line goes on at (after the number
column and the signs).
"""

from __future__ import annotations

import re
from typing import Any

Style = dict[str, Any]
Run = list[Any]
Rendering = dict[str, Any]

_STYLE_RULE = re.compile(r"^\.(\S+) \{(.*)\}$")
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
# An opening `span` with its class, a closing `span`, or text up to the next tag (a lone `<` is text: `virt_lines` text is not escaped)
_TOKEN = re.compile(r'<span class="([^"]*)">|</span>|([^<]+|<)')
_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"'}
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot);")
_PRE = re.compile(r"<pre>(.*)</pre>", re.DOTALL)


def _unescape(text: str) -> str:
    return _ENTITY.sub(lambda match: _ENTITIES[match.group()], text) if "&" in text else text


def _rules(html: str) -> dict[str, Style]:
    """The style of every class in the `<style>` block of *html*."""
    rules: dict[str, Style] = {}
    start, end = html.find("<style>"), html.find("</style>")
    if start < 0 or end < 0:
        return rules
    for line in html[start:end].splitlines():
        if (match := _STYLE_RULE.match(line)) is None:
            continue
        declarations = dict(part.split(": ", 1) for part in match.group(2).split("; ") if ": " in part)
        style: Style = {}
        for key, name in (("color", "fg"), ("background-color", "bg"), ("text-decoration-color", "sp")):
            if _COLOR.fullmatch(declarations.get(key, "")):
                style[name] = declarations[key]
        if declarations.get("font-weight") == "bold":
            style["bold"] = True
        if declarations.get("font-style") == "italic":
            style["italic"] = True
        decoration = declarations.get("text-decoration-line", "")
        if "underline" in decoration:
            style["underline"] = "curl" if declarations.get("text-decoration-style") == "wavy" else "line"
        if "line-through" in decoration:
            style["strike"] = True
        rules[match.group(1)] = style
    return rules


def _merged(stack: list[Style]) -> Style:
    """The style of text inside the nested classes *stack*: the innermost color wins, decorations add up."""
    style: Style = {}
    for layer in stack:
        style.update(layer)
    return style


_FLAGS = ("bold", "italic", "strike")


def _clean(style: object) -> Style:
    """*style* reduced to what a style may hold: colors as `#rrggbb`, flags as `True`, `underline` as `"line"` or `"curl"`."""
    cleaned: Style = {}
    if not isinstance(style, dict):
        return cleaned
    for name in ("fg", "bg", "sp"):
        if isinstance(style.get(name), str) and _COLOR.fullmatch(style[name]):
            cleaned[name] = style[name]
    cleaned.update({name: True for name in _FLAGS if style.get(name) is True})
    if style.get("underline") in ("line", "curl"):
        cleaned["underline"] = style["underline"]
    return cleaned


class _Styles:
    """The distinct styles of one rendering, `styles[0]` is the empty one."""

    def __init__(self) -> None:
        self.styles: list[Style] = [{}]
        self._indexes: dict[str, int] = {repr([]): 0}

    def index(self, style: Style) -> int:
        key = repr(sorted(style.items()))
        if key not in self._indexes:
            self._indexes[key] = len(self.styles)
            self.styles.append(style)
        return self._indexes[key]


def _gutter_runs(gutter: object, styles: _Styles) -> list[Run]:
    """The runs of one gutter entry: a list of `[text, style]` as Neovim's Lua wrote it, anything else is no gutter."""
    runs: list[Run] = []
    if not isinstance(gutter, list):
        return runs
    for run in gutter:
        if isinstance(run, list) and len(run) == 2 and isinstance(run[0], str) and run[0]:
            runs.append([run[0], styles.index(_clean(run[1]))])
    return runs


def parse_tohtml(
    html: str, gutter: object = None, base: object = None, rows: object = None, height: object = None, wrap: object = None
) -> Rendering:
    """The styled lines of a `tohtml` document, `{"styles", "lines", "base", "rows", "height"}` and `wrap` when given. A document
    without `<pre>` has no lines.

    *gutter* is a list with one entry per output line, the runs `[[text, style dict], ...]` that lead that line (the number column
    and the signs, which `tohtml` cannot place in a range), `None` where a line has none. *base* is the style of the `Normal` group:
    `base` of the result holds its `fg` and `bg` when Neovim's colorscheme sets them (what text without a color is drawn in).
    *rows* is the cell line (0-based) of every output line (a `virt_lines` line belongs to the line it stands above), `rows` of the
    result has one per line: the index of the line where *rows* has no fitting entry. *height* is the screen rows the cell takes in
    Neovim's window (the number of lines when it is no positive integer), *wrap* the column a wrapped line goes on at when the
    window wraps lines (left out when it is no non-negative integer).
    """
    rules = _rules(html)
    pre = _PRE.search(html)
    styles = _Styles()
    lines: list[list[Run]] = [[]]
    stack: list[Style] = []
    index = 0
    if pre is not None:
        for token in _TOKEN.finditer(pre.group(1)):
            opening, text = token.group(1), token.group(2)
            if opening is not None:
                stack.append(rules.get(opening, {}))
                index = styles.index(_merged(stack))
            elif text is None:
                if stack:
                    stack.pop()
                    index = styles.index(_merged(stack))
            else:
                for number, part in enumerate(_unescape(text).split("\n")):
                    if number:
                        lines.append([])
                        if len(lines) == 2:
                            # The rows above the range only leave tags, and what they leave open is not always closed again
                            # (a diagnostic's underline that ends past its line): the range starts without any
                            stack.clear()
                            index = 0
                    if part:
                        runs = lines[-1]
                        if runs and runs[-1][1] == index:
                            runs[-1][0] += part
                        else:
                            runs.append([part, index])
    # The text before the first line break holds only the tags of rows above the range. `tohtml` also writes the row after the
    # range (without text) and ends with a line break
    body = lines[1:-2] if len(lines) > 2 else []
    if isinstance(gutter, list):
        body = [_gutter_runs(gutter[number] if number < len(gutter) else None, styles) + line for number, line in enumerate(body)]
    base_style = {name: value for name, value in _clean(base).items() if name in ("fg", "bg")}
    given = rows if isinstance(rows, list) else []
    source = [_row(given[number] if number < len(given) else None, number) for number in range(len(body))]
    screen = height if _row(height, 0) > 0 else len(body)
    rendering = {"styles": styles.styles, "lines": body, "base": base_style, "rows": source, "height": screen}
    if _row(wrap, -1) >= 0:
        rendering["wrap"] = wrap
    return rendering


def _row(value: object, fallback: int) -> int:
    """*value* as a cell line (a non-negative integer), else *fallback*."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else fallback
