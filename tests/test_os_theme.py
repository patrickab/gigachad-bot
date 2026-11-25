"""The ~/.os palette becomes CSS overrides, and anything unusable falls back to the bundled scheme."""

import json

from lib.os_theme import render_theme_css

PALETTE = {
    "dark": {"red": "#FF0000", "blue": "#0000FF"},
    "light": {"red": "#AA0000", "blue": "#0000AA"},
    "markup": {"link_url": "red"},
    "syntax": {"type": "blue"},
}


def test_roles_resolve_to_each_modes_colors(tmp_path):
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps(PALETTE))

    css = render_theme_css(palette)

    assert css is not None
    assert "html:root:root {\n  --md-link-url: #FF0000;\n  --nb-syn-type: #0000FF;\n}" in css
    assert "html.light:root:root {\n  --md-link-url: #AA0000;\n  --nb-syn-type: #0000AA;\n}" in css


def test_missing_or_broken_palette_gives_none(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"dark": {}}')

    assert render_theme_css(tmp_path / "absent.json") is None
    assert render_theme_css(broken) is None
