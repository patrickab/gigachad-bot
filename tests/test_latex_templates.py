import os
from pathlib import Path

import pytest

from lib import latex_templates
from tests.conftest import PRESENTATION_SOURCE, PREVIEW_PNG, PRIMITIVE_FRAME
from tests.test_latex import ROOT_SOURCE, write_files


def preview_version(base: Path) -> int:
    """The `preview` of `text/bullets` in the fake template sets at *base*."""
    return (base / "demo" / "previews" / "text" / "01-bullets.png").stat().st_mtime_ns // 1_000_000


def demo() -> latex_templates.TemplateSet:
    template = latex_templates.template_set("demo")
    assert template is not None
    return template


@pytest.mark.parametrize(("source", "theme"), [
    (PRESENTATION_SOURCE, "demo"),
    ("\\documentclass{beamer}\n\\usetheme { demo }\n", "demo"),
    ("\\documentclass{beamer}\n\\usetheme{Madrid}\n\\usetheme{compact}\n", "compact"),
    ("\\documentclass{beamer}\n% \\usetheme{demo}\n", None),
    ("\\documentclass{beamer}\n\\usetheme{Madrid}\n", None),
    ("\\documentclass{beamer}\n\\usetheme{letter}\n", None),
    ("\\documentclass{beamer}\n\\usetheme{bare}\n", None),
    ("\\documentclass{beamer}\n\\usetheme{broken}\n", None),
    (ROOT_SOURCE, None),
])
def test_a_root_is_a_presentation_when_its_theme_names_a_template_set(tmp_path, template_sets, source, theme):
    write_files(tmp_path, {"main.tex": source})

    found = latex_templates.presentation_theme(tmp_path / "main.tex")

    assert (found and found.name) == theme
    assert latex_templates.presentation_theme(tmp_path / "missing.tex") is None


def test_template_sets_are_the_folders_with_a_skeleton_and_a_valid_manifest(template_sets, caplog):
    found = latex_templates.template_sets()

    assert [record.getMessage().split("/")[-1] for record in caplog.records] == ["broken with a malformed template.toml"]
    assert [(template.name, template.label, template.files, template.is_presentation) for template in found] == [
        ("compact", "Compact", ("beamerthemecompact.sty",), True),
        ("demo", "Demo deck", ("beamerthemedemo.sty",), True),
        ("letter", "Letter", ("sender.lco",), False),
    ]
    assert latex_templates.template_set("compact").path == template_sets / "compact"
    assert latex_templates.template_set("letter").primitives() == []
    assert latex_templates.template_set("letter").image_reference("imgs/a.png") is None
    assert latex_templates.template_set("bare") is None
    assert latex_templates.template_set("../sets/demo") is None


def test_primitives_come_by_group_then_number_skipping_malformed_files(template_sets, caplog):
    template = demo()
    caplog.clear()

    found = template.primitives()

    assert [primitive.id for primitive in found] == [
        "text/bullets", "text/statement", "numbers/early", "numbers/late", "figures/figrow", "appendix/extra",
    ]
    assert found[0] == latex_templates.LatexPrimitive(
        id="text/bullets", group="text", group_label="Text", label="Bullets", description="points with one message",
        summary="Points with one message", text=PRIMITIVE_FRAME, preview=preview_version(template_sets),
    )
    assert found[1].preview is None
    assert found[1].summary == "A thesis or quote"
    assert found[2].label == "Key figures"
    assert found[-1].group_label == "Appendix"
    assert sorted(record.getMessage().rsplit("/", 1)[-1] for record in caplog.records) == [
        "03-headless.tex", "04-twice.tex", "unnumbered.tex",
    ]


def test_a_second_set_orders_by_its_own_groups_and_references_images_by_its_own_macro(template_sets):
    compact = latex_templates.template_set("compact")

    assert [(primitive.id, primitive.group_label) for primitive in compact.primitives()] == [
        ("figures/photo", "Pictures"), ("text/quote", "Words"),
    ]
    assert compact.image_reference("imgs/a.png") == "\\photo{imgs/a.png}"
    assert demo().image_reference("imgs/a.png") == "\\demoimage{imgs/a.png}"


def test_primitives_follow_changes_to_their_files(template_sets):
    template = demo()
    path = template_sets / "demo" / "primitives" / "text" / "01-bullets.tex"
    read_at = path.stat().st_mtime_ns
    assert template.primitives()[0].label == "Bullets"

    path.write_text(f"% Points: renamed\n{PRIMITIVE_FRAME}\n", encoding="utf-8")
    os.utime(path, ns=(read_at, read_at + 1_000_000_000))

    assert template.primitives()[0].label == "Points"


def test_every_shipped_primitive_parses():
    for template in filter(lambda template: template.is_presentation, latex_templates.template_sets()):
        files = list((template.path / "primitives").glob("*/*.tex"))
        assert len(template.primitives()) == len(files) > 0
        assert {primitive.group for primitive in template.primitives()} <= {group.key for group in template.groups}


def test_templates_lists_the_primitives_of_a_presentation_root_only(tex_client, vault, tmp_path, template_sets):
    write_files(vault, {"deck/main.tex": PRESENTATION_SOURCE, "paper/main.tex": ROOT_SOURCE, "notes.md": ""})
    write_files(tmp_path, {"outside.tex": PRESENTATION_SOURCE})

    def templates(path: Path):
        return tex_client.get("/api/latex/templates", params={"root": str(path)})

    body = templates(vault / "deck" / "main.tex").json()
    assert body["theme"] == "demo"
    assert [primitive["id"] for primitive in body["primitives"]][:2] == ["text/bullets", "text/statement"]
    assert body["primitives"][0] == {
        "id": "text/bullets", "group": "text", "group_label": "Text", "label": "Bullets",
        "description": "points with one message", "summary": "Points with one message", "text": PRIMITIVE_FRAME,
        "preview": preview_version(template_sets),
    }
    assert templates(vault / "paper" / "main.tex").json() == {"theme": None, "primitives": []}
    assert templates(tmp_path / "outside.tex").status_code == 403
    assert templates(vault / "notes.md").status_code == 400
    assert templates(vault / "missing.tex").status_code == 404


def test_template_sets_route_lists_names_and_labels(tex_client, template_sets):
    assert tex_client.get("/api/latex/template-sets").json() == [
        {"name": "compact", "label": "Compact"}, {"name": "demo", "label": "Demo deck"}, {"name": "letter", "label": "Letter"},
    ]


def test_every_shipped_primitive_has_a_preview():
    for template in latex_templates.template_sets():
        assert all(primitive.preview is not None for primitive in template.primitives())


def test_a_primitives_preview_is_served_with_a_long_private_cache(tex_client, template_sets):
    response = tex_client.get("/api/latex/templates/demo/previews/text/bullets.png", params={"v": 1})

    assert response.status_code == 200
    assert response.content == PREVIEW_PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"


@pytest.mark.parametrize("path", [
    "demo/previews/text/statement.png",
    "demo/previews/text/headless.png",
    "demo/previews/text/01-bullets.png",
    "demo/previews/numbers/bullets.png",
    "letter/previews/text/bullets.png",
    "Madrid/previews/text/bullets.png",
    "demo/previews/text/..%2F..%2Fsecret.png",
    "demo/previews/%2E%2E/secret.png",
    "..%2Fsets%2Fdemo/previews/text/bullets.png",
    "%2E%2E/previews/text/bullets.png",
])
def test_previews_of_unknown_sets_or_primitives_missing_ones_and_other_files_answer_404(tex_client, template_sets, path):
    assert tex_client.get(f"/api/latex/templates/{path}").status_code == 404
