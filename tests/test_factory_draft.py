"""The factory.yaml draft: each source, the precedence between them and what it skips."""

import json
from pathlib import Path

import yaml

from lib.factory_draft import draft_factory_yaml


def draft(root: Path) -> dict:
    parsed = yaml.safe_load(draft_factory_yaml(root))
    assert set(parsed) == {"hooks", "gates", "merge_gates"}
    return parsed


def write(root: Path, name: str, text: str) -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)


def test_nothing_found_is_an_empty_draft_with_a_comment_per_key(tmp_path):
    text = draft_factory_yaml(tmp_path)
    assert draft(tmp_path)["gates"] == [] and draft(tmp_path)["merge_gates"] == []
    lines = text.splitlines()
    for key in ("hooks:", "gates:", "merge_gates:"):
        position = next(index for index, line in enumerate(lines) if line.startswith(key))
        assert lines[position - 1].startswith("#")
    assert "preview:" not in text


def test_a_gate_script_wins_over_everything(tmp_path):
    write(tmp_path, "factory/gates.sh", "")
    write(tmp_path, "Makefile", "test:\n\ttrue\n")
    write(tmp_path, ".gitlab-ci.yml", "t:\n  script: [pytest]\n")
    result = draft(tmp_path)
    assert result["gates"] == [["bash", "factory/gates.sh", "--changed", "--reuse", "lint", "types"]]
    assert result["merge_gates"] == [["bash", "factory/gates.sh", "--changed", "--reuse"]]


def test_makefile_targets_beat_gitlab_ci_and_tools(tmp_path):
    write(tmp_path, "Makefile", "lint:\n\truff\ntypecheck: deps\n\tmypy\ncheck:\n\tx\ntest:\n\tpytest\nVAR := 1\n")
    write(tmp_path, ".gitlab-ci.yml", "t:\n  script: [pytest]\n")
    result = draft(tmp_path)
    assert result["gates"] == [["make", "lint"], ["make", "typecheck"]]
    assert result["merge_gates"] == [["make", "lint"], ["make", "typecheck"], ["make", "test"]]


def test_a_check_target_stands_in_for_lint_and_typecheck(tmp_path):
    write(tmp_path, "Makefile", "check:\n\tx\n")
    assert draft(tmp_path)["gates"] == [["make", "check"]]


def test_gitlab_jobs_are_grouped_by_stage_and_the_rest_skipped(tmp_path):
    write(
        tmp_path,
        ".gitlab-ci.yml",
        """
stages: [build, test, lint, deploy]
variables: {A: b}
.template:
  script: [rm -rf /]
image-job:
  stage: build
  script: [docker build .]
unit:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest -q
    - cd app && make test
style:
  stage: lint
  script: [ruff check .]
ship:
  stage: deploy
  script: [./deploy.sh]
secret-lint:
  stage: lint
  script: [curl -H "$CI_JOB_TOKEN" x]
publish-in-test:
  stage: test
  script: [twine upload dist/*]
manual:
  stage: test
  when: manual
  script: [pytest --slow]
""",
    )
    result = draft(tmp_path)
    assert result["gates"] == [["ruff", "check", "."]]
    assert result["merge_gates"] == [["ruff", "check", "."], ["pytest", "-q"], ["bash", "-c", "cd app && make test"]]


def test_gitlab_ci_with_nothing_usable_falls_through_to_the_tools(tmp_path):
    write(tmp_path, ".gitlab-ci.yml", "ship:\n  stage: deploy\n  script: [./deploy.sh]\n")
    write(tmp_path, "package.json", json.dumps({"scripts": {"lint": "eslint ."}}))
    assert draft(tmp_path)["gates"] == [["npm", "run", "lint"]]


def test_package_json_scripts_use_the_lockfile_runner_and_skip_the_npm_default_test(tmp_path):
    default_test = 'echo "Error: no test specified" && exit 1'
    write(tmp_path, "package.json", json.dumps({"scripts": {"lint": "x", "typecheck": "y", "test": default_test}}))
    write(tmp_path, "pnpm-lock.yaml", "")
    result = draft(tmp_path)
    assert result["gates"] == [["pnpm", "run", "lint"], ["pnpm", "run", "typecheck"]]
    assert result["merge_gates"] == result["gates"]
    write(tmp_path, "package.json", json.dumps({"scripts": {"test": "vitest run"}}))
    assert draft(tmp_path)["merge_gates"] == [["pnpm", "run", "test"]]


def test_pyproject_tools_and_pre_commit(tmp_path):
    write(tmp_path, "pyproject.toml", "[tool.ruff]\n[tool.ruff.format]\n[tool.mypy]\n[tool.pytest.ini_options]\n")
    result = draft(tmp_path)
    assert result["gates"] == [["ruff", "check", "."], ["ruff", "format", "--check", "."], ["mypy", "."]]
    assert result["merge_gates"][-1] == ["pytest", "-q"]

    write(tmp_path, ".pre-commit-config.yaml", "repos: []\n")
    write(tmp_path, "uv.lock", "")
    result = draft(tmp_path)
    assert result["gates"] == [["pre-commit", "run", "--all-files"], ["uv", "run", "mypy", "."]]


def test_tools_of_both_languages_combine(tmp_path):
    write(tmp_path, "package.json", json.dumps({"scripts": {"test": "vitest"}}))
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\n")
    assert draft(tmp_path)["merge_gates"] == [["npm", "run", "test"], ["pytest", "-q"]]


def test_arguments_that_yaml_would_misread_are_quoted(tmp_path):
    write(tmp_path, ".gitlab-ci.yml", "t:\n  stage: lint\n  script:\n    - echo on\n    - 'echo \"a: b\"'\n")
    assert draft(tmp_path)["gates"] == [["echo", "on"], ["echo", "a: b"]]
    assert '[echo, "on"]' in draft_factory_yaml(tmp_path)
