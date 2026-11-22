"""A draft `factory.yaml` from what a repo already declares: a pure function of the checkout's files, no model call.

- `draft_factory_yaml(root)` returns commented YAML with `hooks`, `gates` (fast: lint, types, format) and `merge_gates`
  (everything, tests included). An argv per gate, judged by exit code (0 pass, 75 infra, else fail)
- Sources, the highest-level one wins: `factory/gates.sh`, Makefile targets, `.gitlab-ci.yml` scripts, then the single
  tool calls (`.pre-commit-config.yaml`, `package.json` scripts, `pyproject.toml`) together
- GitLab jobs that deploy, publish, build images or need CI-only variables are skipped, and so are install lines
- Nothing found: every key stays empty, with its comment
"""

import json
from pathlib import Path
import re
import shlex
import tomllib

import yaml

Argv = list[str]

GATES_SCRIPT = "factory/gates.sh"

_PLAIN = re.compile(r"[A-Za-z0-9_./=:@%+-]+")
_SHELL_SYNTAX = re.compile(r"&&|\|\||[|;<>$`(){}*?\n]")

_CI_RESERVED = {
    "stages", "variables", "default", "include", "workflow", "image", "services", "cache", "before_script",
    "after_script", "pages",
}  # fmt: skip
_LINT_STAGE = re.compile(r"lint|check|static|format|style|quality|analy|type|validate|verify|sast", re.I)
_TEST_STAGE = re.compile(r"test|unit|e2e|integration|spec", re.I)
_DEPLOY_STAGE = re.compile(r"deploy|publish|release|package|build|image|docker|pages", re.I)
_DEPLOY_LINE = re.compile(
    r"docker\s+(build|push|login)|kaniko|buildah|helm\b|kubectl|deploy|publish|twine\s+upload|git\s+push|terraform\s+apply"
    r"|\brelease\b",
    re.I,
)
_CI_VARIABLE = re.compile(r"\$\{?(CI_\w+|\w*(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIALS?)\w*)", re.I)
_SETUP_LINE = re.compile(
    r"^(sudo\s+)?(pip3?\s+install|python3?\s+-m\s+pip\s+install|npm\s+(ci|install)|yarn\s+install|pnpm\s+install"
    r"|apt(-get)?\s|apk\s|uv\s+sync|poetry\s+install|bundle\s+install)"
)

_NODE_RUNNERS = {"pnpm-lock.yaml": "pnpm", "yarn.lock": "yarn", "bun.lock": "bun", "bun.lockb": "bun"}
_NODE_LINT = ("lint", "format:check", "check-format", "prettier:check")
_NODE_TYPES = ("typecheck", "type-check", "check-types", "tsc")
_NPM_DEFAULT_TEST = "no test specified"


class Draft:
    """The two lists of argv a draft proposes and where they came from."""

    def __init__(
        self,
        source: str = "",
        gates: list[Argv] | None = None,
        tests: list[Argv] | None = None,
        everything: list[Argv] | None = None,
    ) -> None:
        self.source = source
        self.gates: list[Argv] = _unique(gates or [])
        self.merge_gates: list[Argv] = _unique(everything if everything is not None else self.gates + (tests or []))


def _unique(commands: list[Argv]) -> list[Argv]:
    seen: list[Argv] = []
    for command in commands:
        if command not in seen:
            seen.append(command)
    return seen


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _gate_script(root: Path) -> Draft | None:
    if not (root / GATES_SCRIPT).is_file():
        return None
    base = ["bash", GATES_SCRIPT, "--changed", "--reuse"]
    return Draft(GATES_SCRIPT, [[*base, "lint", "types"]], everything=[base])


def _makefile(root: Path) -> Draft | None:
    text = _read(root / "Makefile")
    if text is None:
        return None
    targets = set(re.findall(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", text, re.M))
    quick = [name for name in ("lint", "typecheck") if name in targets] or (["check"] if "check" in targets else [])
    tests = ["test"] if "test" in targets else []
    if not quick and not tests:
        return None
    return Draft("Makefile", [["make", name] for name in quick], [["make", name] for name in tests])


def _script_argv(line: str) -> Argv:
    if not _SHELL_SYNTAX.search(line):
        try:
            return shlex.split(line)
        except ValueError:
            pass
    return ["bash", "-c", line]


def _script_lines(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [line for item in value for line in _script_lines(item)]
    return []


def _gitlab_ci(root: Path) -> Draft | None:
    text = _read(root / ".gitlab-ci.yml")
    try:
        document = yaml.safe_load(text) if text else None
    except yaml.YAMLError:
        return None
    if not isinstance(document, dict):
        return None
    stages = document.get("stages")
    order = [stage for stage in stages if isinstance(stage, str)] if isinstance(stages, list) else []
    jobs = []
    for name, job in document.items():
        if not isinstance(name, str) or name.startswith(".") or name in _CI_RESERVED or not isinstance(job, dict):
            continue
        lines = _script_lines(job.get("script"))
        stage = job.get("stage", "test")
        if not lines or not isinstance(stage, str) or job.get("when") == "manual" or "environment" in job:
            continue
        jobs.append((order.index(stage) if stage in order else len(order), stage, lines))
    quick: list[Argv] = []
    tests: list[Argv] = []
    for _, stage, lines in sorted(jobs, key=lambda job: job[0]):
        if _DEPLOY_STAGE.search(stage) or any(_DEPLOY_LINE.search(line) or _CI_VARIABLE.search(line) for line in lines):
            continue
        target = quick if _LINT_STAGE.search(stage) else tests if _TEST_STAGE.search(stage) else None
        if target is not None:
            target.extend(_script_argv(line.strip()) for line in lines if line.strip() and not _SETUP_LINE.match(line.strip()))
    if not quick and not tests:
        return None
    return Draft(".gitlab-ci.yml", quick, tests)


def _package_json(root: Path) -> tuple[list[Argv], list[Argv]]:
    text = _read(root / "package.json")
    try:
        scripts = json.loads(text).get("scripts") if text else None
    except (json.JSONDecodeError, AttributeError):
        return [], []
    if not isinstance(scripts, dict):
        return [], []
    runner = next((tool for lockfile, tool in _NODE_RUNNERS.items() if (root / lockfile).exists()), "npm")

    def run(name: str) -> Argv:
        return [runner, "run", name]

    quick = [run(name) for name in (*_NODE_LINT, *_NODE_TYPES) if isinstance(scripts.get(name), str)]
    test = scripts.get("test")
    tests = [run("test")] if isinstance(test, str) and _NPM_DEFAULT_TEST not in test else []
    return quick, tests


def _pyproject(root: Path, pre_commit: bool) -> tuple[list[Argv], list[Argv]]:
    text = _read(root / "pyproject.toml")
    if text is None:
        return [], []
    try:
        tool = tomllib.loads(text).get("tool", {})
    except tomllib.TOMLDecodeError:
        return [], []
    prefix = ["uv", "run"] if (root / "uv.lock").exists() else []

    def declared(name: str) -> bool:
        return name in tool or re.search(rf"[\"'\s]{name}\b", text) is not None

    quick: list[Argv] = []
    if "ruff" in tool and not pre_commit:
        quick.append([*prefix, "ruff", "check", "."])
        if "format" in tool["ruff"]:
            quick.append([*prefix, "ruff", "format", "--check", "."])
    if "mypy" in tool:
        quick.append([*prefix, "mypy", "."])
    elif "pyright" in tool:
        quick.append([*prefix, "pyright"])
    tests = [[*prefix, "pytest", "-q"]] if "pytest" in tool or (declared("pytest") and (root / "tests").is_dir()) else []
    return quick, tests


def _tools(root: Path) -> Draft | None:
    pre_commit = (root / ".pre-commit-config.yaml").is_file()
    sources = ["pre-commit"] if pre_commit else []
    quick: list[Argv] = [["pre-commit", "run", "--all-files"]] if pre_commit else []
    tests: list[Argv] = []
    found = (("package.json", _package_json(root)), ("pyproject.toml", _pyproject(root, pre_commit)))
    for source, (found_quick, found_tests) in found:
        if found_quick or found_tests:
            sources.append(source)
        quick += found_quick
        tests += found_tests
    return Draft(", ".join(sources), quick, tests) if quick or tests else None


def draft_for(root: Path) -> Draft:
    """The gates the highest-level source in *root* declares, an empty `Draft` when it has none."""
    for found in (_gate_script, _makefile, _gitlab_ci, _tools):
        draft = found(root)
        if draft is not None:
            return draft
    return Draft()


def _scalar(item: str) -> str:
    plain = _PLAIN.fullmatch(item) and isinstance(yaml.safe_load(item), str)
    return item if plain else json.dumps(item)


def _argv_lines(commands: list[Argv]) -> str:
    return "".join(f"  - [{', '.join(_scalar(item) for item in command)}]\n" for command in commands)


def draft_factory_yaml(root: Path) -> str:
    """The commented `factory.yaml` for the checkout at *root*."""
    draft = draft_for(root)
    found = f"Drafted from {draft.source}." if draft.source else "Nothing found to draft from: fill the gates in."
    gates = _argv_lines(draft.gates)
    merge_gates = _argv_lines(draft.merge_gates)
    return (
        f"# What the agent factory needs to know about this repo. {found} Review it, then commit it.\n"
        "# A gate is one argv list, run in the task's worktree and judged by its exit code: 0 passes, 75 means the\n"
        "# environment is broken (the task blocks as `infra`), anything else fails. Print a summary of the failure\n"
        "# (at most 50 lines) on stdout: that is all the coder gets back. Call the same commands from the repo's CI.\n"
        "#\n"
        "# Commands run in the task's worktree: after_create once it exists, after_code after every coder turn, and\n"
        "# after_merge in the main checkout after a merge that changed a lockfile.\n"
        "hooks:\n"
        "  after_create: []\n"
        "  after_code: []\n"
        "  after_merge: []\n"
        "# Fast checks (lint, types, format) after every coder turn.\n"
        f"gates:{' []' if not gates else chr(10) + gates.rstrip(chr(10))}\n"
        "# Everything, tests included: run in the merge queue on the rebased branch, before the fast-forward.\n"
        f"merge_gates:{' []' if not merge_gates else chr(10) + merge_gates.rstrip(chr(10))}\n"
    )
