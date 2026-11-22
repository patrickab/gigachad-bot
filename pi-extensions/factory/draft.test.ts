import { afterAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { draftFactoryYaml, initDraft, repoFiles, TEMPLATE_PATH } from "./draft.ts";

const dirs: string[] = [];
afterAll(() => dirs.forEach((dir) => rmSync(dir, { recursive: true, force: true })));

function repo(files: Record<string, string> = {}): string {
  const root = mkdtempSync(join(tmpdir(), "factory-draft-"));
  dirs.push(root);
  for (const [name, text] of Object.entries(files)) {
    mkdirSync(dirname(join(root, name)), { recursive: true });
    writeFileSync(join(root, name), text);
  }
  return root;
}

function draft(files: Record<string, string>) {
  const text = initDraft(repo(files)).text;
  const parsed = Bun.YAML.parse(text) as { hooks: unknown; gates: string[][]; merge_gates: string[][] };
  expect(Object.keys(parsed).sort()).toEqual(["gates", "hooks", "merge_gates"]);
  return parsed;
}

describe("draftFactoryYaml", () => {
  test("nothing found leaves the template as it is, with a comment per key", () => {
    const template = readFileSync(TEMPLATE_PATH, "utf-8");
    const result = initDraft(repo());
    expect(result.text).toBe(template);
    expect(result.source).toBe("");
    const lines = template.split("\n");
    for (const key of ["hooks:", "gates:", "merge_gates:"]) {
      expect(lines[lines.findIndex((line) => line.startsWith(key)) - 1]!.startsWith("#")).toBe(true);
    }
    expect(template).not.toContain("preview:");
  });

  test("works on any file source, not only the disk", () => {
    const files = { read: (path: string) => (path === "Makefile" ? "test:\n\ttrue\n" : undefined), has: (path: string) => path === "Makefile" };
    const result = draftFactoryYaml(files, "gates: []\nmerge_gates: []\n");
    expect(result.text).toBe("# Drafted from Makefile. Review it, then commit it.\ngates: []\nmerge_gates:\n  - [make, test]\n");
  });

  test("a gate script wins over everything", () => {
    const result = draft({ "factory/gates.sh": "", Makefile: "test:\n\ttrue\n", ".gitlab-ci.yml": "t:\n  script: [pytest]\n" });
    expect(result.gates).toEqual([["bash", "factory/gates.sh", "--changed", "--reuse", "lint", "types"]]);
    expect(result.merge_gates).toEqual([["bash", "factory/gates.sh", "--changed", "--reuse"]]);
  });

  test("makefile targets beat gitlab ci and tools", () => {
    const result = draft({
      Makefile: "lint:\n\truff\ntypecheck: deps\n\tmypy\ncheck:\n\tx\ntest:\n\tpytest\nVAR := 1\n",
      ".gitlab-ci.yml": "t:\n  script: [pytest]\n",
    });
    expect(result.gates).toEqual([["make", "lint"], ["make", "typecheck"]]);
    expect(result.merge_gates).toEqual([["make", "lint"], ["make", "typecheck"], ["make", "test"]]);
  });

  test("a check target stands in for lint and typecheck", () => {
    expect(draft({ Makefile: "check:\n\tx\n" }).gates).toEqual([["make", "check"]]);
  });

  test("gitlab jobs are grouped by stage and the rest skipped", () => {
    const result = draft({
      ".gitlab-ci.yml": `
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
`,
    });
    expect(result.gates).toEqual([["ruff", "check", "."]]);
    expect(result.merge_gates).toEqual([["ruff", "check", "."], ["pytest", "-q"], ["bash", "-c", "cd app && make test"]]);
  });

  test("gitlab ci with nothing usable falls through to the tools", () => {
    const result = draft({
      ".gitlab-ci.yml": "ship:\n  stage: deploy\n  script: [./deploy.sh]\n",
      "package.json": JSON.stringify({ scripts: { lint: "eslint ." } }),
    });
    expect(result.gates).toEqual([["npm", "run", "lint"]]);
  });

  test("package.json scripts use the lockfile runner and skip the npm default test", () => {
    const defaultTest = 'echo "Error: no test specified" && exit 1';
    const files = { "pnpm-lock.yaml": "", "package.json": JSON.stringify({ scripts: { lint: "x", typecheck: "y", test: defaultTest } }) };
    const result = draft(files);
    expect(result.gates).toEqual([["pnpm", "run", "lint"], ["pnpm", "run", "typecheck"]]);
    expect(result.merge_gates).toEqual(result.gates);
    expect(draft({ ...files, "package.json": JSON.stringify({ scripts: { test: "vitest run" } }) }).merge_gates).toEqual([["pnpm", "run", "test"]]);
  });

  test("pyproject tools and pre-commit", () => {
    const pyproject = "[tool.ruff]\n[tool.ruff.format]\n[tool.mypy]\n[tool.pytest.ini_options]\n";
    const result = draft({ "pyproject.toml": pyproject });
    expect(result.gates).toEqual([["ruff", "check", "."], ["ruff", "format", "--check", "."], ["mypy", "."]]);
    expect(result.merge_gates.at(-1)).toEqual(["pytest", "-q"]);
    expect(draft({ "pyproject.toml": pyproject, ".pre-commit-config.yaml": "repos: []\n", "uv.lock": "" }).gates).toEqual([
      ["pre-commit", "run", "--all-files"],
      ["uv", "run", "mypy", "."],
    ]);
  });

  test("tools of both languages combine and name their sources", () => {
    const files = { "package.json": JSON.stringify({ scripts: { test: "vitest" } }), "pyproject.toml": "[tool.pytest.ini_options]\n" };
    expect(draft(files).merge_gates).toEqual([["npm", "run", "test"], ["pytest", "-q"]]);
    expect(initDraft(repo(files)).source).toBe("package.json, pyproject.toml");
  });

  test("arguments that yaml would misread are quoted", () => {
    const files = { ".gitlab-ci.yml": "t:\n  stage: lint\n  script:\n    - echo on\n    - 'echo \"a: b\"'\n" };
    expect(draft(files).gates).toEqual([["echo", "on"], ["echo", "a: b"]]);
    expect(initDraft(repo(files)).text).toContain('[echo, "on"]');
  });

  test("repoFiles reads a checkout", () => {
    const files = repoFiles(repo({ "a/b.txt": "x" }));
    expect([files.read("a/b.txt"), files.read("nope"), files.has("a"), files.has("nope")]).toEqual(["x", undefined, true, false]);
  });
});
