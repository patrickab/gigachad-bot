/**
 * A draft `factory.yaml` from what a repo already declares: a pure function of the checkout's files, no model call.
 *
 * - `draftFactoryYaml(files, template)` fills the commented template (`factory.template.yaml`) with `gates` (fast: lint,
 *   types, format) and `merge_gates` (everything, tests included): an argv per gate, judged by exit code (0 pass, 75
 *   infra, else fail). Nothing detected leaves the template as it is
 * - Sources, the highest-level one wins: `factory/gates.sh`, Makefile targets, `.gitlab-ci.yml` scripts, then the single
 *   tool calls (`.pre-commit-config.yaml`, `package.json` scripts, `pyproject.toml`) together
 * - GitLab jobs that deploy, publish, build images or need CI-only variables are skipped, and so are install lines
 * - `repoFiles(root)` reads a checkout from disk, `initDraft(root)` is the whole thing with the shipped template
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export type Argv = string[];

/** The files of a checkout: `read` is undefined for a missing file, `has` is true for a file or a directory. */
export interface RepoFiles {
  read(path: string): string | undefined;
  has(path: string): boolean;
}

export interface Draft {
  /** The YAML text. */
  text: string;
  /** The files the gates come from, comma separated; empty when nothing was detected. */
  source: string;
  gates: Argv[];
  mergeGates: Argv[];
}

interface Found {
  source: string;
  gates: Argv[];
  mergeGates: Argv[];
}

export const GATES_SCRIPT = "factory/gates.sh";

const PLAIN = /^[A-Za-z0-9_./=:@%+-]+$/;
/** Plain scalars a YAML 1.1 reader (the backend's) turns into a bool, null or number. */
const NOT_A_STRING = /^(y|n|yes|no|on|off|true|false|null|~|=|<<)$|^[-+.]?\d|^[-?:]$|^[@%]|:$/i;
const SHELL_SYNTAX = /&&|\|\||[|;<>$`(){}*?\n]/;

const CI_RESERVED = new Set(["stages", "variables", "default", "include", "workflow", "image", "services", "cache", "before_script", "after_script", "pages"]);
const LINT_STAGE = /lint|check|static|format|style|quality|analy|type|validate|verify|sast/i;
const TEST_STAGE = /test|unit|e2e|integration|spec/i;
const DEPLOY_STAGE = /deploy|publish|release|package|build|image|docker|pages/i;
const DEPLOY_LINE = /docker\s+(build|push|login)|kaniko|buildah|helm\b|kubectl|deploy|publish|twine\s+upload|git\s+push|terraform\s+apply|\brelease\b/i;
const CI_VARIABLE = /\$\{?(CI_\w+|\w*(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIALS?)\w*)/i;
const SETUP_LINE = /^(sudo\s+)?(pip3?\s+install|python3?\s+-m\s+pip\s+install|npm\s+(ci|install)|yarn\s+install|pnpm\s+install|apt(-get)?\s|apk\s|uv\s+sync|poetry\s+install|bundle\s+install)/;

const NODE_RUNNERS: [string, string][] = [["pnpm-lock.yaml", "pnpm"], ["yarn.lock", "yarn"], ["bun.lock", "bun"], ["bun.lockb", "bun"]];
const NODE_LINT = ["lint", "format:check", "check-format", "prettier:check"];
const NODE_TYPES = ["typecheck", "type-check", "check-types", "tsc"];
const NPM_DEFAULT_TEST = "no test specified";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unique(commands: Argv[]): Argv[] {
  const seen: Argv[] = [];
  for (const command of commands) if (!seen.some((other) => JSON.stringify(other) === JSON.stringify(command))) seen.push(command);
  return seen;
}

function found(source: string, gates: Argv[], tests: Argv[] = [], everything?: Argv[]): Found {
  const quick = unique(gates);
  return { source, gates: quick, mergeGates: unique(everything ?? [...quick, ...tests]) };
}

function tryParse<T>(parse: () => T): T | undefined {
  try {
    return parse();
  } catch {
    return undefined;
  }
}

/** POSIX word splitting (Python's `shlex.split`), undefined for an unbalanced quote or a trailing backslash. */
function shlexSplit(line: string): string[] | undefined {
  const words: string[] = [];
  let word = "";
  let inWord = false;
  for (let index = 0; index < line.length; index++) {
    const char = line[index]!;
    if (char === "'" || char === '"') {
      const end = (() => {
        for (let at = index + 1; at < line.length; at++) {
          if (char === '"' && line[at] === "\\" && '"\\'.includes(line[at + 1] ?? "")) at++;
          else if (line[at] === char) return at;
        }
        return -1;
      })();
      if (end < 0) return undefined;
      const body = line.slice(index + 1, end);
      word += char === '"' ? body.replace(/\\(["\\])/g, "$1") : body;
      inWord = true;
      index = end;
    } else if (char === "\\") {
      if (index + 1 >= line.length) return undefined;
      word += line[++index];
      inWord = true;
    } else if (/\s/.test(char)) {
      if (inWord) words.push(word);
      word = "";
      inWord = false;
    } else {
      word += char;
      inWord = true;
    }
  }
  if (inWord) words.push(word);
  return words;
}

function scriptArgv(line: string): Argv {
  const words = SHELL_SYNTAX.test(line) ? undefined : shlexSplit(line);
  return words && words.length > 0 ? words : ["bash", "-c", line];
}

function scriptLines(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(scriptLines);
  return [];
}

function gateScript(files: RepoFiles): Found | undefined {
  if (!files.has(GATES_SCRIPT)) return undefined;
  const base = ["bash", GATES_SCRIPT, "--changed", "--reuse"];
  return found(GATES_SCRIPT, [[...base, "lint", "types"]], [], [base]);
}

function makefile(files: RepoFiles): Found | undefined {
  const text = files.read("Makefile");
  if (text === undefined) return undefined;
  const targets = new Set([...text.matchAll(/^([A-Za-z0-9_.-]+)\s*:(?!=)/gm)].map((match) => match[1]!));
  const lintTargets = ["lint", "typecheck"].filter((name) => targets.has(name));
  const quick = lintTargets.length > 0 ? lintTargets : targets.has("check") ? ["check"] : [];
  const tests = targets.has("test") ? ["test"] : [];
  if (quick.length === 0 && tests.length === 0) return undefined;
  return found("Makefile", quick.map((name) => ["make", name]), tests.map((name) => ["make", name]));
}

function gitlabCi(files: RepoFiles): Found | undefined {
  const text = files.read(".gitlab-ci.yml");
  const document = text ? tryParse(() => Bun.YAML.parse(text)) : undefined;
  if (!isRecord(document)) return undefined;
  const order = Array.isArray(document.stages) ? document.stages.filter((stage): stage is string => typeof stage === "string") : [];
  const jobs: { rank: number; stage: string; lines: string[] }[] = [];
  for (const [name, job] of Object.entries(document)) {
    if (name.startsWith(".") || CI_RESERVED.has(name) || !isRecord(job)) continue;
    const lines = scriptLines(job.script);
    const stage = job.stage ?? "test";
    if (lines.length === 0 || typeof stage !== "string" || job.when === "manual" || "environment" in job) continue;
    jobs.push({ rank: order.includes(stage) ? order.indexOf(stage) : order.length, stage, lines });
  }
  const quick: Argv[] = [];
  const tests: Argv[] = [];
  for (const { stage, lines } of jobs.sort((a, b) => a.rank - b.rank)) {
    if (DEPLOY_STAGE.test(stage) || lines.some((line) => DEPLOY_LINE.test(line) || CI_VARIABLE.test(line))) continue;
    const target = LINT_STAGE.test(stage) ? quick : TEST_STAGE.test(stage) ? tests : undefined;
    for (const line of lines.map((line) => line.trim())) if (target && line && !SETUP_LINE.test(line)) target.push(scriptArgv(line));
  }
  return quick.length > 0 || tests.length > 0 ? found(".gitlab-ci.yml", quick, tests) : undefined;
}

function packageJson(files: RepoFiles): [Argv[], Argv[]] {
  const text = files.read("package.json");
  const document = text ? tryParse(() => JSON.parse(text) as unknown) : undefined;
  const scripts = isRecord(document) ? document.scripts : undefined;
  if (!isRecord(scripts)) return [[], []];
  const runner = NODE_RUNNERS.find(([lockfile]) => files.has(lockfile))?.[1] ?? "npm";
  const quick = [...NODE_LINT, ...NODE_TYPES].filter((name) => typeof scripts[name] === "string").map((name) => [runner, "run", name]);
  const test = scripts.test;
  return [quick, typeof test === "string" && !test.includes(NPM_DEFAULT_TEST) ? [[runner, "run", "test"]] : []];
}

function pyproject(files: RepoFiles, preCommit: boolean): [Argv[], Argv[]] {
  const text = files.read("pyproject.toml");
  const document = text === undefined ? undefined : tryParse(() => Bun.TOML.parse(text) as unknown);
  if (text === undefined || !isRecord(document)) return [[], []];
  const tool = isRecord(document.tool) ? document.tool : {};
  const prefix = files.has("uv.lock") ? ["uv", "run"] : [];
  const quick: Argv[] = [];
  if ("ruff" in tool && !preCommit) {
    quick.push([...prefix, "ruff", "check", "."]);
    if (isRecord(tool.ruff) && "format" in tool.ruff) quick.push([...prefix, "ruff", "format", "--check", "."]);
  }
  if ("mypy" in tool) quick.push([...prefix, "mypy", "."]);
  else if ("pyright" in tool) quick.push([...prefix, "pyright"]);
  const declared = "pytest" in tool || /["'\s]pytest\b/.test(text);
  return [quick, declared && ("pytest" in tool || files.has("tests")) ? [[...prefix, "pytest", "-q"]] : []];
}

function tools(files: RepoFiles): Found | undefined {
  const preCommit = files.has(".pre-commit-config.yaml");
  const sources = preCommit ? ["pre-commit"] : [];
  const quick: Argv[] = preCommit ? [["pre-commit", "run", "--all-files"]] : [];
  const tests: Argv[] = [];
  for (const [source, [foundQuick, foundTests]] of [["package.json", packageJson(files)], ["pyproject.toml", pyproject(files, preCommit)]] as const) {
    if (foundQuick.length > 0 || foundTests.length > 0) sources.push(source);
    quick.push(...foundQuick);
    tests.push(...foundTests);
  }
  return quick.length > 0 || tests.length > 0 ? found(sources.join(", "), quick, tests) : undefined;
}

function scalar(item: string): string {
  return PLAIN.test(item) && !NOT_A_STRING.test(item) ? item : JSON.stringify(item);
}

function argvLines(commands: Argv[]): string {
  return commands.map((command) => `  - [${command.map(scalar).join(", ")}]\n`).join("");
}

/** The gates the highest-level source in `files` declares. */
export function detectGates(files: RepoFiles): Found {
  for (const source of [gateScript, makefile, gitlabCi, tools]) {
    const draft = source(files);
    if (draft) return draft;
  }
  return { source: "", gates: [], mergeGates: [] };
}

/** `template` (a `factory.yaml` with `gates: []` and `merge_gates: []`) filled with what `files` declare. */
export function draftFactoryYaml(files: RepoFiles, template: string): Draft {
  const { source, gates, mergeGates } = detectGates(files);
  if (gates.length === 0 && mergeGates.length === 0) return { text: template, source: "", gates, mergeGates };
  const fill = (text: string, key: string, commands: Argv[]) =>
    commands.length === 0 ? text : text.replace(new RegExp(`^${key}: \\[\\]$`, "m"), () => `${key}:\n${argvLines(commands).trimEnd()}`);
  const filled = fill(fill(template, "gates", gates), "merge_gates", mergeGates);
  const note = `# Drafted from ${source}. Review it, then commit it.\n`;
  return { text: note + filled, source, gates, mergeGates };
}

export function repoFiles(root: string): RepoFiles {
  return {
    read: (path) => tryParse(() => readFileSync(join(root, path), "utf-8")),
    has: (path) => existsSync(join(root, path)),
  };
}

export const TEMPLATE_PATH = join(import.meta.dir, "factory.template.yaml");

/** The draft for the checkout at `root`, from the template the plugin ships. */
export function initDraft(root: string): Draft {
  return draftFactoryYaml(repoFiles(root), readFileSync(TEMPLATE_PATH, "utf-8"));
}
